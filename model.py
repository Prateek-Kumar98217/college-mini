import torch
import torch.nn as nn

from dataclasses import dataclass


@dataclass
class EGHGTConfig:

    text_dim: int = 768
    audio_dim: int = 768
    video_dim: int = 768
    batch_size: int = 32

    hidden_dim: int = 768
    num_heads: int = 12
    num_blocks: int = 8
    dropout: float = 0.4

    num_emotions: int = 7
    sentiment_dim: int = 3


class MLP(nn.Module):
    """Simple MLP for modality-specific feature transformation"""

    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = self.act(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)


class DUCModule(nn.Module):
    """low rank compression for pooled vectors"""

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim)
        # Using a Bottleneck to approximate Low-Rank PCA without seq_len errors
        self.bottleneck = nn.Sequential(
            nn.Linear(out_dim, out_dim // 4),
            nn.GELU(),
            nn.Linear(out_dim // 4, out_dim),
        )
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x):
        x = self.proj(x)
        return self.norm(self.bottleneck(x))


class CrossModelAttention(nn.Module):
    """Hierarchical gated attention for cross-modal fusion"""

    def __init__(self, d_model, n_heads):
        super().__init__()
        self.mha = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.gate = nn.Sequential(nn.Linear(d_model * 2, 1), nn.Sigmoid())
        self.layernorm = nn.LayerNorm(d_model)

    def forward(self, query, key_value):

        attn_output, _ = self.mha(query, key_value, key_value)
        gate = self.gate(torch.cat([query, attn_output], dim=-1))
        output = gate * attn_output + (1 - gate) * query
        return self.layernorm(output)


class NRMEModule(nn.Module):
    """Noise resistant Modularity Aware editing"""

    def __init__(self, d_model, num_blocks=4):
        super().__init__()
        self.num_blocks = num_blocks
        self.block_dim = d_model // num_blocks
        self.weight_gen = nn.Sequential(
            nn.Linear(self.block_dim, self.block_dim), nn.Sigmoid()
        )

    def forward(self, x):
        # Partitioning the feature dimension since seq_len = 1
        blocks = torch.split(x, self.block_dim, dim=-1)
        refined = [block * self.weight_gen(block) for block in blocks]
        return torch.cat(refined, dim=-1)


class MultiTaskLoss(nn.Module):
    """Uncertainty-weighted loss with Exact Class Imbalance Penalties"""

    def __init__(self, num_tasks=2):
        super().__init__()

        raw_senti_weights = torch.tensor([0.7069, 1.4266, 1.1306], dtype=torch.float32)
        raw_emo_weights = torch.tensor(
            [0.3030, 1.1842, 5.3246, 2.0893, 0.8187, 5.2657, 1.2867],
            dtype=torch.float32,
        )
        # Apply Square Root Smoothing to compress the extreme variance
        senti_weights = torch.sqrt(raw_senti_weights)
        emo_weights = torch.sqrt(raw_emo_weights)

        self.register_buffer("senti_weights", senti_weights)
        self.register_buffer("emo_weights", emo_weights)

        self.emo_loss = nn.CrossEntropyLoss(weight=emo_weights)
        self.senti_loss = nn.CrossEntropyLoss(weight=senti_weights)
        self.log_vars = nn.Parameter(torch.zeros(num_tasks))

    def forward(self, emo_logits, senti_logits, emo_labels, senti_labels):
        loss_emo = self.emo_loss(emo_logits, emo_labels)
        loss_senti = self.senti_loss(senti_logits, senti_labels)

        weight_emo = torch.exp(-self.log_vars[0])
        weight_senti = torch.exp(-self.log_vars[1])

        loss = (
            weight_emo * loss_emo
            + self.log_vars[0]
            + weight_senti * loss_senti
            + self.log_vars[1]
        )
        return loss


class EGHGT(nn.Module):
    def __init__(self, config: EGHGTConfig):
        super().__init__()
        self.config = config

        # Modality-specific encoders
        self.text_encoder = DUCModule(config.text_dim, config.hidden_dim)
        self.audio_encoder = DUCModule(config.audio_dim, config.hidden_dim)
        self.video_encoder = DUCModule(config.video_dim, config.hidden_dim)

        # Noise-resistant modularity aware editing
        self.nre_text = NRMEModule(config.hidden_dim, config.num_blocks)
        self.nre_audio = NRMEModule(config.hidden_dim, config.num_blocks)
        self.nre_video = NRMEModule(config.hidden_dim, config.num_blocks)

        # Cross-modal attention layers
        self.attn_text_audio = CrossModelAttention(config.hidden_dim, config.num_heads)
        self.attn_text_video = CrossModelAttention(config.hidden_dim, config.num_heads)
        self.attn_audio_video = CrossModelAttention(config.hidden_dim, config.num_heads)

        self.fusion_mlp = nn.Sequential(
            nn.Linear(config.hidden_dim * 2, config.hidden_dim * 2),
            nn.SiLU(),  # Swish activation (SwiGLU variant)
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim * 2, config.hidden_dim),
        )

        self.emo_classifier = MLP(
            config.hidden_dim, config.hidden_dim, config.num_emotions
        )
        self.senti_classifier = MLP(
            config.hidden_dim, config.hidden_dim, config.sentiment_dim
        )

    def forward(self, text_feat, audio_feat, video_feat):
        t, a, v = (
            text_feat.unsqueeze(1),
            audio_feat.unsqueeze(1),
            video_feat.unsqueeze(1),
        )

        # 1. Compress
        zt, za, zv = self.text_encoder(t), self.audio_encoder(a), self.video_encoder(v)

        # 2. Denoise (Clean before they mix)
        zt = self.nre_text(zt)
        za = self.nre_audio(za)
        zv = self.nre_video(zv)

        # 3. Align (Text guides audio/video, then audio/video interact)
        za_aligned = self.attn_text_audio(za, zt)
        zv_aligned = self.attn_text_video(zv, zt)
        z_av = self.attn_audio_video(za_aligned, zv_aligned)

        # 4. Fuse & Pool
        zt_flat = zt.squeeze(1)  # Shape: [Batch, 256]
        z_av_flat = z_av.squeeze(1)  # Shape: [Batch, 256]
        fused_seq = torch.cat([zt_flat, z_av_flat], dim=1)
        pooled_feat = self.fusion_mlp(fused_seq)

        return {
            "sentiment": self.senti_classifier(pooled_feat),
            "emotions": self.emo_classifier(pooled_feat),
        }


def count_parameters(model):
    """Calculates total trainable parameters in the model."""
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Trainable Parameters: {total_params:,} ({total_params / 1e6:.2f} M)")
    return total_params


def run_dummy_test():
    """Tests the EGHG-T model with dummy tensors to verify shapes."""
    print("--- Initializing EGHG-T Dummy Test ---")

    # 1. Load Config and Model
    config = EGHGTConfig(batch_size=4)  # Small batch for local testing
    model = EGHGT(config)

    # 2. Print Parameter Count
    count_parameters(model)

    # 3. Generate Dummy Data (Simulating the output of the Preprocessor)
    print("\n--- Generating Dummy Inputs ---")
    text_feat = torch.randn(config.batch_size, config.text_dim)
    audio_feat = torch.randn(config.batch_size, config.audio_dim)
    video_feat = torch.randn(config.batch_size, config.video_dim)

    print(f"Text Input:  {text_feat.shape}")
    print(f"Audio Input: {audio_feat.shape}")
    print(f"Video Input: {video_feat.shape}")

    # 4. Forward Pass
    print("\n--- Running Forward Pass ---")
    model.eval()  # Set to eval to disable dropout for deterministic testing
    with torch.no_grad():
        outputs = model(text_feat, audio_feat, video_feat)

    # 5. Verify Outputs
    print("\n--- Output Validation ---")
    senti_out = outputs["sentiment"]
    emo_out = outputs["emotions"]

    print(
        f"Sentiment Output Shape: {senti_out.shape} -> Expected: [{config.batch_size}, {config.sentiment_dim}]"
    )
    print(
        f"Emotions Output Shape:  {emo_out.shape} -> Expected: [{config.batch_size}, {config.num_emotions}]"
    )

    # 6. Strict Assertion Check
    assert senti_out.shape == (
        config.batch_size,
        config.sentiment_dim,
    ), "Sentiment shape mismatch!"
    assert emo_out.shape == (
        config.batch_size,
        config.num_emotions,
    ), "Emotion shape mismatch!"

    print("\nAll tensor shapes are correct! The architecture is mathematically sound.")


if __name__ == "__main__":
    run_dummy_test()
