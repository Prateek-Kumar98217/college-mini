import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass


class WSOFeatureSelector(nn.Module):
    """
    Implements the selection logic for the EGHG-T Stage 4.
    This layer applies a learned binary mask to the fused features.
    """

    def __init__(self, feature_dim):
        super().__init__()
        self.feature_dim = feature_dim
        # The 'Best Shark' mask: 1 means keep feature, 0 means drop
        self.register_buffer("best_mask", torch.ones(feature_dim))

    def forward(self, x):
        # x: [Batch, feature_dim]
        return x * self.best_mask


class WhiteSharkOptimizer:
    def __init__(self, model, config, population_size=10, mutation_rate=0.1):
        self.model = model
        self.pop_size = population_size
        self.mutation_rate = mutation_rate
        self.dim = config.d_model
        # Initialize population of binary masks
        self.population = torch.randint(0, 2, (population_size, self.dim)).float()
        self.best_shark = self.population[0]
        self.best_fitness = -float("inf")

    def update_masks(self, fitness_scores):
        """
        Mimics shark movement and mutation to update the feature masks.
        """
        # 1. Selection: Find the current best
        current_best_idx = torch.argmax(fitness_scores)
        if fitness_scores[current_best_idx] > self.best_fitness:
            self.best_fitness = fitness_scores[current_best_idx]
            self.best_shark = self.population[current_best_idx].clone()

        # 2. Movement & Mutation (Evolution)
        for i in range(self.pop_size):
            # Move toward best shark (Crossover-like)
            r1 = torch.rand(self.dim)
            move_condition = r1 < 0.5
            self.population[i] = torch.where(
                move_condition, self.best_shark, self.population[i]
            )

            # Mutation (Random lunge)
            mutation_mask = torch.rand(self.dim) < self.mutation_rate
            self.population[i][mutation_mask] = 1 - self.population[i][mutation_mask]

        return self.best_shark

    @torch.no_grad()
    def evaluate_fitness(self, val_loader, device):
        """
        Fitness is defined as (1 / Validation Loss) or Validation Accuracy.
        """
        self.model.eval()
        fitness_scores = []

        for mask in self.population:
            # Temporarily apply mask to model
            self.model.feature_selector.best_mask = mask.to(device)

            # Run a small validation subset to get score
            # (Simplified for brevity: assume a function get_val_score exists)
            score = self._get_val_score(val_loader, device)
            fitness_scores.append(score)

        return torch.tensor(fitness_scores)

    def _get_val_score(self, loader, device):
        # Placeholder for actual validation logic
        return np.random.random()  # Replace with actual Acc or 1/Loss


@dataclass
class EGHGTConfig:
    # Feature Dimensions (Output from DistilBERT, HuBERT, VideoMAE)
    d_text: int = 768
    d_audio: int = 768
    d_video: int = 768

    # Model Hyperparameters
    d_model: int = 256
    n_heads: int = 8
    num_blocks: int = 4
    dropout: float = 0.1

    # Stage 1: DUC
    low_rank_k: int = 32

    # Stage 4: Task parameters
    num_emotions: int = 6  # MELD/MOSI specific
    sentiment_dim: int = 1


class DUCModule(nn.Module):
    """Stage 1: Disentangled Unimodal Compression using Low-Rank Rényi Entropy."""

    def __init__(self, in_dim, out_dim, rank_k):
        super().__init__()
        self.projection = nn.Linear(in_dim, out_dim)
        self.rank_k = rank_k
        self.layer_norm = nn.LayerNorm(out_dim)

    def forward(self, x):
        # x: [batch, seq_len, in_dim]
        z = self.projection(x)

        # Simplified Low-Rank Bottleneck (approximating principal patterns)
        # We perform a SVD-like compression to retain only top-K patterns
        u, s, v = torch.pca_lowrank(z, q=self.rank_k)
        z_compressed = torch.matmul(z, v[:, :, : self.rank_k])
        z_reconstructed = torch.matmul(
            z_compressed, v[:, :, : self.rank_k].transpose(-1, -2)
        )

        return self.layer_norm(z_reconstructed)


class CrossModalAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.mha = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.gate = nn.Sequential(nn.Linear(d_model * 2, 1), nn.Sigmoid())
        self.norm = nn.LayerNorm(d_model)

    def forward(self, query, key_value):
        attn_out, _ = self.mha(query, key_value, key_value)
        # Gating mechanism to prevent modality dominance
        g = self.gate(torch.cat([query, attn_out], dim=-1))
        return self.norm(query + g * attn_out)


class NRMEModule(nn.Module):
    """Stage 3: Noise-Resistant Modality-Aware Editing."""

    def __init__(self, d_model, num_blocks=4):
        super().__init__()
        self.num_blocks = num_blocks
        self.weight_gen = nn.Sequential(
            nn.AdaptiveAvgPool1d(1), nn.Linear(d_model, d_model), nn.Sigmoid()
        )

    def forward(self, x):
        # x: [B, L, D] -> Block partitioning
        b, l, d = x.shape
        block_size = l // self.num_blocks

        # Calculate weights for sub-blocks to denoise
        refined_blocks = []
        for i in range(self.num_blocks):
            block = x[:, i * block_size : (i + 1) * block_size, :]
            w = self.weight_gen(block.transpose(1, 2)).transpose(1, 2)
            refined_blocks.append(block * w)

        return torch.cat(refined_blocks, dim=1)


class EGHGTransformer(nn.Module):
    def __init__(self, config: EGHGTConfig):
        super().__init__()

        # Stage 1: DUC
        self.duc_t = DUCModule(config.d_text, config.d_model, config.low_rank_k)
        self.duc_a = DUCModule(config.d_audio, config.d_model, config.low_rank_k)
        self.duc_v = DUCModule(config.d_video, config.d_model, config.low_rank_k)

        # Stage 2: HSCMA (Hierarchical Alignment)
        self.align_ta = CrossModalAttention(config.d_model, config.n_heads)
        self.align_tv = CrossModalAttention(config.d_model, config.n_heads)
        self.align_av = CrossModalAttention(config.d_model, config.n_heads)

        # Stage 3: NRME (Denoising)
        self.denoiser = NRMEModule(config.d_model)

        # Global Fusion Transformer
        self.fusion_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=config.d_model, nhead=config.n_heads, batch_first=True
            ),
            num_layers=2,
        )

        self.feature_selector = WSOFeatureSelector(config.d_model)

        # Stage 4: SEJO (Multi-task Heads)
        self.sentiment_head = nn.Sequential(
            nn.Linear(config.d_model, config.d_model // 2),
            nn.ReLU(),
            nn.Linear(config.d_model // 2, config.sentiment_dim),
        )

        self.emotion_head = nn.Sequential(
            nn.Linear(config.d_model, config.d_model // 2),
            nn.ReLU(),
            nn.Linear(config.d_model // 2, config.num_emotions),
        )

    def forward(self, text_feats, audio_feats, video_feats):
        # Stage 1: Compression
        zt = self.duc_t(text_feats)
        za = self.duc_a(audio_feats)
        zv = self.duc_v(video_feats)

        # Stage 2: Hierarchical Cross-Modal Alignment
        # Round 1: Text guides Audio and Video
        za_aligned = self.align_ta(za, zt)
        zv_aligned = self.align_tv(zv, zt)

        # Round 2: Bidirectional Audio-Video interaction
        z_av = self.align_av(za_aligned, zv_aligned)

        # Stage 3: Denoising
        z_dn = self.denoiser(z_av)

        # Global Fusion
        global_context = self.fusion_transformer(z_dn)
        pooled_feat = torch.mean(global_context, dim=1)

        selected_feat = self.feature_selector(pooled_feat)

        # Stage 4: SEJO (Joint Optimization)
        sentiment = self.sentiment_head(selected_feat)
        emotions = self.emotion_head(selected_feat)

        return {"sentiment": sentiment, "emotions": emotions}
