import torch
import torch.nn as nn
import torch.optim as optim
import torch.functional as F
from torch.amp import autocast, grad_scaler
from .model import WhiteSharkOptimizer, EGHGTConfig, EGHGTransformer
from .preprocess import get_dataloader


# --- 1. Multi-Task Loss (SEJO Stage) ---
class MultiTaskLoss(nn.Module):
    def __init__(self, num_tasks=2):
        super().__init__()
        # Learnable log-variance to balance task weights automatically
        self.log_vars = nn.Parameter(torch.zeros(num_tasks))

    def forward(self, pred_sent, target_sent, pred_emo, target_emo):
        # Task 1: Sentiment Regression (MSE)
        loss_sent = F.mse_loss(pred_sent.squeeze(), target_sent.squeeze())
        precision1 = torch.exp(-self.log_vars[0])
        weighted_loss1 = precision1 * loss_sent + self.log_vars[0]

        # Task 2: Emotion Classification (Cross Entropy)
        loss_emo = F.cross_entropy(pred_emo, target_emo)
        precision2 = torch.exp(-self.log_vars[1])
        weighted_loss2 = precision2 * loss_emo + self.log_vars[1]

        return weighted_loss1 + weighted_loss2


# --- 2. The Training Function ---
def train_one_epoch(model, loader, optimizer, criterion, scaler, device):
    model.train()
    total_loss = 0
    for batch in loader:
        # 1. Prepare data
        t, a, v = (
            batch["text"].to(device),
            batch["audio"].to(device),
            batch["video"].to(device),
        )
        label_sent = batch["label_sentiment"].to(device)
        label_emo = batch["label_emotion"].to(device)

        # 2. Forward pass with AMP
        optimizer.zero_grad()
        with autocast(device_type="cuda"):
            output = model(t, a, v)
            loss = criterion(
                output["sentiment"], label_sent, output["emotions"], label_emo
            )

        # 3. Backward & Step
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
    return total_loss / len(loader)


# --- 3. Main Execution Block ---
def main():
    # Setup
    config = EGHGTConfig()  # From our previous steps
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize components
    model = EGHGTransformer(config).to(device)
    shark_manager = WhiteSharkOptimizer(model, config)  # Binary WSO Manager

    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
    criterion = MultiTaskLoss().to(device)
    scaler = grad_scaler()

    # Data (Assuming your pre-extracted features)
    train_loader = get_dataloader(train_data, batch_size=32)
    val_loader = get_dataloader(val_data, batch_size=32)

    epochs = 50
    best_loss = float("inf")

    print(f"Starting EGHG-T Training on {device}...")

    for epoch in range(epochs):
        # A. Train base weights
        avg_train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, scaler, device
        )

        # B. "The Lunge": White Shark Evolutionary Feature Selection
        # Start pruning features after model stability (e.g., epoch 5)
        if epoch >= 5:
            print(f"[WSO] Senses Active: Evolving Binary Mask...")
            # evaluate_fitness tests masks on validation set
            fitness_scores = shark_manager.evaluate_fitness(val_loader, device)
            best_mask = shark_manager.update_masks(fitness_scores)

            # Apply evolved mask to the static Feature Selector layer
            model.feature_selector.best_mask.copy_(best_mask)

        scheduler.step()

        print(f"Epoch {epoch+1:02d} | Train Loss: {avg_train_loss:.4f}")

        # C. Checkpointing
        if avg_train_loss < best_loss:
            best_loss = avg_train_loss
            torch.save(model.state_dict(), "best_eghgt_model.pt")

    print("Success: EGHG-T trained with Evolutionary Feature Selection.")


if __name__ == "__main__":
    main()
