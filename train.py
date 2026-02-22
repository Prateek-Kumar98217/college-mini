import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from torch.amp import autocast, GradScaler
import numpy as np
from tqdm import tqdm

from model import EGHGT, EGHGTConfig, MultiTaskLoss
from sklearn.metrics import accuracy_score, f1_score


# ==========================================
# 1. Dataset Handler
# ==========================================
class ProcessedMELDDataset(Dataset):
    def __init__(self, pt_file):
        self.data = torch.load(pt_file, weights_only=False)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            "text": torch.tensor(item["text_features"], dtype=torch.float32).squeeze(),
            "audio": torch.tensor(
                item["audio_features"], dtype=torch.float32
            ).squeeze(),
            "video": torch.tensor(
                item["video_features"], dtype=torch.float32
            ).squeeze(),
            "label_sent": torch.tensor(item["sentiment"], dtype=torch.long),
            "label_emo": torch.tensor(item["emotion"], dtype=torch.long),
        }


# ==========================================
# 2. Evaluation Helper
# ==========================================
@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0

    all_senti_preds = []
    all_senti_labels = []
    all_emo_preds = []
    all_emo_labels = []

    for batch in loader:
        t, a, v = (
            batch["text"].to(device),
            batch["audio"].to(device),
            batch["video"].to(device),
        )
        l_sent, l_emo = batch["label_sent"].to(device), batch["label_emo"].to(device)

        with autocast(device_type="cuda"):
            out = model(t, a, v)
            loss = criterion(out["emotions"], out["sentiment"], l_emo, l_sent)

        total_loss += loss.item()

        senti_preds = torch.argmax(out["sentiment"], dim=1)
        emo_preds = torch.argmax(out["emotions"], dim=1)

        # Move to CPU and append to our tracking lists
        all_senti_preds.extend(senti_preds.cpu().numpy())
        all_senti_labels.extend(l_sent.cpu().numpy())
        all_emo_preds.extend(emo_preds.cpu().numpy())
        all_emo_labels.extend(l_emo.cpu().numpy())

    avg_loss = total_loss / len(loader)

    # Calculate Metrics
    senti_acc = accuracy_score(all_senti_labels, all_senti_preds)
    emo_acc = accuracy_score(all_emo_labels, all_emo_preds)

    senti_f1 = f1_score(all_senti_labels, all_senti_preds, average="weighted")
    emo_f1 = f1_score(all_emo_labels, all_emo_preds, average="weighted")

    return avg_loss, senti_acc, emo_acc, senti_f1, emo_f1


# ==========================================
# 3. The Core Training Loop
# ==========================================
def train_model():
    # --- Setup ---
    config = EGHGTConfig(batch_size=32)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Initializing Training on {device}")

    # Load Distinct Preprocessed Data Files
    print("Loading preprocessed data...")
    train_data = ProcessedMELDDataset("./data/MELD.Raw/train_features.pt")
    val_data = ProcessedMELDDataset("./data/MELD.Raw/val_features.pt")
    test_data = ProcessedMELDDataset("./data/MELD.Raw/test_features.pt")

    print(len(train_data), "training samples loaded.")
    print(len(val_data), "validation samples loaded.")
    print(len(test_data), "test samples loaded.")

    # Static Loaders for Validation and Testing
    val_loader = DataLoader(val_data, batch_size=config.batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=config.batch_size, shuffle=False)

    epochs = 25
    test_start_epoch = 15

    # Initialize Model, Optimizer, Loss, and Scaler
    model = EGHGT(config).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.05)
    criterion = MultiTaskLoss().to(device)
    scaler = GradScaler()

    # Cosine Annealing Learning Rate Scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_loss = float("inf")

    print("\n" + "=" * 40)
    print("STARTING TRAINING PIPELINE")
    print("=" * 40)

    for epoch in range(epochs):
        # --- 70% Dynamic Subsampling ---
        current_train_size = int(0.75 * len(train_data))
        epoch_train_idx = np.random.choice(
            len(train_data), current_train_size, replace=False
        )

        train_loader = DataLoader(
            train_data,
            batch_size=config.batch_size,
            sampler=SubsetRandomSampler(epoch_train_idx),
        )

        # --- Training Phase ---
        model.train()
        train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [TRAIN]")

        for batch in pbar:
            t, a, v = (
                batch["text"].to(device),
                batch["audio"].to(device),
                batch["video"].to(device),
            )
            l_sent, l_emo = batch["label_sent"].to(device), batch["label_emo"].to(
                device
            )

            optimizer.zero_grad()

            # Automatic Mixed Precision (AMP) Forward Pass
            with autocast(device_type="cuda"):
                out = model(t, a, v)
                loss = criterion(out["emotions"], out["sentiment"], l_emo, l_sent)

            # AMP Backward Pass
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=1.0
            )  # Prevent exploding gradients
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        scheduler.step()

        # --- Validation Phase ---
        val_loss, val_senti_acc, val_emo_acc, val_senti_f1, val_emo_f1 = evaluate(
            model, val_loader, criterion, device
        )
        print(
            f"Val Loss: {val_loss:.4f} | Senti Acc: {val_senti_acc:.2%} (F1: {val_senti_f1:.4f}) | Emo Acc: {val_emo_acc:.2%} (F1: {val_emo_f1:.4f})"
        )

        # --- Checkpointing ---
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "eghgt_best_model.pt")
            print(f"Checkpoint saved! (Val Loss improved to {best_val_loss:.4f})")

        # --- Test Phase (After Epoch 10) ---
        if epoch + 1 >= test_start_epoch:
            test_loss, test_senti_acc, test_emo_acc, test_senti_f1, test_emo_f1 = (
                evaluate(model, test_loader, criterion, device)
            )
            print(
                f"Test Loss: {test_loss:.4f} | Senti Acc: {test_senti_acc:.2%} (F1: {test_senti_f1:.4f}) | Emo Acc: {test_emo_acc:.2%} (F1: {test_emo_f1:.4f})"
            )
        torch.cuda.empty_cache()


if __name__ == "__main__":
    train_model()
