"""Script to generate and save a confusion matrix(emotion) for the EGHG-T model on the MELD test set."""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader
from model import (
    EGHGT,
    EGHGTConfig,
)
from train import ProcessedMELDDataset


def plot_publication_confusion_matrix(model_path, test_pt_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = EGHGTConfig()

    model = EGHGT(config).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    test_data = ProcessedMELDDataset(test_pt_path)
    test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

    all_preds = []
    all_labels = []

    print("Generating predictions for confusion matrix...")
    with torch.no_grad():
        for batch in test_loader:
            t = batch["text"].to(device)
            a = batch["audio"].to(device)
            v = batch["video"].to(device)
            labels = batch["label_emo"].cpu().numpy()

            with torch.amp.autocast(device_type="cuda"):
                out = model(t, a, v)

            preds = torch.argmax(out["emotions"], dim=1).cpu().numpy()

            all_preds.extend(preds)
            all_labels.extend(labels)

    emotions = ["Neutral", "Surprise", "Fear", "Sadness", "Joy", "Disgust", "Anger"]
    cm = confusion_matrix(all_labels, all_preds)

    cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(10, 8), dpi=300)
    sns.set_theme(style="white")

    ax = sns.heatmap(
        cm_normalized,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=emotions,
        yticklabels=emotions,
        cbar_kws={"label": "Prediction Probability"},
        annot_kws={"size": 12},
    )

    plt.title("Normalized Emotion Confusion Matrix (EGHG-T)", fontsize=16, pad=20)
    plt.ylabel("True Emotion", fontsize=14, weight="bold")
    plt.xlabel("Predicted Emotion", fontsize=14, weight="bold")

    plt.xticks(rotation=45, ha="right", fontsize=11)
    plt.yticks(rotation=0, fontsize=11)

    plt.tight_layout()
    plt.savefig("eghgt_confusion_matrix.png", format="png", bbox_inches="tight")
    print("Saved confusion matrix to 'eghgt_confusion_matrix.png'")
    plt.show()


if __name__ == "__main__":
    plot_publication_confusion_matrix(
        "eghgt_best_model.pt",
        "/home/detxonr/Documents/projects/research/data/MELD.Raw/test_features.pt",
    )
