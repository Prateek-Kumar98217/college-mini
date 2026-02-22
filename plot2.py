"""Script to generate and save both confusion matrix(emotion and sentiment) for the EGHG-T model on the MELD test set."""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader

from model import EGHGT, EGHGTConfig
from train import ProcessedMELDDataset


def plot_publication_confusion_matrices(model_path, test_pt_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = EGHGTConfig()

    model = EGHGT(config).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    test_data = ProcessedMELDDataset(test_pt_path)
    test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

    all_emo_preds, all_emo_labels = [], []
    all_senti_preds, all_senti_labels = [], []

    print("Generating predictions for confusion matrices...")
    with torch.no_grad():
        for batch in test_loader:
            t = batch["text"].to(device)
            a = batch["audio"].to(device)
            v = batch["video"].to(device)

            l_emo = batch["label_emo"].cpu().numpy()
            l_sent = batch["label_sent"].cpu().numpy()

            with torch.amp.autocast(device_type="cuda"):
                out = model(t, a, v)

            emo_preds = torch.argmax(out["emotions"], dim=1).cpu().numpy()
            senti_preds = torch.argmax(out["sentiment"], dim=1).cpu().numpy()

            all_emo_preds.extend(emo_preds)
            all_emo_labels.extend(l_emo)

            all_senti_preds.extend(senti_preds)
            all_senti_labels.extend(l_sent)

    emotions = ["Neutral", "Surprise", "Fear", "Sadness", "Joy", "Disgust", "Anger"]
    sentiments = ["Neutral", "Positive", "Negative"]

    cm_emo = confusion_matrix(all_emo_labels, all_emo_preds)
    cm_emo_norm = cm_emo.astype("float") / cm_emo.sum(axis=1)[:, np.newaxis]

    cm_senti = confusion_matrix(all_senti_labels, all_senti_preds)
    cm_senti_norm = cm_senti.astype("float") / cm_senti.sum(axis=1)[:, np.newaxis]

    fig, axes = plt.subplots(1, 2, figsize=(16, 7), dpi=300)
    sns.set_theme(style="white")

    sns.heatmap(
        cm_emo_norm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=emotions,
        yticklabels=emotions,
        cbar_kws={"label": "Prediction Probability"},
        ax=axes[0],
        annot_kws={"size": 11},
    )

    axes[0].set_title("Emotion Classification (EGHG-T)", fontsize=16, pad=15)
    axes[0].set_ylabel("True Emotion", fontsize=14, weight="bold")
    axes[0].set_xlabel("Predicted Emotion", fontsize=14, weight="bold")
    axes[0].tick_params(axis="x", rotation=45)

    sns.heatmap(
        cm_senti_norm,
        annot=True,
        fmt=".2f",
        cmap="Greens",
        xticklabels=sentiments,
        yticklabels=sentiments,
        cbar_kws={"label": "Prediction Probability"},
        ax=axes[1],
        annot_kws={"size": 11},
    )

    axes[1].set_title("Sentiment Classification (EGHG-T)", fontsize=16, pad=15)
    axes[1].set_ylabel("True Sentiment", fontsize=14, weight="bold")
    axes[1].set_xlabel("Predicted Sentiment", fontsize=14, weight="bold")
    axes[1].tick_params(axis="x", rotation=45)

    plt.tight_layout()
    plt.savefig("eghgt_dual_confusion_matrices.png", format="png", bbox_inches="tight")
    print("Saved dual confusion matrices to 'eghgt_dual_confusion_matrices.png'")
    plt.show()


if __name__ == "__main__":
    plot_publication_confusion_matrices(
        "eghgt_best_model.pt",
        "/home/detxonr/Documents/projects/research/data/MELD.Raw/test_features.pt",
    )
