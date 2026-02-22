import pandas as pd
import numpy as np
from sklearn.utils.class_weight import compute_class_weight


def calculate_exact_weights(train_csv_path):
    print(f"Loading {train_csv_path}...")
    df = pd.read_csv(train_csv_path)

    # 1. Exact mappings used during your preprocessing
    emo_map = {
        "neutral": 0,
        "surprise": 1,
        "fear": 2,
        "sadness": 3,
        "joy": 4,
        "disgust": 5,
        "anger": 6,
    }
    sent_map = {"neutral": 0, "positive": 1, "negative": 2}

    # 2. Map string labels to integers and drop any weird NaNs
    y_emo = df["Emotion"].map(emo_map).dropna().astype(int)
    y_sent = df["Sentiment"].map(sent_map).dropna().astype(int)

    # 3. Calculate 'balanced' weights
    emo_classes = np.unique(y_emo)
    emo_weights = compute_class_weight(
        class_weight="balanced", classes=emo_classes, y=y_emo
    )

    sent_classes = np.unique(y_sent)
    sent_weights = compute_class_weight(
        class_weight="balanced", classes=sent_classes, y=y_sent
    )

    # 4. Print the exact PyTorch tensors to paste into your model
    print("\n" + "=" * 50)
    print("PASTE THESE INTO YOUR MultiTaskLoss CLASS:")
    print("=" * 50)

    # Formatting nicely for copy-pasting
    emo_str = ", ".join([f"{w:.4f}" for w in emo_weights])
    sent_str = ", ".join([f"{w:.4f}" for w in sent_weights])

    print(f"senti_weights = torch.tensor([{sent_str}], dtype=torch.float32)")
    print(f"emo_weights = torch.tensor([{emo_str}], dtype=torch.float32)")
    print("=" * 50)


if __name__ == "__main__":
    # Update this path if your CSV is located somewhere else
    calculate_exact_weights("./data/MELD.Raw/train_sent_emo.csv")
