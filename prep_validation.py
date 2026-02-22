import torch
import random
import numpy as np

# Load the processed file
data_path = "data/MELD.Raw/val_features.pt"  # Update to your filename
data = torch.load(data_path, weights_only=False)

print(f"Total samples processed: {len(data)}")

# 1. Check a random sample
for _ in range(5):
    sample = random.choice(data)
    print("\n--- Random Sample Audit ---")
    for key, value in sample.items():
        if isinstance(value, (torch.Tensor, np.ndarray)):
            print(f"{key:15} | Shape: {value.shape} | Type: {type(value)}")
            print(f"Sample values (first 5): {value.flatten()[:5]}")
        else:
            print(f"{key:15} | Value: {value}")

# 2. Bulk Shape Validation
print("\n--- Shape Consistency Check ---")
expected_shapes = {
    "text_features": (1, 768),  # DistilBERT pooled output
    "audio_features": (1, 768),  # HuBERT hidden state mean
    "video_features": (1, 768),  # VideoMAE hidden state mean
}

issues = 0
for i, item in enumerate(data):
    for feat_name, expected in expected_shapes.items():
        if item[feat_name].shape != expected:
            print(
                f"[FATAL ERROR]Shape Mismatch at index {i}: {feat_name} is {item[feat_name].shape}"
            )
            issues += 1
            if issues > 5:
                break  # Don't flood the console
    if issues > 5:
        break

if issues == 0:
    print("[STATUS SUCCESS]All feature shapes are consistent!")
