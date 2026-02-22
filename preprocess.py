import os
import pandas as pd
import torch
import librosa
from transformers import (
    DistilBertModel,
    DistilBertTokenizer,
    HubertModel,
    Wav2Vec2FeatureExtractor,
    VideoMAEModel,
    VideoMAEImageProcessor,
)
import cv2
import numpy as np
from tqdm import tqdm


class MELDFeatureExtractor:
    def __init__(self, device="cuda"):
        self.device = device
        # Load Models & Processors
        self.tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
        self.text_model = DistilBertModel.from_pretrained("distilbert-base-uncased").to(
            device
        )

        self.audio_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
            "facebook/hubert-base-ls960"
        )
        self.audio_model = HubertModel.from_pretrained("facebook/hubert-base-ls960").to(
            device
        )

        self.video_processor = VideoMAEImageProcessor.from_pretrained(
            "MCG-NJU/videomae-base"
        )
        self.video_model = VideoMAEModel.from_pretrained("MCG-NJU/videomae-base").to(
            device
        )

    @torch.no_grad()
    def get_text_feat(self, text):
        inputs = self.tokenizer(
            text, return_tensors="pt", padding=True, truncation=True
        ).to(self.device)
        outputs = self.text_model(**inputs)
        # Last hidden state: [Batch, Seq, 768] -> Mean Pool to [768]
        return outputs.last_hidden_state.mean(dim=1).cpu().numpy()

    @torch.no_grad()
    def get_audio_feat(self, audio_path):
        wav, _ = librosa.load(audio_path, sr=16000)
        inputs = self.audio_extractor(wav, sampling_rate=16000, return_tensors="pt").to(
            self.device
        )
        outputs = self.audio_model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).cpu().numpy()

    @torch.no_grad()
    def get_video_feat(self, video_path):
        # Sample 16 frames uniformly
        cap = cv2.VideoCapture(video_path)
        frames = []
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        indices = np.linspace(0, total - 1, 16).astype(int)
        for i in range(total):
            ret, frame = cap.read()
            if ret and i in indices:
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()

        # Ensure 16 frames
        while len(frames) < 16:
            frames.append(np.zeros((224, 224, 3), dtype=np.uint8))

        inputs = self.video_processor(list(frames), return_tensors="pt").to(self.device)
        outputs = self.video_model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).cpu().numpy()


def preprocess_meld(csv_path, video_dir, save_path):
    df = pd.read_csv(csv_path)
    extractor = MELDFeatureExtractor()
    processed_data = []

    # Mappings for Labels
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

    for _, row in tqdm(df.iterrows(), total=len(df)):
        vid_filename = f"dia{row['Dialogue_ID']}_utt{row['Utterance_ID']}.mp4"
        vid_path = os.path.join(video_dir, vid_filename)

        if os.path.exists(vid_path):
            try:
                # Extract Features
                t_feat = extractor.get_text_feat(row["Utterance"])
                a_feat = extractor.get_audio_feat(
                    vid_path
                )  # Extracted from mp4 directly
                v_feat = extractor.get_video_feat(vid_path)

                processed_data.append(
                    {
                        "text_features": t_feat,
                        "audio_features": a_feat,
                        "video_features": v_feat,
                        "sentiment": sent_map[row["Sentiment"]],
                        "emotion": emo_map[row["Emotion"]],
                    }
                )
            except Exception as e:
                print(f"Error processing {vid_filename}: {e}")

    # Save as a consolidated file
    torch.save(processed_data, save_path)
