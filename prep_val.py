from preprocess import preprocess_meld

if __name__ == "__main__":
    preprocess_meld(
        csv_path="/home/detxonr/Documents/projects/research/data/MELD.Raw/dev_sent_emo.csv",
        video_dir="/home/detxonr/Documents/projects/research/data/MELD.Raw/dev_splits_complete",
        save_path="/home/detxonr/Documents/projects/research/data/MELD.Raw/val_features.pt",
    )
