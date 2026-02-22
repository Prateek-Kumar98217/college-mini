from preprocess import preprocess_meld

if __name__ == "__main__":
    preprocess_meld(
        csv_path="/home/detxonr/Documents/projects/research/data/MELD.Raw/train_sent_emo.csv",
        video_dir="/home/detxonr/Documents/projects/research/data/MELD.Raw/train_splits",
        save_path="/home/detxonr/Documents/projects/research/data/MELD.Raw/train_features.pt",
    )
