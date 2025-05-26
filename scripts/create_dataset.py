import pandas as pd
import os
import glob

def create_dataset_csv(base_dir, csv_dir, output_csv):
    data = []
    csv_files = {
        "original": "original.csv",
        "Deepfakes": "Deepfakes.csv",
        "Face2Face": "Face2Face.csv",
        "FaceShifter": "FaceShifter.csv",
        "FaceSwap": "FaceSwap.csv",
        "NeuralTextures": "NeuralTextures.csv"
    }
    
    for label, csv_file in csv_files.items():
        csv_path = os.path.join(csv_dir, csv_file)
        if os.path.exists(csv_path):
            df_meta = pd.read_csv(csv_path)
            for img_path in glob.glob(os.path.join(base_dir, label, "*/*.jpg")):
                video_name = os.path.basename(img_path).split('_frame_')[0] + ".mp4"
                if not df_meta.empty and video_name in df_meta["File Path"].values:
                    label_id = df_meta[df_meta["File Path"] == video_name]["Label"].iloc[0]
                    data.append([img_path, label_id])
                else:
                    print(f"Warning: No metadata for {video_name}")
    
    df = pd.DataFrame(data, columns=["image_path", "label"])
    df = df.sample(frac=1).reset_index(drop=True)
    df.to_csv(output_csv, index=False)
    print(f"Created dataset CSV with {len(df)} samples")

if __name__ == "__main__":
    base_dir = "data/resized_frames"  # Hoặc "data/augmented_frames"
    csv_dir = "data/raw_videos/csv"
    create_dataset_csv(base_dir, csv_dir, "data/dataset.csv")