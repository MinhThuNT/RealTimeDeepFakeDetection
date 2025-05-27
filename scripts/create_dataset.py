import pandas as pd
import os
import glob

def create_dataset_csv(base_dir, csv_dir, output_csv, use_augmented=False):
    data = []
    csv_files = {
        "original": "original.csv",
        "Deepfakes": "Deepfakes.csv",
        "Face2Face": "Face2Face.csv",
        "FaceShifter": "FaceShifter.csv",
        "FaceSwap": "FaceSwap.csv",
        "NeuralTextures": "NeuralTextures.csv"
    }
    
    # Xử lý video thật
    real_dir = os.path.join(base_dir, "real")
    if os.path.exists(real_dir):
        csv_path = os.path.join(csv_dir, csv_files["original"])
        if os.path.exists(csv_path):
            try:
                df_meta = pd.read_csv(csv_path)
                if "File Path" not in df_meta.columns or "Label" not in df_meta.columns:
                    print(f"Error: {csv_path} missing required columns (File Path, Label)")
                    return
                for img_path in glob.glob(os.path.join(real_dir, "*/*.jpg")):
                    # Xử lý tiền tố nếu dùng ảnh tăng cường
                    base_name = os.path.basename(img_path)
                    if use_augmented and base_name.startswith("aug_"):
                        base_name = "_".join(base_name.split("_")[2:])  # Bỏ "aug_0_" hoặc "aug_1_"
                    video_name = base_name.split('_frame_')[0] + ".mp4"
                    row = df_meta[df_meta["File Path"].str.contains(video_name, na=False)]
                    if not row.empty:
                        label_id = row["Label"].iloc[0]
                        data.append([img_path, label_id])
                    else:
                        print(f"Warning: No metadata for {video_name} in {csv_path}")
            except Exception as e:
                print(f"Error reading {csv_path}: {str(e)}")
    
    # Xử lý video giả
    fake_types = ["Deepfakes", "Face2Face", "FaceShifter", "FaceSwap", "NeuralTextures"]
    fake_dir = os.path.join(base_dir, "fake")
    if os.path.exists(fake_dir):
        for fake_type in fake_types:
            sub_dir = os.path.join(fake_dir, fake_type)
            if os.path.exists(sub_dir):
                csv_path = os.path.join(csv_dir, csv_files[fake_type])
                if os.path.exists(csv_path):
                    try:
                        df_meta = pd.read_csv(csv_path)
                        if "File Path" not in df_meta.columns or "Label" not in df_meta.columns:
                            print(f"Error: {csv_path} missing required columns (File Path, Label)")
                            continue
                        for img_path in glob.glob(os.path.join(sub_dir, "*/*.jpg")):
                            # Xử lý tiền tố nếu dùng ảnh tăng cường
                            base_name = os.path.basename(img_path)
                            if use_augmented and base_name.startswith("aug_"):
                                base_name = "_".join(base_name.split("_")[2:])  # Bỏ "aug_0_" hoặc "aug_1_"
                            video_name = base_name.split('_frame_')[0] + ".mp4"
                            row = df_meta[df_meta["File Path"].str.contains(video_name, na=False)]
                            if not row.empty:
                                label_id = row["Label"].iloc[0]
                                data.append([img_path, label_id])
                            else:
                                print(f"Warning: No metadata for {video_name} in {csv_path}")
                    except Exception as e:
                        print(f"Error reading {csv_path}: {str(e)}")
    
    # Lưu dataset
    if data:
        df = pd.DataFrame(data, columns=["image_path", "label"])
        df = df.sample(frac=1).reset_index(drop=True)  # Xáo trộn
        df.to_csv(output_csv, index=False)
        print(f"Created dataset CSV with {len(df)} samples")
    else:
        print("Error: No data collected, dataset CSV not created")

if __name__ == "__main__":
    base_dir = "data/augmented_frames"
    csv_dir = "data/raw_videos/csv"
    output_csv = "data/dataset.csv"
    use_augmented = True
    create_dataset_csv(base_dir, csv_dir, output_csv, use_augmented)