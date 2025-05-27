import cv2
import os
import glob
import pandas as pd

def resize_frames(input_dir, output_dir, size=(128, 128), metadata_csv=None):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    df_meta = None
    if metadata_csv and os.path.exists(metadata_csv):
        df_meta = pd.read_csv(metadata_csv)
    else:
        print(f"Warning: Metadata file {metadata_csv} not found, skipping aspect ratio check")
    
    warned_videos = set()
    for img_path in glob.glob(os.path.join(input_dir, "*.jpg")):
        img = cv2.imread(img_path)
        if img is not None:
            video_name = os.path.basename(img_path).split('_frame_')[0] + ".mp4"
            if df_meta is not None and video_name not in warned_videos:
                row = df_meta[df_meta["File Path"].str.contains(video_name, na=False)]
                if not row.empty:
                    orig_width, orig_height = row["Width"].iloc[0], row["Height"].iloc[0]
                    aspect_ratio = orig_width / orig_height
                    target_ratio = size[0] / size[1]
                    if abs(aspect_ratio - target_ratio) > 0.1:
                        print(f"Warning: Aspect ratio mismatch for {video_name}")
                        warned_videos.add(video_name)
            resized_img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
            output_path = os.path.join(output_dir, os.path.basename(img_path))
            cv2.imwrite(output_path, resized_img)

def process_resized_frames(base_input_dir, base_output_dir, csv_dir):
    # Xử lý video thật
    input_dir = os.path.join(base_input_dir, "real")
    if os.path.exists(input_dir):
        output_dir = os.path.join(base_output_dir, "real")
        metadata_csv = f"{csv_dir}/original.csv"  # Dùng original.csv cho video REAL
        if not os.path.exists(metadata_csv):
            print(f"Warning: Metadata file {metadata_csv} does not exist, skipping real videos...")
            return
        for video_dir in glob.glob(os.path.join(input_dir, "*")):
            video_name = os.path.basename(video_dir) + ".mp4"
            resize_frames(video_dir, os.path.join(output_dir, video_name.split('.')[0]), metadata_csv=metadata_csv)
            print(f"Processed real video: {video_name}")
    
    # Xử lý video giả
    fake_types = ["Deepfakes", "Face2Face", "FaceShifter", "FaceSwap", "NeuralTextures", "DeepFakeDetection"]
    input_dir = os.path.join(base_input_dir, "fake")
    if os.path.exists(input_dir):
        for fake_type in fake_types:
            sub_input_dir = os.path.join(input_dir, fake_type)
            if os.path.exists(sub_input_dir):
                sub_output_dir = os.path.join(base_output_dir, "fake", fake_type)
                metadata_csv = f"{csv_dir}/FF++_Metadata.csv"  # Dùng FF++_Metadata.csv cho tất cả fake types
                if not os.path.exists(metadata_csv):
                    print(f"Warning: Metadata file {metadata_csv} does not exist, skipping {fake_type}...")
                    continue
                for video_dir in glob.glob(os.path.join(sub_input_dir, "*")):
                    video_name = os.path.basename(video_dir) + ".mp4"
                    # Đảm bảo output giữ nguyên cấu trúc thư mục con
                    resize_frames(video_dir, os.path.join(sub_output_dir, os.path.basename(video_dir)), metadata_csv=metadata_csv)
                    print(f"Processed {fake_type} video: {video_name}")
            else:
                print(f"Warning: Directory {sub_input_dir} does not exist, skipping...")

if __name__ == "__main__":
    input_base_dir = "data/frames"
    output_base_dir = "data/resized_frames"
    csv_dir = "data/raw_videos/csv"
    process_resized_frames(input_base_dir, output_base_dir, csv_dir)