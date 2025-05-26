import cv2
import os
import glob
import pandas as pd

def extract_frames(video_path, output_dir, frame_rate=1, expected_frames=None):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video {video_path}")
        return 0
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(fps / frame_rate) if fps > 0 else 1
    
    count = 0
    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if count % frame_interval == 0:
            frame_name = f"{os.path.splitext(os.path.basename(video_path))[0]}_frame_{frame_idx}.jpg"
            cv2.imwrite(os.path.join(output_dir, frame_name), frame)
            frame_idx += 1
        count += 1
    
    cap.release()
    if expected_frames:
        expected_extracted = expected_frames // frame_interval
        print(f"Extracted {frame_idx} frames from {video_path}, expected ~{expected_extracted}")
        if abs(frame_idx - expected_extracted) > expected_extracted * 0.1:
            print(f"Warning: Frame count mismatch for {video_path}")
    return frame_idx

def process_videos(video_dir, output_base_dir, label, metadata_csv=None):
    df_meta = pd.read_csv(metadata_csv) if metadata_csv else None
    video_list = glob.glob(os.path.join(video_dir, "*.mp4"))
    
    # Sắp xếp theo kích thước file nếu có metadata
    if df_meta is not None:
        video_sizes = []
        for video_path in video_list:
            video_name = os.path.basename(video_path)
            row = df_meta[df_meta["File Path"] == f"{os.path.basename(video_dir)}/{video_name}"]
            if not row.empty:
                size = row["File Size(MB)"].iloc[0]
                video_sizes.append((video_path, size))
        video_sizes.sort(key=lambda x: x[1])
        video_list = [v[0] for v in video_sizes]
    
    for video_path in video_list:
        video_name = os.path.basename(video_path).split('.')[0]
        output_dir = os.path.join(output_base_dir, label, video_name)
        expected_frames = None
        if df_meta is not None:
            row = df_meta[df_meta["File Path"] == f"{os.path.basename(video_dir)}/{video_name}.mp4"]
            if not row.empty:
                expected_frames = row["Frame Count"].iloc[0]
        extract_frames(video_path, output_dir, frame_rate=1, expected_frames=expected_frames)

if __name__ == "__main__":
    dataset_path = "data/raw_videos"
    output_base_dir = "data/frames"
    csv_dir = "data/raw_videos/csv"
    
    # Xử lý video thật
    real_dir = f"{dataset_path}/original"
    process_videos(real_dir, output_base_dir, "real", metadata_csv=f"{csv_dir}/original.csv")
    
    # Xử lý video giả
    video_dirs = ["Deepfakes", "Face2Face", "FaceShifter", "FaceSwap", "NeuralTextures"]
    for dir_name in video_dirs:
        fake_dir = f"{dataset_path}/{dir_name}"
        csv_file = f"{dir_name}.csv"
        metadata_path = f"{csv_dir}/{csv_file}"
        if os.path.exists(metadata_path):
            process_videos(fake_dir, output_base_dir, "fake", metadata_csv=metadata_path)
        else:
            print(f"Warning: Metadata file {metadata_path} not found, skipping {fake_dir}")