import cv2
import os
import glob
import pandas as pd

def resize_frames(input_dir, output_dir, size=(128, 128), metadata_csv=None):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    df_meta = pd.read_csv(metadata_csv) if metadata_csv else None
    for img_path in glob.glob(os.path.join(input_dir, "*.jpg")):
        img = cv2.imread(img_path)
        if img is not None:
            video_name = os.path.basename(img_path).split('_frame_')[0] + ".mp4"
            if df_meta is not None:
                row = df_meta[df_meta["File Path"].str.contains(video_name)]
                if not row.empty:
                    orig_width, orig_height = row["Width"].iloc[0], row["Height"].iloc[0]
                    aspect_ratio = orig_width / orig_height
                    target_ratio = size[0] / size[1]
                    if abs(aspect_ratio - target_ratio) > 0.1:
                        print(f"Warning: Aspect ratio mismatch for {video_name}")
            resized_img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
            output_path = os.path.join(output_dir, os.path.basename(img_path))
            cv2.imwrite(output_path, resized_img)

def process_resized_frames(base_input_dir, base_output_dir, csv_dir):
    for label in ["real", "fake"]:
        input_dir = os.path.join(base_input_dir, label)
        output_dir = os.path.join(base_output_dir, label)
        metadata_csv = f"{csv_dir}/original.csv" if label == "real" else f"{csv_dir}/FF+_Metadata.csv"
        for video_dir in glob.glob(os.path.join(input_dir, "*")):
            video_name = os.path.basename(video_dir)
            resize_frames(video_dir, os.path.join(output_dir, video_name), metadata_csv=metadata_csv)

if __name__ == "__main__":
    input_base_dir = "data/frames"
    output_base_dir = "data/resized_frames"
    csv_dir = "data/raw_videos/csv"
    process_resized_frames(input_base_dir, output_base_dir, csv_dir)