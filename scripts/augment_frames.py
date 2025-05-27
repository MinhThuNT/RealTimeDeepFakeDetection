import cv2
import os
import glob
import albumentations as A
from multiprocessing import Pool

augmentation = A.Compose([
    A.Rotate(limit=30, p=0.5),
    A.HorizontalFlip(p=0.5),
    A.GaussNoise(p=0.3),
    A.RandomBrightnessContrast(p=0.3)
])

def augment_single_frame(args):
    img_path, output_dir, augmentations, i = args
    img = cv2.imread(img_path)
    if img is None:
        print(f"Error: Cannot read image {img_path}, skipping...")
        return
    augmented = augmentations(image=img)['image']
    output_path = os.path.join(output_dir, f"aug_{i}_{os.path.basename(img_path)}")
    cv2.imwrite(output_path, augmented)
    print(f"Augmented and saved: {output_path}")

def augment_frames(input_dir, output_dir, augmentations, num_augmentations=2):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    img_paths = glob.glob(os.path.join(input_dir, "*.jpg"))
    if not img_paths:
        print(f"Warning: No images found in {input_dir}, skipping...")
        return
    tasks = [(img_path, output_dir, augmentations, i) for img_path in img_paths for i in range(num_augmentations)]
    with Pool() as pool:
        pool.map(augment_single_frame, tasks)

def process_augmented_frames(base_input_dir, base_output_dir):
    if not os.path.exists(base_input_dir):
        print(f"Error: Input directory {base_input_dir} does not exist")
        return
    # Xử lý video thật
    input_dir = os.path.join(base_input_dir, "real")
    if os.path.exists(input_dir):
        output_dir = os.path.join(base_output_dir, "real")
        for video_dir in glob.glob(os.path.join(input_dir, "*")):
            if os.path.exists(video_dir):
                video_name = os.path.basename(video_dir)
                augment_frames(video_dir, os.path.join(output_dir, video_name), augmentation)
    
    # Xử lý video giả
    fake_types = ["Deepfakes", "Face2Face", "FaceShifter", "FaceSwap", "NeuralTextures"]
    input_dir = os.path.join(base_input_dir, "fake")
    if os.path.exists(input_dir):
        for fake_type in fake_types:
            sub_input_dir = os.path.join(input_dir, fake_type)
            if os.path.exists(sub_input_dir):
                sub_output_dir = os.path.join(base_output_dir, "fake", fake_type)
                for video_dir in glob.glob(os.path.join(sub_input_dir, "*")):
                    if os.path.exists(video_dir):
                        video_name = os.path.basename(video_dir)
                        augment_frames(video_dir, os.path.join(sub_output_dir, video_name), augmentation)

if __name__ == "__main__":
    input_base_dir = "data/resized_frames"
    output_base_dir = "data/augmented_frames"
    process_augmented_frames(input_base_dir, output_base_dir)