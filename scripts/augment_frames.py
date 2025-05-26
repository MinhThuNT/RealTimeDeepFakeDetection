import cv2
import os
import glob
import albumentations as A

augmentation = A.Compose([
    A.Rotate(limit=30, p=0.5),
    A.HorizontalFlip(p=0.5),
    A.GaussNoise(p=0.3),
    A.RandomBrightnessContrast(p=0.3)
])

def augment_frames(input_dir, output_dir, augmentations, num_augmentations=2):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for img_path in glob.glob(os.path.join(input_dir, "*.jpg")):
        img = cv2.imread(img_path)
        if img is not None:
            for i in range(num_augmentations):
                augmented = augmentations(image=img)['image']
                output_path = os.path.join(output_dir, f"aug_{i}_{os.path.basename(img_path)}")
                cv2.imwrite(output_path, augmented)

def process_augmented_frames(base_input_dir, base_output_dir):
    for label in ["real", "fake"]:
        input_dir = os.path.join(base_input_dir, label)
        output_dir = os.path.join(base_output_dir, label)
        for video_dir in glob.glob(os.path.join(input_dir, "*")):
            video_name = os.path.basename(video_dir)
            augment_frames(video_dir, os.path.join(output_dir, video_name), augmentation)

if __name__ == "__main__":
    input_base_dir = "data/resized_frames"
    output_base_dir = "data/augmented_frames"
    process_augmented_frames(input_base_dir, output_base_dir)