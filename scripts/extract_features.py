import numpy as np
import pandas as pd
import cv2
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.models import Model
import os
import glob

# Tải mô hình VGG16 (không bao gồm lớp fully connected)
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(128, 128, 3))
model = Model(inputs=base_model.input, outputs=base_model.get_layer('block5_pool').output)

def extract_features(image_path):
    # Đọc và tiền xử lý ảnh
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Cannot read image {image_path}")
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (128, 128))  # Đảm bảo đúng kích thước
    img = np.expand_dims(img, axis=0)  # Thêm batch dimension
    img = preprocess_input(img)  # Tiền xử lý cho VGG16
    
    # Trích xuất đặc trưng
    features = model.predict(img, verbose=0)
    return features.flatten()

def process_features(csv_path, output_csv):
    df = pd.read_csv(csv_path)
    features_list = []
    labels = []
    
    for idx, row in df.iterrows():
        image_path = row["image_path"]
        label = row["label"]
        features = extract_features(image_path)
        if features is not None:
            features_list.append(features)
            labels.append(label)
        if idx % 100 == 0:
            print(f"Processed {idx} images")
    
    # Chuyển đặc trưng thành DataFrame
    features_df = pd.DataFrame(features_list)
    features_df["label"] = labels
    features_df.to_csv(output_csv, index=False)
    print(f"Saved features to {output_csv}")

if __name__ == "__main__":
    # Trích xuất đặc trưng từ tập train, val, test
    process_features("data/train.csv", "data/train_features.csv")
    process_features("data/val.csv", "data/val_features.csv")
    process_features("data/test.csv", "data/test_features.csv")