import cv2
import pandas as pd
import matplotlib.pyplot as plt

def visualize_data(csv_path, num_samples=3):
    df = pd.read_csv(csv_path)
    plt.figure(figsize=(10, 4))
    for i in range(num_samples):
        img = cv2.imread(df["image_path"].iloc[i])
        plt.subplot(1, num_samples, i+1)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title(f"Label: {df['label'].iloc[i]}")
        plt.axis("off")
    plt.show()

if __name__ == "__main__":
    visualize_data("data/train.csv")