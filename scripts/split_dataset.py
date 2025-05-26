from sklearn.model_selection import train_test_split
import pandas as pd

def split_dataset(csv_path, train_csv, val_csv, test_csv, train_size=0.7, val_size=0.15):
    df = pd.read_csv(csv_path)
    train_df, temp_df = train_test_split(df, train_size=train_size, stratify=df["label"], random_state=42)
    val_df, test_df = train_test_split(temp_df, train_size=val_size/(1-train_size), stratify=temp_df["label"], random_state=42)
    
    train_df.to_csv(train_csv, index=False)
    val_df.to_csv(val_csv, index=False)
    test_df.to_csv(test_csv, index=False)
    print(f"Train: {len(train_df)}, Validation: {len(val_df)}, Test: {len(test_df)}")

if __name__ == "__main__":
    split_dataset(
        "data/dataset.csv",
        "data/train.csv",
        "data/val.csv",
        "data/test.csv"
    )