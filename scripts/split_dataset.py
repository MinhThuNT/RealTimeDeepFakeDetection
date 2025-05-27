from sklearn.model_selection import train_test_split
import pandas as pd

def split_dataset(csv_path, train_csv, val_csv, test_csv, train_size=0.7, val_size=0.15):
    # Kiểm tra sự tồn tại của file
    if not pd.io.common.is_file_like(csv_path) and not os.path.exists(csv_path):
        print(f"Error: Input file {csv_path} does not exist")
        return
    
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Error reading {csv_path}: {str(e)}")
        return
    
    # Kiểm tra cột label
    if "label" not in df.columns:
        print(f"Error: {csv_path} missing required column 'label'")
        return
    
    # Kiểm tra số lượng mẫu
    if len(df) < 10:
        print(f"Warning: Dataset has only {len(df)} samples, splitting may not be meaningful")
    
    # Phân phối nhãn
    print("Label distribution before splitting:")
    print(df["label"].value_counts())
    
    # Chia dữ liệu
    train_df, temp_df = train_test_split(df, train_size=train_size, stratify=df["label"], random_state=42)
    val_ratio = val_size / (1 - train_size)  # Tỷ lệ validation trên tập còn lại
    val_df, test_df = train_test_split(temp_df, train_size=val_ratio, stratify=temp_df["label"], random_state=42)
    
    # Số lượng mẫu và phân phối nhãn
    print(f"\nTrain: {len(train_df)} samples")
    print("Train label distribution:")
    print(train_df["label"].value_counts())
    print(f"Validation: {len(val_df)} samples")
    print("Validation label distribution:")
    print(val_df["label"].value_counts())
    print(f"Test: {len(test_df)} samples")
    print("Test label distribution:")
    print(test_df["label"].value_counts())
    
    try:
        train_df.to_csv(train_csv, index=False)
        val_df.to_csv(val_csv, index=False)
        test_df.to_csv(test_csv, index=False)
        print("\nDataset split completed and saved successfully")
    except Exception as e:
        print(f"Error saving files: {str(e)}")

if __name__ == "__main__":
    split_dataset(
        "data/dataset.csv",
        "data/train.csv",
        "data/val.csv",
        "data/test.csv"
    )