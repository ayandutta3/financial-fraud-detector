import os
import shutil
import sys
import pandas as pd

def balance_csv(filename, label_col):
    csv_path = os.path.join("sample-dataset", filename)
    backup_path = os.path.join("sample-dataset", filename + ".bak")

    # 1. Create a backup of the original dataset if it does not already exist
    if not os.path.exists(backup_path):
        print(f"Creating backup of original dataset at: {backup_path}")
        shutil.copyfile(csv_path, backup_path)
    else:
        print(f"Backup already exists at: {backup_path}")

    # 2. Read the dataset
    print(f"Reading dataset {filename}...")
    df = pd.read_csv(csv_path)

    # 3. Separate fraud and non-fraud records
    df_fraud = df[df[label_col] == 1]
    df_non_fraud = df[df[label_col] == 0]

    n_fraud = len(df_fraud)
    n_non_fraud = len(df_non_fraud)

    print(f"Original fraud count: {n_fraud}")
    print(f"Original non-fraud count: {n_non_fraud}")

    if n_fraud == 0:
        print(f"Error: No fraud records found in the dataset to balance.")
        return

    # 4. Undersample non-fraud to match fraud records (keep all fraud, match non-fraud)
    print(f"Undersampling non-fraud records to match fraud count ({n_fraud})...")
    df_non_fraud_sampled = df_non_fraud.sample(n=n_fraud, random_state=42)

    # 5. Combine and shuffle the dataset
    df_balanced = pd.concat([df_fraud, df_non_fraud_sampled]).sample(frac=1, random_state=42).reset_index(drop=True)

    # 6. Save the balanced dataset back to the CSV
    print(f"Saving balanced dataset to: {csv_path}")
    df_balanced.to_csv(csv_path, index=False)

    # 7. Verify the new counts
    print("\nVerification:")
    print(df_balanced[label_col].value_counts())
    print(f"New dataset shape: {df_balanced.shape}")

if __name__ == "__main__":
    dataset_num = 2
    if len(sys.argv) > 1:
        try:
            dataset_num = int(sys.argv[1])
        except ValueError:
            print("Usage: python balance_dataset.py [1 or 2]")
            sys.exit(1)

    if dataset_num == 1:
        balance_csv("fraud-detection-dataset-1.csv", "is_fraud")
    elif dataset_num == 2:
        balance_csv("fraud-detection-dataset-2.csv", "isFraud")
    else:
        print("Invalid dataset number. Specify 1 or 2.")

