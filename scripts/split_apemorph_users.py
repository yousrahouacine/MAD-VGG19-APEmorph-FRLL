import pandas as pd
import numpy as np
from pathlib import Path

# --------------------------------------------------
# CONFIG (COLAB + DRIVE)
# --------------------------------------------------
WORKSPACE = Path("/content/drive/MyDrive/workspace3")

INPUT_CSV = WORKSPACE / "apemorph_dataframe.csv"
TRAIN_CSV = WORKSPACE / "apemorph_train.csv"
TEST_CSV  = WORKSPACE / "apemorph_test.csv"

# --------------------------------------------------
# Load dataframe
# --------------------------------------------------
df = pd.read_csv(INPUT_CSV)

print("Total samples:", len(df))
print("Unique users:", df["user_id"].nunique())

# --------------------------------------------------
# Split users (identity-based split)
# --------------------------------------------------
users = df["user_id"].dropna().unique()

np.random.seed(42)
np.random.shuffle(users)

split_point = int(0.80 * len(users))
train_users = users[:split_point]
test_users  = users[split_point:]

print("Train users:", len(train_users))
print("Test users:", len(test_users))

# --------------------------------------------------
# Create train / test dataframes
# --------------------------------------------------
train_df = df[df["user_id"].isin(train_users)].reset_index(drop=True)
test_df  = df[df["user_id"].isin(test_users)].reset_index(drop=True)

print("Train samples:", len(train_df))
print("Test samples:", len(test_df))

# --------------------------------------------------
# Save CSVs
# --------------------------------------------------
train_df.to_csv(TRAIN_CSV, index=False)
test_df.to_csv(TEST_CSV, index=False)

print("\n  SPLIT DONE")
print("Saved to:")
print(" -", TRAIN_CSV)
print(" -", TEST_CSV)
