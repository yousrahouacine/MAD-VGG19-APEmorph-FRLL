import pandas as pd
from pathlib import Path

# =====================================
# PATHS
# =====================================
WORKSPACE = Path("/content/drive/MyDrive/workspace3")

INPUT_TRAIN = WORKSPACE / "apemorph_train.csv"
OUTPUT_TRAIN = WORKSPACE / "apemorph_train_balanced.csv"

# =====================================
# LOAD TRAIN CSV
# =====================================
df = pd.read_csv(INPUT_TRAIN)

print("Original train size:", len(df))

# =====================================
# SPLIT REAL / MORPH
# =====================================
df_real = df[df["label"] == "real"]
df_morph = df[df["label"] == "morph"]

print("Real samples :", len(df_real))
print("Morph samples:", len(df_morph))

# =====================================
# UNDERSAMPLE REAL
# =====================================
df_real_balanced = df_real.sample(
    n=len(df_morph),
    random_state=42
)

# =====================================
# MERGE + SHUFFLE
# =====================================
df_balanced = pd.concat([df_real_balanced, df_morph])
df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

# =====================================
# SAVEE
# =====================================
df_balanced.to_csv(OUTPUT_TRAIN, index=False)

print("\n TRAIN BALANCED")
print("Final train size:", len(df_balanced))
print("Real :", sum(df_balanced["label"] == "real"))
print("Morph:", sum(df_balanced["label"] == "morph"))
print("Saved to:", OUTPUT_TRAIN)
