import pandas as pd
from pathlib import Path

# =====================================
# PATHS
# =====================================
WORKSPACE = Path("/content/drive/MyDrive/workspace3")

INPUT_TEST = WORKSPACE / "apemorph_test.csv"
OUTPUT_TEST = WORKSPACE / "apemorph_test_balanced.csv"

# =====================================
# LOAD TEST CSV
# =====================================
df = pd.read_csv(INPUT_TEST)

print("Original test size:", len(df))

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
# SAVE
# =====================================
df_balanced.to_csv(OUTPUT_TEST, index=False)

print("\n TEST BALANCED")
print("Final test size:", len(df_balanced))
print("Real :", sum(df_balanced["label"] == "real"))
print("Morph:", sum(df_balanced["label"] == "morph"))
print("Saved to:", OUTPUT_TEST)
