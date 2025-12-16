import pandas as pd
from pathlib import Path

# ======================================================
# PATHS
# ======================================================
ROOT = Path("/content/drive/MyDrive/workspace3/second_dataset")
OUTPUT_CSV = Path("/content/drive/MyDrive/workspace3/frll_test.csv")

entries = []

# ======================================================
# BONAFIDE CLASSES
# ======================================================
bonafide_folders = ["neutral", "smiling"]

for folder in bonafide_folders:
    folder_path = ROOT / folder
    if not folder_path.exists():
        print(f" Folder not found: {folder_path}")
        continue

    for img in folder_path.iterdir():
        if img.suffix.lower() in [".jpg", ".jpeg", ".png"]:
            entries.append({
                "path": str(img),
                "label": "bonafide"
            })

# ======================================================
# MORPH CLASS
# ======================================================
morph_folder = ROOT / "morphed"
if morph_folder.exists():
    for img in morph_folder.iterdir():
        if img.suffix.lower() in [".jpg", ".jpeg", ".png"]:
            entries.append({
                "path": str(img),
                "label": "morph"
            })
else:
    print(f" Folder not found: {morph_folder}")

# ======================================================
# SAVE CSV
# ======================================================
df = pd.DataFrame(entries)
df.to_csv(OUTPUT_CSV, index=False)

print(" FRLL CSV created")
print("Total samples:", len(df))
print(df["label"].value_counts())
print("Saved to:", OUTPUT_CSV)
