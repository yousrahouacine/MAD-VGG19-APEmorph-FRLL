import re
import pandas as pd
from pathlib import Path

# -----------------------------------------------------------
# Extract FIRST UserXX from string
# -----------------------------------------------------------
def extract_user_id(text):
    match = re.search(r"User(\d+)_", text)
    return int(match.group(1)) if match else None


# -----------------------------------------------------------
# Build APEmorph dataframe (COLAB + DRIVE)
# -----------------------------------------------------------
def build_apemorph_dataframe(root):

    root = Path(root)
    entries = []

    # =======================================================
    # 1) REAL IMAGES
    # /content/drive/MyDrive/APEmorph_complete/Real/UserXX_*/
    # =======================================================
    real_root = root / "Real"

    if not real_root.exists():
        raise FileNotFoundError(f"Real folder not found: {real_root}")

    for user_folder in real_root.iterdir():
        if not user_folder.is_dir():
            continue

        user_id = extract_user_id(user_folder.name)

        for img in user_folder.iterdir():
            if img.suffix.lower() in [".png", ".jpg", ".jpeg"]:
                entries.append({
                    "path": str(img),
                    "label": "real",
                    "user_id": user_id,
                    "source": "Real"
                })

    # =======================================================
    # 2) MORPH IMAGES (ALL 4 TYPES)
    # images directly inside folders
    # =======================================================
    morph_folders = [
        "FaceBlender",
        "FaceBlender_restored",
        "FaceMorpher",
        "FaceMorpher_restored",
    ]

    for folder_name in morph_folders:
        morph_root = root / folder_name

        if not morph_root.exists():
            print(f"Warning!!!: {morph_root} not found")
            continue

        for img in morph_root.iterdir():
            if img.suffix.lower() in [".png", ".jpg", ".jpeg"]:

                user_id = extract_user_id(img.name)

                entries.append({
                    "path": str(img),
                    "label": "morph",
                    "user_id": user_id,
                    "source": folder_name
                })

    df = pd.DataFrame(entries)
    return df


# -----------------------------------------------------------
# MAIN (COLAB)
# -----------------------------------------------------------
if __name__ == "__main__":

    ROOT = "/content/drive/MyDrive/APEmorph_complete"
    OUTPUT = "/content/drive/MyDrive/workspace3/apemorph_dataframe.csv"

    df = build_apemorph_dataframe(ROOT)
    df.to_csv(OUTPUT, index=False)

    print("\n DONE")
    print("Saved to:", OUTPUT)
    print("Total samples:", len(df))
    print(df.head())
