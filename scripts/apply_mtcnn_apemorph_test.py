import os
import cv2
import pandas as pd
from tqdm import tqdm
from facenet_pytorch import MTCNN
from PIL import Image

# ==========================
# PATHS
# ==========================
CSV_IN = "/content/drive/MyDrive/workspace3/apemorph_test_balanced.csv"
IMG_ROOT = "/content/drive/MyDrive/workspace3"

OUT_IMG_DIR = "/content/drive/MyDrive/workspace3/apemorph_test_mtcnn"
OUT_CSV = "/content/drive/MyDrive/workspace3/apemorph_test_balanced_mtcnn.csv"

os.makedirs(OUT_IMG_DIR, exist_ok=True)

# ==========================
# LOAD CSV
# ==========================
df = pd.read_csv(CSV_IN)

# ==========================
# INIT MTCNN
# ==========================
mtcnn = MTCNN(
    image_size=224,
    margin=20,
    select_largest=True,
    post_process=True,
    device="cuda" if cv2.cuda.getCudaEnabledDeviceCount() > 0 else "cpu"
)

new_paths = []

# ==========================
# PROCESS IMAGES
# ==========================
for idx, row in tqdm(df.iterrows(), total=len(df)):
    img_path = os.path.join(IMG_ROOT, row["path"])

    try:
        img = Image.open(img_path).convert("RGB")
        face = mtcnn(img)

        if face is None:
            new_paths.append(None)
            continue

        out_name = f"{idx}.jpg"
        out_path = os.path.join(OUT_IMG_DIR, out_name)

        face_img = face.permute(1, 2, 0).numpy()
        face_img = (face_img * 255).astype("uint8")
        Image.fromarray(face_img).save(out_path)

        new_paths.append(out_path)

    except Exception as e:
        print(f"Error processing {img_path}: {e}")
        new_paths.append(None)

# ==========================
# SAVE NEW CSV
# ==========================
df["path"] = new_paths
df = df.dropna(subset=["path"])
df.to_csv(OUT_CSV, index=False)

print("MTCNN preprocessing (TEST) completed")
print("Saved images to:", OUT_IMG_DIR)
print("Saved CSV to:", OUT_CSV)
