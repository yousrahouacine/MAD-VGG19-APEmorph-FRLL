import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from PIL import Image

import torch
import torch.nn as nn
from torchvision import models, transforms

from facenet_pytorch import MTCNN


# ======================================================
# PATHS
# ======================================================
WORKSPACE = Path("/content/drive/MyDrive/workspace3")

APEMORPH_CSV = WORKSPACE / "apemorph_test_balanced.csv"
MODEL_PATH   = WORKSPACE / "vgg19_apemorph_final.pth"

OUT_DIR = WORKSPACE / "embeddings_apemorph_mtcnn"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_EMB   = OUT_DIR / "embeddings_25088.npy"
OUT_LABEL = OUT_DIR / "labels.npy"
OUT_PATHS = OUT_DIR / "paths.npy"


# ======================================================
# PARAMS
# ======================================================
BATCH_SIZE = 16 


# ======================================================
# DEVICE
# ======================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# ======================================================
# FALLBACK TRANSFORM (if MTCNN fails)
# (Keeping it consistent with the training: Resize + ToTensor only)
# ======================================================
fallback_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])


# ======================================================
# MTCNN
# ======================================================
mtcnn = MTCNN(
    image_size=224,
    margin=20,
    post_process=True,
    device=device,
    keep_all=False
)


# ======================================================
# LOAD CSV
# ======================================================
df = pd.read_csv(APEMORPH_CSV)
paths = df["path"].tolist()
labels = df["label"].replace({"bonafide": 0, "real": 0, "morph": 1}).astype(int).to_numpy()

print("Samples:", len(df))
print("Label counts:", pd.Series(labels).value_counts().to_dict())


# ======================================================
# LOAD MODEL (baseline trained on APEmorph)
# ======================================================
model = models.vgg19(weights=models.VGG19_Weights.DEFAULT)
for p in model.features.parameters():
    p.requires_grad = False

model.classifier = nn.Sequential(nn.Linear(25088, 2))
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model = model.to(device)
model.eval()

print(" Loaded baseline model:", MODEL_PATH.name)


# ======================================================
# EMBEDDING EXTRACTOR
# 25088D = flatten output of conv features
# ======================================================
embed_extractor = nn.Sequential(
    model.features,
    nn.Flatten()
).to(device)
embed_extractor.eval()


# ======================================================
# BATCHED EXTRACTION
# ======================================================
all_emb = []

def load_and_crop(path_str: str) -> torch.Tensor:
    """
    Returns a torch tensor [3,224,224] in range [0,1].
    Uses MTCNN; if it fails, uses full image fallback resize+ToTensor.
    """
    img = Image.open(path_str).convert("RGB")
    face = mtcnn(img)  # returns tensor [3,224,224] or None
    if face is None:
        face = fallback_tf(img)
    # Ensure float32
    return face.float()

with torch.no_grad():
    for i in tqdm(range(0, len(paths), BATCH_SIZE), desc="Extracting (MTCNN)", ncols=100):
        batch_paths = paths[i:i+BATCH_SIZE]

        batch_tensors = [load_and_crop(p) for p in batch_paths]
        x = torch.stack(batch_tensors, dim=0).to(device)  # [B,3,224,224]

        emb = embed_extractor(x)  # [B,25088]
        all_emb.append(emb.cpu().numpy())

embeddings = np.vstack(all_emb).astype(np.float32)

# ======================================================
# SAVE
# ======================================================
np.save(OUT_EMB, embeddings)
np.save(OUT_LABEL, labels)
np.save(OUT_PATHS, np.array(paths, dtype=object))

print("\n Saved:")
print(" -", OUT_EMB, "shape =", embeddings.shape)
print(" -", OUT_LABEL, "shape =", labels.shape)
print(" -", OUT_PATHS, "shape =", len(paths))
