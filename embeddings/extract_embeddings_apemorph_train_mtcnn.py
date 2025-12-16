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

APEMORPH_CSV = WORKSPACE / "apemorph_train_balanced.csv"
MODEL_PATH  = WORKSPACE / "vgg19_apemorph_final.pth"

OUT_DIR = WORKSPACE / "embeddings_apemorph_train_mtcnn"
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
# FALLBACK TRANSFORM (same as training)
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
labels = df["label"].replace({
    "bonafide": 0,
    "real": 0,
    "morph": 1
}).astype(int).to_numpy()

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
# EMBEDDING EXTRACTOR (25088D)
# ======================================================
embed_extractor = nn.Sequential(
    model.features,
    nn.Flatten()
).to(device)

embed_extractor.eval()


# ======================================================
# EXTRACTION
# ======================================================
all_emb = []

def load_and_crop(path_str: str) -> torch.Tensor:
    img = Image.open(path_str).convert("RGB")
    face = mtcnn(img)
    if face is None:
        face = fallback_tf(img)
    return face.float()

with torch.no_grad():
    for i in tqdm(range(0, len(paths), BATCH_SIZE),
                  desc="Extracting APEmorph TRAIN embeddings (MTCNN)",
                  ncols=100):

        batch_paths = paths[i:i+BATCH_SIZE]
        batch_imgs = [load_and_crop(p) for p in batch_paths]

        x = torch.stack(batch_imgs, dim=0).to(device)
        emb = embed_extractor(x)

        all_emb.append(emb.cpu().numpy())

embeddings = np.vstack(all_emb).astype(np.float32)


# ======================================================
# SAVE
# ======================================================
np.save(OUT_EMB, embeddings)
np.save(OUT_LABEL, labels)
np.save(OUT_PATHS, np.array(paths, dtype=object))

print("\n APEmorph TRAIN embeddings saved:")
print(" -", OUT_EMB, "shape =", embeddings.shape)
print(" -", OUT_LABEL, "shape =", labels.shape)
print(" -", OUT_PATHS, "shape =", len(paths))
