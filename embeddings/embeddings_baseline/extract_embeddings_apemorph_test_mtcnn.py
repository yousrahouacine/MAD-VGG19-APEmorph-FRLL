import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from PIL import Image

import torch
import torch.nn as nn
from torchvision import models, transforms


# ======================================================
# PATHS
# ======================================================
WORKSPACE = Path("/content/drive/MyDrive/workspace3")

APEMORPH_CSV = WORKSPACE / "apemorph_test_balanced_mtcnn.csv"
MODEL_PATH  = WORKSPACE / "vgg19_apemorph_final.pth"

OUT_DIR = WORKSPACE / "embeddings_apemorph_test"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_EMB   = OUT_DIR / "embeddings_25088.npy"
OUT_LABEL = OUT_DIR / "labels.npy"
OUT_PATHS = OUT_DIR / "paths.npy"


# ======================================================
# PARAMS
# ======================================================
BATCH_SIZE = 32


# ======================================================
# DEVICE
# ======================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# ======================================================
# TRANSFORM (SAME AS TRAIN / TEST)
# ======================================================
tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])


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
# LOAD TRAINED MODEL
# ======================================================
model = models.vgg19(weights=models.VGG19_Weights.DEFAULT)

for p in model.features.parameters():
    p.requires_grad = False

model.classifier = nn.Sequential(nn.Linear(25088, 2))
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model = model.to(device)
model.eval()

print("Loaded trained model:", MODEL_PATH.name)


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

with torch.no_grad():
    for i in tqdm(range(0, len(paths), BATCH_SIZE),
                  desc="Extracting APEmorph TEST embeddings",
                  ncols=100):

        batch_paths = paths[i:i+BATCH_SIZE]

        batch_imgs = []
        for p in batch_paths:
            img = Image.open(p).convert("RGB")
            img = tf(img)
            batch_imgs.append(img)

        x = torch.stack(batch_imgs).to(device)
        emb = embed_extractor(x)

        all_emb.append(emb.cpu().numpy())

embeddings = np.vstack(all_emb).astype(np.float32)


# ======================================================
# SAVE
# ======================================================
np.save(OUT_EMB, embeddings)
np.save(OUT_LABEL, labels)
np.save(OUT_PATHS, np.array(paths, dtype=object))

print("\nAPEmorph TEST embeddings saved:")
print(" -", OUT_EMB, embeddings.shape)
print(" -", OUT_LABEL, labels.shape)
print(" -", OUT_PATHS, len(paths))
