import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# ======================================================
# PATHS
# ======================================================
WORKSPACE = Path("/content/drive/MyDrive/workspace3")

# CSVs
APE_TRAIN_CSV = WORKSPACE / "apemorph_train_balanced.csv"
APE_TEST_CSV  = WORKSPACE / "apemorph_test_balanced.csv"
FRLL_CSV      = WORKSPACE / "frll_test.csv"

# EMBEDDINGS
APE_TRAIN_DIR = WORKSPACE / "embeddings_apemorph_train_mtcnn"
APE_TEST_DIR  = WORKSPACE / "embeddings_apemorph_test_mtcnn"
FRLL_DIR      = WORKSPACE / "embeddings_frll_mtcnn"

APE_TRAIN_X = np.load(APE_TRAIN_DIR / "embeddings_25088.npy")
APE_TRAIN_Y = np.load(APE_TRAIN_DIR / "labels.npy")

APE_TEST_X  = np.load(APE_TEST_DIR / "embeddings_25088.npy")
APE_TEST_Y  = np.load(APE_TEST_DIR / "labels.npy")

FRLL_X = np.load(FRLL_DIR / "embeddings_25088.npy")
FRLL_Y = np.load(FRLL_DIR / "labels.npy")

OUT_DIR = WORKSPACE / "tsne_results"
OUT_DIR.mkdir(exist_ok=True)

OUT_FIG = OUT_DIR / "tsne_APE_train_test_FRLL_6groups.png"

# ======================================================
# PARAMETERS
# ======================================================
MAX_PER_GROUP = 800
PCA_DIM = 50
PERPLEXITY = 30
RANDOM_STATE = 42

# ======================================================
# BUILD GROUP LABELS
# group_id:
# 0 = APE train bonafide
# 1 = APE train morph
# 2 = APE test bonafide
# 3 = APE test morph
# 4 = FRLL bonafide
# 5 = FRLL morph
# ======================================================
groups = []

for y in APE_TRAIN_Y:
    groups.append(0 if y == 0 else 1)

for y in APE_TEST_Y:
    groups.append(2 if y == 0 else 3)

for y in FRLL_Y:
    groups.append(4 if y == 0 else 5)

groups = np.array(groups)

# ======================================================
# COMBINE ALL EMBEDDINGS
# ======================================================
X = np.vstack([
    APE_TRAIN_X,
    APE_TEST_X,
    FRLL_X
]).astype(np.float32)

print("Combined embeddings shape:", X.shape)
print("Groups shape:", groups.shape)

# ======================================================
# SUBSAMPLE PER GROUP 
# ======================================================
def subsample_per_group(X, groups, max_n, seed=42):
    rng = np.random.default_rng(seed)
    idx_all = []

    for g in range(6):
        idx = np.where(groups == g)[0]
        if len(idx) > max_n:
            idx = rng.choice(idx, size=max_n, replace=False)
        idx_all.append(idx)

    idx_all = np.concatenate(idx_all)
    return X[idx_all], groups[idx_all]

X, groups = subsample_per_group(X, groups, MAX_PER_GROUP)

# ======================================================
# PCA → 50D
# ======================================================
print("Running PCA...")
pca = PCA(n_components=PCA_DIM, random_state=RANDOM_STATE)
X_pca = pca.fit_transform(X)
print("PCA variance retained:", pca.explained_variance_ratio_.sum())

# ======================================================
# t-SNE → 2D
# ======================================================
print("Running t-SNE...")
tsne = TSNE(
    n_components=2,
    perplexity=PERPLEXITY,
    init="pca",
    learning_rate="auto",
    random_state=RANDOM_STATE
)
Z = tsne.fit_transform(X_pca)

# ======================================================
# PLOT (6 COLORS + 6 SHAPES)
# ======================================================
labels_names = [
    "APE train bonafide",
    "APE train morph",
    "APE test bonafide",
    "APE test morph",
    "FRLL bonafide",
    "FRLL morph"
]

colors = ["tab:blue", "tab:orange", "tab:green",
          "tab:red", "tab:purple", "tab:brown"]

markers = ["o", "^", "s", "D", "x", "*"]

plt.figure(figsize=(9, 8))

for g in range(6):
    idx = np.where(groups == g)[0]
    plt.scatter(
        Z[idx, 0],
        Z[idx, 1],
        s=20,
        c=colors[g],
        marker=markers[g],
        label=labels_names[g],
        alpha=0.75
    )

plt.title(
    "t-SNE visualization of VGG19 embeddings\n"
    "APEmorph (train/test) and FRLL – 6 groups"
)
plt.xlabel("t-SNE 1")
plt.ylabel("t-SNE 2")
plt.legend(markerscale=1.2)
plt.grid(True)
plt.tight_layout()
plt.savefig(OUT_FIG, dpi=300)
plt.close()

print(" t-SNE saved to:", OUT_FIG)
