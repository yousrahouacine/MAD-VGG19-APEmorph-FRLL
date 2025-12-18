import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, models

from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score
from tqdm import tqdm

from morph_dataset import MorphDataset


# ======================================================
# PATHS
# ======================================================
WORKSPACE = Path("/content/drive/MyDrive/workspace3")

FRLL_CSV = WORKSPACE / "frll_test_mtcnn.csv"
MODEL_PATH = WORKSPACE / "vgg19_apemorph_fc7.pth"

METRICS_TXT = WORKSPACE / "metrics_frll_fc7.txt"

DIST_05 = WORKSPACE / "frll_fc7_dist_thr_0.5.png"
DIST_OPT = WORKSPACE / "frll_fc7_dist_thr_optimal.png"

ROC_05 = WORKSPACE / "frll_fc7_roc_thr_0.5.png"
ROC_OPT = WORKSPACE / "frll_fc7_roc_thr_optimal.png"


# ======================================================
# PARAMETERS
# ======================================================
BATCH_SIZE = 32


# ======================================================
# TRANSFORMS
# ======================================================
test_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])


# ======================================================
# LOAD FRLL DATA
# ======================================================
frll_df = pd.read_csv(FRLL_CSV)
frll_set = MorphDataset(frll_df, transform=test_tf)

frll_loader = DataLoader(
    frll_set,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=2,
    pin_memory=True
)


# ======================================================
# LOAD MODEL (FC7 ARCHITECTURE)
# ======================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

model = models.vgg19(weights=None)

model.classifier = nn.Sequential(
    nn.Linear(25088, 4096),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(4096, 2)
)

model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model = model.to(device)
model.eval()

print("Loaded FC7 VGG19 model")


# ======================================================
# EVALUATION
# ======================================================
scores, labels = [], []

with torch.no_grad():
    for images, gt in tqdm(frll_loader, ncols=100, desc="Testing FRLL (FC7)"):
        images = images.to(device)
        outputs = model(images)
        probs = torch.softmax(outputs, dim=1)

        scores.extend(probs[:, 1].cpu().numpy())
        labels.extend(gt.numpy())

scores = np.array(scores)
labels = np.array(labels)


# ======================================================
# THRESHOLD = 0.5
# ======================================================
preds_05 = (scores >= 0.5).astype(int)

acc_05 = accuracy_score(labels, preds_05)
apcer_05 = np.mean((scores < 0.5) & (labels == 1))
bpcer_05 = np.mean((scores >= 0.5) & (labels == 0))
acer_05 = 0.5 * (apcer_05 + bpcer_05)


# ======================================================
# OPTIMAL THRESHOLD (EER)
# ======================================================
fpr, tpr, thresholds = roc_curve(labels, scores)
fnr = 1 - tpr
idx = np.nanargmin(np.abs(fpr - fnr))

thr_opt = thresholds[idx]
eer = fpr[idx]
auc = roc_auc_score(labels, scores)

preds_opt = (scores >= thr_opt).astype(int)

acc_opt = accuracy_score(labels, preds_opt)
apcer_opt = np.mean((scores < thr_opt) & (labels == 1))
bpcer_opt = np.mean((scores >= thr_opt) & (labels == 0))
acer_opt = 0.5 * (apcer_opt + bpcer_opt)


# ======================================================
# SAVE METRICS
# ======================================================
report = f"""
=== FRLL Evaluation (APEmorph-trained VGG19 FC7) ===

--- Threshold = 0.5 ---
Accuracy : {acc_05:.4f}
APCER    : {apcer_05:.4f}
BPCER    : {bpcer_05:.4f}
ACER     : {acer_05:.4f}

--- Optimal Threshold ---
Threshold* : {thr_opt:.4f}
Accuracy*  : {acc_opt:.4f}
APCER*     : {apcer_opt:.4f}
BPCER*     : {bpcer_opt:.4f}
ACER*      : {acer_opt:.4f}
EER        : {eer:.4f}
AUC        : {auc:.4f}
"""

print(report)
with open(METRICS_TXT, "w") as f:
    f.write(report)


# ======================================================
# PLOTS
# ======================================================
def plot_distribution(thr, path, title):
    plt.figure(figsize=(7, 6))
    plt.hist(scores[labels == 0], bins=40, alpha=0.6, label="Bonafide")
    plt.hist(scores[labels == 1], bins=40, alpha=0.6, label="Morph")
    plt.axvline(thr, color="red", linestyle="--")
    plt.xlabel("Morph score")
    plt.ylabel("Frequency")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()

plot_distribution(0.5, DIST_05, "FRLL FC7 Score Distribution (Threshold = 0.5)")
plot_distribution(thr_opt, DIST_OPT, "FRLL FC7 Score Distribution (Optimal Threshold)")


def plot_roc(path, title):
    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    plt.scatter(fpr[idx], tpr[idx], color="red", label="Optimal threshold")
    plt.plot([0, 1], [0, 1], "--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()

plot_roc(ROC_05, "FRLL FC7 ROC Curve (Threshold = 0.5)")
plot_roc(ROC_OPT, "FRLL FC7 ROC Curve (Optimal Threshold)")

print("FRLL FC7 evaluation completed")
