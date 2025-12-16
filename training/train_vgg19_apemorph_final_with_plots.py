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

TRAIN_CSV = WORKSPACE / "apemorph_train_balanced.csv"
TEST_CSV  = WORKSPACE / "apemorph_test_balanced.csv"

MODEL_SAVE = WORKSPACE / "vgg19_apemorph_final.pth"
METRICS_TXT = WORKSPACE / "metrics_apemorph_complete.txt"

DIST_05 = WORKSPACE / "score_distribution_thr_0.5.png"
DIST_OPT = WORKSPACE / "score_distribution_thr_optimal.png"

ROC_05 = WORKSPACE / "roc_curve_thr_0.5.png"
ROC_OPT = WORKSPACE / "roc_curve_thr_optimal.png"


# ======================================================
# PARAMETERS
# ======================================================
NUM_EPOCHS = 5
BATCH_SIZE = 32
LR = 1e-3


# ======================================================
# TRANSFORMS
# ======================================================
train_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()
])

test_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])


# ======================================================
# LOAD DATA
# ======================================================
train_df = pd.read_csv(TRAIN_CSV)
test_df  = pd.read_csv(TEST_CSV)

train_set = MorphDataset(train_df, transform=train_tf)
test_set  = MorphDataset(test_df, transform=test_tf)

train_loader = DataLoader(
    train_set,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=2,
    pin_memory=True
)

test_loader = DataLoader(
    test_set,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=2,
    pin_memory=True
)


# ======================================================
# MODEL
# ======================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

model = models.vgg19(weights=models.VGG19_Weights.DEFAULT)

for p in model.features.parameters():
    p.requires_grad = False

model.classifier = nn.Sequential(
    nn.Linear(25088, 2)
)

model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.classifier.parameters(), lr=LR, momentum=0.9)


# ======================================================
# TRAINING
# ======================================================
def train_model():
    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
        model.train()

        running_loss = 0.0
        running_corrects = 0

        for images, labels in tqdm(train_loader, ncols=100, desc="Training"):
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            running_corrects += torch.sum(preds == labels)

        print(
            f"Train Loss: {running_loss/len(train_set):.4f} | "
            f"Train Acc: {running_corrects.double()/len(train_set):.4f}"
        )

    torch.save(model.state_dict(), MODEL_SAVE)
    print(f"\n Model saved to {MODEL_SAVE}")


# ======================================================
# EVALUATION + METRICS + PLOTS
# ======================================================
def evaluate():

    model.eval()
    scores, labels = [], []

    with torch.no_grad():
        for images, gt in tqdm(test_loader, ncols=100, desc="Testing"):
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)

            scores.extend(probs[:, 1].cpu().numpy())
            labels.extend(gt.numpy())

    scores = np.array(scores)
    labels = np.array(labels)

    # ==================================================
    # THRESHOLD = 0.5
    # ==================================================
    preds_05 = (scores >= 0.5).astype(int)

    acc_05 = accuracy_score(labels, preds_05)
    apcer_05 = np.mean((scores < 0.5) & (labels == 1))
    bpcer_05 = np.mean((scores >= 0.5) & (labels == 0))
    acer_05 = 0.5 * (apcer_05 + bpcer_05)

    # ==================================================
    # OPTIMAL THRESHOLD (EER)
    # ==================================================
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

    # ==================================================
    # SAVE METRICS
    # ==================================================
    report = f"""
=== APEmorph Evaluation ===

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

    # ==================================================
    # SCORE DISTRIBUTIONS
    # ==================================================
    def plot_distribution(thr, path, title):
        plt.figure(figsize=(7, 6))
        plt.hist(scores[labels == 0], bins=40, alpha=0.6, label="Bonafide")
        plt.hist(scores[labels == 1], bins=40, alpha=0.6, label="Morph")
        plt.axvline(thr, color="red", linestyle="--", label=f"Threshold = {thr:.3f}")
        plt.xlabel("Morph score")
        plt.ylabel("Frequency")
        plt.title(title)
        plt.legend()
        plt.tight_layout()
        plt.savefig(path, dpi=300)
        plt.close()

    plot_distribution(0.5, DIST_05, "Score Distribution (Threshold = 0.5)")
    plot_distribution(thr_opt, DIST_OPT, "Score Distribution (Optimal Threshold)")

    # ==================================================
    # ROC CURVES
    # ==================================================
    def plot_roc(thr, path, title):
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

    plot_roc(0.5, ROC_05, "ROC Curve (Threshold = 0.5)")
    plot_roc(thr_opt, ROC_OPT, "ROC Curve (Optimal Threshold)")


# ======================================================
# MAIN
# ======================================================
if __name__ == "__main__":
    print(" Training VGG19 on APEmorph")
    train_model()

    print("\n Evaluating and generating plots")
    evaluate()
