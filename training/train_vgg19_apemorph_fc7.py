import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, models

from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from morph_dataset import MorphDataset


# ======================================================
# PATHS
# ======================================================
WORKSPACE = Path("/content/drive/MyDrive/workspace3")

TRAIN_CSV = WORKSPACE / "apemorph_train_balanced_mtcnn.csv"
TEST_CSV  = WORKSPACE / "apemorph_test_balanced_mtcnn.csv"

MODEL_SAVE = WORKSPACE / "vgg19_apemorph_fc7.pth"
CHECKPOINT_PATH = WORKSPACE / "vgg19_apemorph_fc7_checkpoint.pth"

METRICS_TXT = WORKSPACE / "metrics_apemorph_fc7.txt"

DIST_05  = WORKSPACE / "score_distribution_fc7_thr_0.5.png"
DIST_OPT = WORKSPACE / "score_distribution_fc7_thr_optimal.png"

ROC_05  = WORKSPACE / "roc_curve_fc7_thr_0.5.png"
ROC_OPT = WORKSPACE / "roc_curve_fc7_thr_optimal.png"

LOSS_PLOT = WORKSPACE / "training_loss_fc7.png"
ACC_PLOT  = WORKSPACE / "training_accuracy_fc7.png"


# ======================================================
# PARAMETERS
# ======================================================
NUM_EPOCHS = 100
BATCH_SIZE = 32
LR = 1e-3

PATIENCE = 10
MIN_DELTA = 1e-3


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
# LOAD DATA + TRAIN / VAL SPLIT
# ======================================================
train_df_full = pd.read_csv(TRAIN_CSV)
test_df = pd.read_csv(TEST_CSV)

train_df, val_df = train_test_split(
    train_df_full,
    test_size=0.2,
    stratify=train_df_full["label"],
    random_state=42
)

print(f"Train samples      : {len(train_df)}")
print(f"Validation samples : {len(val_df)}")
print(f"Test samples       : {len(test_df)}")

train_set = MorphDataset(train_df, transform=train_tf)
val_set   = MorphDataset(val_df, transform=test_tf)
test_set  = MorphDataset(test_df, transform=test_tf)

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2, pin_memory=True)
val_loader   = DataLoader(val_set,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
test_loader  = DataLoader(test_set,  batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)


# ======================================================
# MODEL WITH FC7 (4096D)
# ======================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

model = models.vgg19(weights=models.VGG19_Weights.DEFAULT)

# Freeze all conv layers
for p in model.features.parameters():
    p.requires_grad = False

# UNFREEZE ONLY LAST CONV BLOCK
for p in model.features[28:].parameters():
    p.requires_grad = True

model.classifier = nn.Sequential(
    nn.Linear(25088, 4096),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(4096, 2)
)

model = model.to(device)

criterion = nn.CrossEntropyLoss()

# OPTIMIZER: classifier + last conv block
optimizer = torch.optim.SGD(
    list(model.classifier.parameters()) +
    list(model.features[28:].parameters()),
    lr=LR,
    momentum=0.9
)


# ======================================================
# EARLY STOPPING VARIABLES
# ======================================================
best_val_loss = float("inf")
best_val_acc = 0.0
patience_counter = 0

train_losses = []
train_accuracies = []
val_losses = []


# ======================================================
# VALIDATION
# ======================================================
def compute_val_loss_and_acc():
    model.eval()
    loss_sum = 0.0
    correct = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            _, preds = torch.max(outputs, 1)
            loss_sum += loss.item() * images.size(0)
            correct += torch.sum(preds == labels)

    return loss_sum / len(val_set), (correct.double() / len(val_set)).item()


# ======================================================
# TRAINING
# ======================================================
def train_model():
    global best_val_loss, best_val_acc, patience_counter

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

        train_loss = running_loss / len(train_set)
        train_acc = (running_corrects.double() / len(train_set)).item()

        val_loss, val_acc = compute_val_loss_and_acc()

        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        val_losses.append(val_loss)

        print(f"Train Loss: {train_loss:.6f} | Train Acc: {train_acc:.4f}")
        print(f"Val   Loss: {val_loss:.6f} | Val   Acc: {val_acc:.4f}")

        if (val_loss < best_val_loss - MIN_DELTA) or (val_acc > best_val_acc + 1e-3):
            best_val_loss = min(best_val_loss, val_loss)
            best_val_acc = max(best_val_acc, val_acc)
            patience_counter = 0
            torch.save(model.state_dict(), MODEL_SAVE)
            print("New best model (real improvement). Saved.")
        else:
            patience_counter += 1
            print(f"Early stopping patience: {patience_counter}/{PATIENCE}")

        if patience_counter >= PATIENCE:
            print("Early stopping triggered.")
            break


# ======================================================
# TRAINING CURVES
# ======================================================
def plot_training_curves():
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(7, 5))
    plt.plot(epochs, train_losses, label="Train Loss")
    plt.plot(epochs, val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Curve (FC7)")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(LOSS_PLOT, dpi=300)
    plt.close()

    plt.figure(figsize=(7, 5))
    plt.plot(epochs, train_accuracies, label="Train Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training Accuracy Curve (FC7)")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(ACC_PLOT, dpi=300)
    plt.close()


# ======================================================
# EVALUATION + METRICS + PLOTS 
# ======================================================
def evaluate():
    model.load_state_dict(torch.load(MODEL_SAVE, map_location=device))
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

    preds_05 = (scores >= 0.5).astype(int)

    acc_05 = accuracy_score(labels, preds_05)
    apcer_05 = np.mean((scores < 0.5) & (labels == 1))
    bpcer_05 = np.mean((scores >= 0.5) & (labels == 0))
    acer_05 = 0.5 * (apcer_05 + bpcer_05)

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

    report = f"""
=== APEmorph Evaluation (FC7 – 4096D) ===

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

    def plot_distribution(thr, path, title):
        plt.figure(figsize=(7, 6))
        plt.hist(scores[labels == 0], bins=40, alpha=0.6, label="Bonafide")
        plt.hist(scores[labels == 1], bins=40, alpha=0.6, label="Morph")
        plt.axvline(thr, color="red", linestyle="--")
        plt.title(title)
        plt.legend()
        plt.tight_layout()
        plt.savefig(path, dpi=300)
        plt.close()

    plot_distribution(0.5, DIST_05, "FC7 Score Distribution (Threshold = 0.5)")
    plot_distribution(thr_opt, DIST_OPT, "FC7 Score Distribution (Optimal Threshold)")

    def plot_roc(path, title):
        plt.figure(figsize=(7, 6))
        plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
        plt.scatter(fpr[idx], tpr[idx], color="red")
        plt.plot([0, 1], [0, 1], "--")
        plt.title(title)
        plt.legend()
        plt.grid()
        plt.tight_layout()
        plt.savefig(path, dpi=300)
        plt.close()

    plot_roc(ROC_05, "FC7 ROC Curve (Threshold = 0.5)")
    plot_roc(ROC_OPT, "FC7 ROC Curve (Optimal Threshold)")


# ======================================================
# MAIN
# ======================================================
if __name__ == "__main__":
    print("Training VGG19 with FC7 (4096D)")
    train_model()

    plot_training_curves()

    print("\nEvaluating and generating metrics/plots")
    evaluate()
