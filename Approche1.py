#  APPROCHE 1 : R-CNN (Classification) -> YOLOv11s (Detection)
#  Pipeline: Image -> R-CNN (fire/no fire?) -> If fire -> YOLO (where?)

import os, glob, torch, torch.nn as nn, torch.optim as optim, time
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split, Subset
from ultralytics import YOLO
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.patches as mpatches
import seaborn as sns
import json
import numpy as np
from tqdm import tqdm
from PIL import Image
from collections import defaultdict



# 1. PATHS

CLASS_TRAIN = "/content/drive/MyDrive/MEMOIRE/ForestFireDataset(Classifications)/ForestFireDataset/train"
DETECT_YAML = "/content/drive/MyDrive/MEMOIRE/ForesFireDataset(ObjectDetection)/data.yaml"
OUTPUT_DIR  = "/content/drive/MyDrive/MEMOIRE/Approche1_Results2"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Seuils confidence pour Strong / Medium / Weak
STRONG_THRESH = 0.70
WEAK_THRESH   = 0.30

print("=" * 60)
print("  APPROCHE 1 : R-CNN -> YOLOv11s")
print("=" * 60)
print(f"Classification dataset : {CLASS_TRAIN}")
print(f"Detection YAML         : {DETECT_YAML}")
print(f"Resultats sauvegardes  : {OUTPUT_DIR}")
print(f"Classes trouvees       : {os.listdir(CLASS_TRAIN)}")

# 2. CONFIG

BATCH_SIZE  = 32
NUM_EPOCHS  = 15
LR          = 0.001
NUM_CLASSES = 2
IMG_SIZE    = 224
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nDevice: {DEVICE}")



# 3. DATASET - Split 70 / 15 / 15

tf_train = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
tf_val = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

full_dataset = datasets.ImageFolder(CLASS_TRAIN, tf_train)
CLASS_NAMES  = full_dataset.classes
total        = len(full_dataset)
n_train      = int(0.70 * total)
n_valid      = int(0.15 * total)
n_test       = total - n_train - n_valid

print(f"\nDataset: {total} images | Classes: {CLASS_NAMES}")

train_ds, valid_ds, test_ds = random_split(
    full_dataset, [n_train, n_valid, n_test],
    generator=torch.Generator().manual_seed(42)
)

valid_ds_proper = Subset(datasets.ImageFolder(CLASS_TRAIN, tf_val), valid_ds.indices)
test_ds_proper  = Subset(datasets.ImageFolder(CLASS_TRAIN, tf_val), test_ds.indices)

train_dl = DataLoader(train_ds,        BATCH_SIZE, shuffle=True,  num_workers=2, pin_memory=True)
valid_dl = DataLoader(valid_ds_proper, BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
test_dl  = DataLoader(test_ds_proper,  BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

print(f"Train: {n_train} | Valid: {n_valid} | Test: {n_test}")



# 4. R-CNN MODEL (ResNet50 fine-tuned)

print("\n[1/6] Creation du modele R-CNN (ResNet50)...")

model = models.resnet50(weights="IMAGENET1K_V1")
for name, param in model.named_parameters():
    param.requires_grad = False
for name, param in model.layer4.named_parameters():
    param.requires_grad = True

model.fc = nn.Sequential(
    nn.Linear(model.fc.in_features, 512),
    nn.BatchNorm1d(512),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(512, NUM_CLASSES)
)
model = model.to(DEVICE)
print("R-CNN pret!")

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam([
    {'params': model.layer4.parameters(), 'lr': LR * 0.1},
    {'params': model.fc.parameters(),     'lr': LR}
])
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)



# 5. TRAINING R-CNN

print("\n[2/6] Entrainement R-CNN...")
history    = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
best_acc   = 0.0
best_epoch = 0

for epoch in range(NUM_EPOCHS):
    model.train()
    tl, tc = 0.0, 0
    for imgs, lbls in tqdm(train_dl, desc=f"Epoch {epoch+1:02d}/{NUM_EPOCHS} [Train]", leave=False):
        imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
        optimizer.zero_grad()
        out  = model(imgs)
        loss = criterion(out, lbls)
        loss.backward()
        optimizer.step()
        tl += loss.item() * imgs.size(0)
        tc += (out.argmax(1) == lbls).sum().item()

    model.eval()
    vl, vc = 0.0, 0
    with torch.no_grad():
        for imgs, lbls in valid_dl:
            imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
            out = model(imgs)
            vl += criterion(out, lbls).item() * imgs.size(0)
            vc += (out.argmax(1) == lbls).sum().item()

    ta = tc / n_train * 100
    va = vc / n_valid * 100
    history["train_loss"].append(tl / n_train)
    history["train_acc"].append(ta)
    history["val_loss"].append(vl / n_valid)
    history["val_acc"].append(va)
    print(f"Epoch {epoch+1:02d} | Train Loss: {tl/n_train:.4f} | Train Acc: {ta:.1f}% | Val Acc: {va:.1f}%")

    if va > best_acc:
        best_acc   = va
        best_epoch = epoch + 1
        torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "rcnn_best.pth"))
        print(f"   Best model saved! (Val Acc: {best_acc:.1f}%)")

    scheduler.step()

print(f"\nMeilleur modele: Epoch {best_epoch} avec Val Acc = {best_acc:.1f}%")



# 6. EVALUATION R-CNN (TEST SET)

print("\n[3/6] Evaluation R-CNN sur le test set...")
model.load_state_dict(torch.load(os.path.join(OUTPUT_DIR, "rcnn_best.pth")))
model.eval()

preds, labels, probs_list = [], [], []
with torch.no_grad():
    for imgs, lbls in test_dl:
        out = model(imgs.to(DEVICE))
        probs = torch.softmax(out, dim=1)
        preds.extend(out.argmax(1).cpu().numpy())
        labels.extend(lbls.numpy())
        probs_list.extend(probs.cpu().numpy())

print("\n" + "="*50)
print("RAPPORT R-CNN (Classification)")
print("="*50)
report = classification_report(labels, preds, target_names=CLASS_NAMES, digits=4)
print(report)
rcnn_acc = sum(p == l for p, l in zip(preds, labels)) / len(labels) * 100
print(f"Accuracy globale: {rcnn_acc:.2f}%")

with open(os.path.join(OUTPUT_DIR, "rcnn_classification_report.txt"), "w") as f:
    f.write("RAPPORT R-CNN - Approche 1\n" + "="*50 + "\n")
    f.write(report)
    f.write(f"\nAccuracy: {rcnn_acc:.2f}%\n")

# --- Confusion Matrix R-CNN ---
cm = confusion_matrix(labels, preds)
plt.figure(figsize=(7, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
            annot_kws={"size": 14})
plt.title("R-CNN - Matrice de Confusion (Approche 1)", fontsize=14, fontweight='bold')
plt.xlabel("Prediction", fontsize=12)
plt.ylabel("Realite", fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "rcnn_confusion_matrix.png"), dpi=200, bbox_inches='tight')
plt.show()
print("Confusion matrix R-CNN sauvegardee")

# --- Learning Curves ---
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("R-CNN - Courbes d'Apprentissage (Approche 1)", fontsize=14, fontweight='bold')
epochs_range = range(1, NUM_EPOCHS + 1)

axes[0].plot(epochs_range, history["train_acc"],  'b-o', label="Train", markersize=4)
axes[0].plot(epochs_range, history["val_acc"],    'r-o', label="Validation", markersize=4)
axes[0].axvline(x=best_epoch, color='green', linestyle='--', label=f"Best (Ep.{best_epoch})")
axes[0].set_title("Accuracy"); axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Accuracy (%)")
axes[0].legend(); axes[0].grid(True, alpha=0.4)

axes[1].plot(epochs_range, history["train_loss"], 'b-o', label="Train", markersize=4)
axes[1].plot(epochs_range, history["val_loss"],   'r-o', label="Validation", markersize=4)
axes[1].axvline(x=best_epoch, color='green', linestyle='--', label=f"Best (Ep.{best_epoch})")
axes[1].set_title("Loss"); axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Loss")
axes[1].legend(); axes[1].grid(True, alpha=0.4)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "rcnn_learning_curves.png"), dpi=200, bbox_inches='tight')
plt.show()
print("Learning curves sauvegardees")



# 7. YOLO TRAINING (yolo11s)

print("\n[4/6] Entrainement YOLOv11s...")
yolo = YOLO("yolo11s.pt")
yolo.train(
    data     = DETECT_YAML,
    epochs   = 50,
    imgsz    = 640,
    batch    = 16,
    name     = "approche1_yolo",
    project  = os.path.join(OUTPUT_DIR, "yolo_runs"),
    patience = 10,
    exist_ok = True,
    device   = 0 if torch.cuda.is_available() else 'cpu',
    save     = True,
)


# 8. YOLO EVALUATION GLOBALE

print("\n[5/6] Evaluation YOLOv11s...")

best_pt_list = glob.glob(os.path.join(OUTPUT_DIR, "yolo_runs/**/best.pt"), recursive=True)
if not best_pt_list:
    best_pt_list = glob.glob("/content/**/best.pt", recursive=True)
if not best_pt_list:
    raise FileNotFoundError("best.pt introuvable!")

best_pt_path = best_pt_list[0]
print(f"best.pt trouve: {best_pt_path}")

yolo_best = YOLO(best_pt_path)
metrics   = yolo_best.val(data=DETECT_YAML, split="test")

yolo_map50     = float(metrics.box.map50) * 100
yolo_map5095   = float(metrics.box.map)   * 100
yolo_precision = float(metrics.box.mp)    * 100
yolo_recall    = float(metrics.box.mr)    * 100

print("\n" + "="*50)
print("RAPPORT YOLO (Detection)")
print("="*50)
print(f"mAP@0.5      : {yolo_map50:.2f}%")
print(f"mAP@0.5:0.95 : {yolo_map5095:.2f}%")
print(f"Precision    : {yolo_precision:.2f}%")
print(f"Recall       : {yolo_recall:.2f}%")



# HELPER: confidence -> classe intensite

def conf_to_intensity(conf):
    """Retourne (index, label, couleur) selon le confidence score."""
    if conf >= STRONG_THRESH:
        return 0, "Strong Fire", "#c0392b"
    elif conf >= WEAK_THRESH:
        return 1, "Medium Fire", "#e67e22"
    else:
        return 2, "Weak Fire",   "#f1c40f"


# 9. CONFUSION MATRIX YOLO DETECTION
#    Lignes = Ground Truth (fire/nofire)
#    Colonnes = Intensite predite (Strong/Medium/Weak/No Detection)

print("\n[6/6] Matrice de confusion detection YOLO (Strong/Medium/Weak)...")

DETECT_TEST_IMG = "/content/drive/MyDrive/MEMOIRE/ForesFireDataset(ObjectDetection)/test/images"
DETECT_TEST_LBL = "/content/drive/MyDrive/MEMOIRE/ForesFireDataset(ObjectDetection)/test/labels"

# Colonnes: 0=Strong, 1=Medium, 2=Weak, 3=No Detection
# Lignes  : 0=Fire (a des annotations), 1=Background (pas d annotations)
confusion_det = np.zeros((2, 4), dtype=int)
conf_all      = []
counts        = defaultdict(int)
no_detection  = 0

img_files = sorted(glob.glob(os.path.join(DETECT_TEST_IMG, "*.jpg")) +
                   glob.glob(os.path.join(DETECT_TEST_IMG, "*.png")))

for img_path in img_files:
    # Verifier si l image a des annotations (ground truth = fire)
    lbl_path = os.path.join(
        DETECT_TEST_LBL,
        os.path.splitext(os.path.basename(img_path))[0] + ".txt"
    )
    has_annotation = os.path.exists(lbl_path) and os.path.getsize(lbl_path) > 0
    gt_row = 0 if has_annotation else 1   # 0=Fire, 1=Background

    # Inference YOLO
    result = yolo_best(img_path, verbose=False)[0]

    if result.boxes is None or len(result.boxes) == 0:
        confusion_det[gt_row][3] += 1     # No Detection
        no_detection += 1
    else:
        for box in result.boxes:
            conf = float(box.conf[0])
            conf_all.append(conf)
            idx, _, _ = conf_to_intensity(conf)
            confusion_det[gt_row][idx] += 1
            counts[idx] += 1

total_boxes = sum(counts.values())
print(f"Total bounding boxes : {total_boxes}")
print(f"Images sans detection: {no_detection}")
print(f"Strong Fire (>70%)   : {counts[0]}")
print(f"Medium Fire (30-70%) : {counts[1]}")
print(f"Weak Fire   (<30%)   : {counts[2]}")


# --- FIGURE A : Matrice de Confusion Detection ---
col_labels = ["Strong Fire\n(conf>70%)", "Medium Fire\n(30-70%)", "Weak Fire\n(conf<30%)", "No\nDetection"]
row_labels  = ["Fire\n(annote)", "Background\n(non annote)"]

# Normalisation en %
row_sums = confusion_det.sum(axis=1, keepdims=True).astype(float)
row_sums[row_sums == 0] = 1
confusion_pct = confusion_det / row_sums * 100

annot = np.empty_like(confusion_det, dtype=object)
for i in range(2):
    for j in range(4):
        n   = confusion_det[i, j]
        pct = confusion_pct[i, j]
        annot[i, j] = f"{n}\n({pct:.1f}%)"

fig_cm, ax_cm = plt.subplots(figsize=(11, 5))
sns.heatmap(
    confusion_pct,
    annot      = annot,
    fmt        = "",
    cmap       = "YlOrRd",
    linewidths = 0.8,
    linecolor  = "white",
    ax         = ax_cm,
    xticklabels= col_labels,
    yticklabels= row_labels,
    vmin=0, vmax=100,
    cbar_kws   = {"label": "% des images", "shrink": 0.8}
)
ax_cm.set_title(
    "Matrice de Confusion - Detection YOLO par Niveau de Confiance\n"
    "Approche 1 : R-CNN + YOLOv11s",
    fontsize=13, fontweight="bold", pad=14
)
ax_cm.set_xlabel("Niveau de Confiance Predit", fontsize=12, labelpad=10)
ax_cm.set_ylabel("Ground Truth", fontsize=12, labelpad=10)
ax_cm.tick_params(axis="x", labelsize=10)
ax_cm.tick_params(axis="y", labelsize=10, rotation=0)

legend_patches = [
    mpatches.Patch(color="#c0392b", label="Strong Fire  - conf > 70%  : modele tres certain"),
    mpatches.Patch(color="#e67e22", label="Medium Fire - conf 30-70% : modele incertain"),
    mpatches.Patch(color="#f1c40f", label="Weak Fire    - conf < 30%  : possible fausse alarme"),
    mpatches.Patch(color="#95a5a6", label="No Detection               : aucun feu detecte"),
]
ax_cm.legend(handles=legend_patches, loc="upper right",
             bbox_to_anchor=(1.0, -0.22), ncol=2, fontsize=9, frameon=False)

plt.tight_layout()
path_cm = os.path.join(OUTPUT_DIR, "yolo_detection_confusion_matrix.png")
plt.savefig(path_cm, dpi=200, bbox_inches="tight")
plt.show()
print(f"Confusion matrix detection sauvegardee: {path_cm}")


# --- FIGURE B : Distribution des confidence scores ---

if conf_all:
    fig_dist, ax_dist = plt.subplots(figsize=(10, 5))
    n_hist, bins_hist, hist_patches = ax_dist.hist(conf_all, bins=30,
                                                    edgecolor="white", linewidth=0.5)
    for patch, left in zip(hist_patches, bins_hist[:-1]):
        _, _, color = conf_to_intensity(left)
        patch.set_facecolor(color)

    ax_dist.axvline(STRONG_THRESH, color="#7B0000", linestyle="--",
                    linewidth=1.8, label=f"Seuil Strong = {STRONG_THRESH}")
    ax_dist.axvline(WEAK_THRESH,   color="#5D3800", linestyle="--",
                    linewidth=1.8, label=f"Seuil Weak   = {WEAK_THRESH}")

    y_max = n_hist.max()
    ax_dist.text(0.15, y_max * 0.85, "Weak Fire",   color="#5D3800", fontsize=11, fontweight="bold")
    ax_dist.text(0.43, y_max * 0.85, "Medium Fire", color="#7A4200", fontsize=11, fontweight="bold")
    ax_dist.text(0.73, y_max * 0.85, "Strong Fire", color="#7B0000", fontsize=11, fontweight="bold")

    ax_dist.set_xlabel("Confidence Score", fontsize=12)
    ax_dist.set_ylabel("Nombre de detections", fontsize=12)
    ax_dist.set_title(
        "Distribution des Confidence Scores - YOLOv11s\nApproche 1 : R-CNN + YOLOv11s",
        fontsize=13, fontweight="bold"
    )
    ax_dist.legend(fontsize=10)
    ax_dist.grid(True, alpha=0.3)
    plt.tight_layout()
    path_dist = os.path.join(OUTPUT_DIR, "yolo_confidence_distribution.png")
    plt.savefig(path_dist, dpi=200, bbox_inches="tight")
    plt.show()
    print(f"Distribution confidence sauvegardee: {path_dist}")
    avg_conf = np.mean(conf_all) * 100
else:
    avg_conf = 0.0
    print("Aucune detection - distribution non generee")


# 10. PIPELINE VISUALISATION (Bounding Boxes colores)

print("\nVisualisation pipeline R-CNN -> YOLO avec couleurs intensite...")

test_img_paths  = []
test_img_labels_list = []
base_dataset    = datasets.ImageFolder(CLASS_TRAIN)

for idx in test_ds.indices:
    path, label = base_dataset.samples[idx]
    test_img_paths.append(path)
    test_img_labels_list.append(label)

fire_class_idx = CLASS_NAMES.index('fire') if 'fire' in CLASS_NAMES else 0
fire_samples   = [(p, l) for p, l in zip(test_img_paths, test_img_labels_list)
                  if l == fire_class_idx][:8]

tf_inference = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

fig_vis, axes_vis = plt.subplots(2, 4, figsize=(18, 9))
fig_vis.suptitle("Pipeline R-CNN -> YOLO : Detection par Niveau d'Intensite (Approche 1)",
                 fontsize=13, fontweight='bold')
model.eval()

for i, (img_path, _) in enumerate(fire_samples):
    ax = axes_vis[i // 4][i % 4]
    pil_img = Image.open(img_path).convert("RGB")

    inp = tf_inference(pil_img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        out        = model(inp)
        probs      = torch.softmax(out, dim=1)[0]
        pred_class = out.argmax(1).item()

    if pred_class == fire_class_idx:
        result = yolo_best(img_path, verbose=False)[0]
        ax.imshow(pil_img)

        if result.boxes is not None and len(result.boxes) > 0:
            n_strong, n_medium, n_weak = 0, 0, 0
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0])
                _, intensity_label, color = conf_to_intensity(conf)

                if conf >= STRONG_THRESH: n_strong += 1
                elif conf >= WEAK_THRESH: n_medium += 1
                else:                     n_weak   += 1

                w_scale = pil_img.width  / result.orig_shape[1]
                h_scale = pil_img.height / result.orig_shape[0]

                rect = patches.Rectangle(
                    (x1 * w_scale, y1 * h_scale),
                    (x2 - x1) * w_scale, (y2 - y1) * h_scale,
                    linewidth=2.5, edgecolor=color, facecolor='none'
                )
                ax.add_patch(rect)
                ax.text(x1 * w_scale, y1 * h_scale - 5,
                        f"{intensity_label} {conf:.2f}",
                        color='white', fontsize=7, fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.2',
                                  facecolor=color, alpha=0.85))

            title_parts = []
            if n_strong: title_parts.append(f"Strong:{n_strong}")
            if n_medium: title_parts.append(f"Medium:{n_medium}")
            if n_weak:   title_parts.append(f"Weak:{n_weak}")
            ax.set_title(f"R-CNN:{probs[fire_class_idx]*100:.0f}% | " + " | ".join(title_parts),
                         fontsize=7, color='darkred', fontweight='bold')
        else:
            ax.set_title(f"R-CNN: Fire {probs[fire_class_idx]*100:.0f}%\nYOLO: 0 bbox",
                         fontsize=8, color='orange')
    else:
        ax.imshow(pil_img)
        ax.set_title(f"R-CNN: No Fire\nYOLO ignore", fontsize=8, color='gray')

    ax.axis('off')

# Legende couleurs en bas
legend_vis = [
    mpatches.Patch(color="#c0392b", label="Strong Fire (conf>70%)"),
    mpatches.Patch(color="#e67e22", label="Medium Fire (30-70%)"),
    mpatches.Patch(color="#f1c40f", label="Weak Fire (<30%)"),
]
fig_vis.legend(handles=legend_vis, loc='lower center', ncol=3,
               fontsize=10, bbox_to_anchor=(0.5, -0.02), frameon=False)

plt.tight_layout()
path_vis = os.path.join(OUTPUT_DIR, "pipeline_rcnn_yolo_visualizations.png")
plt.savefig(path_vis, dpi=200, bbox_inches='tight')
plt.show()
print(f"Visualisations pipeline sauvegardees: {path_vis}")


# 11. BENCHMARK TIMING

print("\nMesure du temps de traitement du pipeline...")
times_rcnn, times_yolo, times_total = [], [], []

for img_path, _ in fire_samples[:min(50, len(fire_samples))]:
    pil_img = Image.open(img_path).convert("RGB")
    inp = tf_inference(pil_img).unsqueeze(0).to(DEVICE)

    t0 = time.time()
    with torch.no_grad():
        out  = model(inp)
        pred = out.argmax(1).item()
    t_rcnn = time.time() - t0

    t1 = time.time()
    if pred == fire_class_idx:
        _ = yolo_best(img_path, verbose=False)
    t_yolo = time.time() - t1

    times_rcnn.append(t_rcnn * 1000)
    times_yolo.append(t_yolo * 1000)
    times_total.append((t_rcnn + t_yolo) * 1000)

avg_rcnn  = np.mean(times_rcnn)
avg_yolo  = np.mean(times_yolo)
avg_total = np.mean(times_total)
print(f"Temps moyen R-CNN   : {avg_rcnn:.1f} ms")
print(f"Temps moyen YOLO    : {avg_yolo:.1f} ms")
print(f"Temps total pipeline: {avg_total:.1f} ms  ({1000/avg_total:.1f} FPS)")


# 12. RÉSUMÉ FINAL JSON

strong_pct = counts[0] / total_boxes * 100 if total_boxes > 0 else 0
medium_pct = counts[1] / total_boxes * 100 if total_boxes > 0 else 0
weak_pct   = counts[2] / total_boxes * 100 if total_boxes > 0 else 0

results_summary = {
    "Approche": "Approche 1 - R-CNN + YOLOv11s",
    "Classification": {
        "Modele"           : "ResNet50 (R-CNN)",
        "Accuracy"         : round(rcnn_acc, 2),
        "Best_Val_Accuracy": round(best_acc, 2),
        "Best_Epoch"       : best_epoch,
    },
    "Detection": {
        "Modele"           : "YOLOv11s",
        "mAP_50"           : round(yolo_map50, 2),
        "mAP_50_95"        : round(yolo_map5095, 2),
        "Precision"        : round(yolo_precision, 2),
        "Recall"           : round(yolo_recall, 2),
        "Avg_Confidence"   : round(avg_conf, 2),
    },
    "Fire_Intensity": {
        "Strong_Fire_pct"  : round(strong_pct, 1),
        "Medium_Fire_pct"  : round(medium_pct, 1),
        "Weak_Fire_pct"    : round(weak_pct,   1),
        "No_Detection"     : no_detection,
        "Total_Boxes"      : total_boxes,
    },
    "Pipeline_Timing_ms": {
        "R_CNN_avg"        : round(avg_rcnn,  2),
        "YOLO_avg"         : round(avg_yolo,  2),
        "Total_avg"        : round(avg_total, 2),
        "FPS_approx"       : round(1000 / avg_total, 1),
    }
}

json_path = os.path.join(OUTPUT_DIR, "approche1_results.json")
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(results_summary, f, indent=2, ensure_ascii=False)

print("\n" + "=" * 60)
print("  APPROCHE 1 TERMINEE - RESULTATS FINAUX")
print("=" * 60)
print(json.dumps(results_summary, indent=2, ensure_ascii=False))
print(f"\nTous les fichiers sauvegardes dans: {OUTPUT_DIR}")
print("\nFichiers generes:")
print("  rcnn_confusion_matrix.png            -> Chapitre 7 (Classification)")
print("  rcnn_learning_curves.png             -> Chapitre 7 (Classification)")
print("  yolo_detection_confusion_matrix.png  -> Chapitre 7 (Detection)")
print("  yolo_confidence_distribution.png     -> Chapitre 7 (Detection)")
print("  pipeline_rcnn_yolo_visualizations.png-> Chapitre 6 (Implementation)")
print("  approche1_results.json               -> Comparaison avec Approche 2")
