# ============================================================
#  APPROCHE YOLO11 ONLY : Classification + Détection
#  Classification : yolo11n-cls.pt fine-tune 100 epochs
#  Detection      : best_nano_111.pt fine-tune 100 epochs
#  Dataset        : Shamta & Demir 2024
# ============================================================

import os, glob, torch, time, json, shutil, random, csv
from ultralytics import YOLO
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.patches as mpatches
import seaborn as sns
import numpy as np
from PIL import Image
from collections import defaultdict

# ---------------------------------------------
# FIX GOOGLE DRIVE SHORTCUT PATH
# ---------------------------------------------
def find_real_path(relative_path):
    normal = f"/content/drive/MyDrive/{relative_path}"
    if os.path.exists(normal):
        return normal
    shortcuts = glob.glob("/content/drive/.shortcut-targets-by-id/*/")
    for shortcut in shortcuts:
        candidate = os.path.join(shortcut, relative_path)
        if os.path.exists(candidate):
            return candidate
    return normal

BASE_CLS = find_real_path("MEMOIRE/ForestFireDataset(Classifications)/ForestFireDataset")
BASE_DET = find_real_path("MEMOIRE/ForesFireDataset(ObjectDetection)")
BASE_OUT = find_real_path("MEMOIRE/YOLO2_YOLO11_Only_1epochs")

# ---------------------------------------------
# CRÉER SPLIT 70/15/15 AUTOMATIQUEMENT
# ---------------------------------------------
SPLIT_DIR  = f"{BASE_CLS}/split_final"
TRAIN_SRC  = f"{BASE_CLS}/train"
CLASSES    = sorted([d for d in os.listdir(TRAIN_SRC)
                     if os.path.isdir(os.path.join(TRAIN_SRC, d))])

if not os.path.exists(os.path.join(SPLIT_DIR, "train")):
    print("Création du split 70/15/15...")
    random.seed(42)
    for split in ["train", "valid", "test"]:
        for cls in CLASSES:
            os.makedirs(os.path.join(SPLIT_DIR, split, cls), exist_ok=True)
    for cls in CLASSES:
        cls_dir = os.path.join(TRAIN_SRC, cls)
        imgs    = [f for f in os.listdir(cls_dir)
                   if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        random.shuffle(imgs)
        n       = len(imgs)
        n_train = int(0.70 * n)
        n_valid = int(0.15 * n)
        splits  = {
            "train" : imgs[:n_train],
            "valid" : imgs[n_train:n_train + n_valid],
            "test"  : imgs[n_train + n_valid:]
        }
        for split_name, files in splits.items():
            for fname in files:
                shutil.copy2(
                    os.path.join(cls_dir, fname),
                    os.path.join(SPLIT_DIR, split_name, cls, fname)
                )
        print(f"  {cls}: train={len(splits['train'])} | valid={len(splits['valid'])} | test={len(splits['test'])}")
    print("Split créé!")
else:
    print(f"Split déjà existant : {SPLIT_DIR}")

# ---------------------------------------------
# PATHS & CONFIG
# ---------------------------------------------
CLASS_TRAIN     = f"{SPLIT_DIR}/train"
CLASS_VAL       = f"{SPLIT_DIR}/valid"
CLASS_TEST      = f"{SPLIT_DIR}/test"
DETECT_YAML     = f"{BASE_DET}/data.yaml"
DETECT_TEST_IMG = f"{BASE_DET}/test/images"
DETECT_TEST_LBL = f"{BASE_DET}/test/labels"
OUTPUT_DIR      = BASE_OUT
os.makedirs(OUTPUT_DIR, exist_ok=True)

STRONG_THRESH = 0.70
WEAK_THRESH   = 0.30
DEVICE = 0 if torch.cuda.is_available() else "cpu"

print("=" * 60)
print("  APPROCHE YOLO11 ONLY : Classification + Détection")
print("=" * 60)
print(f"CLASS_TRAIN : {CLASS_TRAIN}")
print(f"DETECT_YAML : {DETECT_YAML}")
print(f"OUTPUT_DIR  : {OUTPUT_DIR}")
print(f"Device      : {'GPU' if torch.cuda.is_available() else 'CPU'}")
print(f"valid/ : {os.path.exists(CLASS_VAL)}")
print(f"test/  : {os.path.exists(CLASS_TEST)}")

# ---------------------------------------------
# HELPER
# ---------------------------------------------
def conf_to_intensity(conf):
    if conf >= STRONG_THRESH:
        return 0, "Strong Fire", "#c0392b"
    elif conf >= WEAK_THRESH:
        return 1, "Medium Fire", "#e67e22"
    else:
        return 2, "Weak Fire",   "#f1c40f"

# ---------------------------------------------
# CHARGEMENT MODELE DETECTION PRE-ENTRAINE
# ---------------------------------------------
print("\n[1/6] Chargement du modèle détection pré-entraîné...")

REPO_PATH      = "/content/smoke-fire-yolo"
DET_MODEL_NAME = "best_nano_111.pt"
REPO_MODEL     = os.path.join(REPO_PATH, "models", DET_MODEL_NAME)
DET_PRETRAINED = os.path.join(OUTPUT_DIR, DET_MODEL_NAME)

if not os.path.exists(REPO_PATH):
    print("  Clonage du repo sayedgamal99...")
    os.system("git clone https://github.com/sayedgamal99/Real-Time-Smoke-Fire-Detection-YOLO11.git /content/smoke-fire-yolo")

if not os.path.exists(DET_PRETRAINED):
    shutil.copy(REPO_MODEL, DET_PRETRAINED)
    print(f"  Modèle copié : {DET_PRETRAINED}")

if not os.path.exists(DET_PRETRAINED) or os.path.getsize(DET_PRETRAINED) < 1000:
    raise FileNotFoundError(f"Modèle détection introuvable : {DET_PRETRAINED}")

CLS_BASE_MODEL = "yolo11n-cls.pt"
print(f"  Classification : {CLS_BASE_MODEL}")
print(f"  Détection      : {DET_MODEL_NAME} ({os.path.getsize(DET_PRETRAINED)/1e6:.1f} MB)")


# ============================================================
# PARTIE A — CLASSIFICATION FINE-TUNING
# ============================================================
print("\n[2/6] Fine-tuning YOLOv11 Classification (100 epochs)...")

yolo_cls = YOLO(CLS_BASE_MODEL)
yolo_cls.train(
    task     = "classify",
    data     = SPLIT_DIR,    # ✅ dossier pas yaml
    epochs   = 1,
    imgsz    = 224,
    batch    = 32,
    lr0      = 0.001,
    name     = "yolo11_classify",
    project  = os.path.join(OUTPUT_DIR, "cls_runs"),
    patience = 7,
    exist_ok = True,
    device   = DEVICE,
    save     = True,
    degrees  = 15,
    fliplr   = 0.5,
    hsv_v    = 0.3,
    hsv_s    = 0.3,
)

best_cls_list = glob.glob(os.path.join(OUTPUT_DIR, "cls_runs/**/best.pt"), recursive=True)
if not best_cls_list:
    raise FileNotFoundError("Classification best.pt introuvable!")
best_cls_path = best_cls_list[0]
print(f"  best.pt classification : {best_cls_path}")


# ============================================================
# PARTIE B — EVALUATION CLASSIFICATION
# ============================================================
print("\n[3/6] Évaluation classification YOLOv11...")

yolo_cls_best = YOLO(best_cls_path)

cls_metrics = yolo_cls_best.val(
    task   = "classify",
    data   = SPLIT_DIR,    # ✅ dossier pas yaml
    split  = "test",       # ✅ test pas val
    imgsz  = 224,
    device = DEVICE,
)
cls_top1 = float(cls_metrics.top1) * 100
cls_top5 = float(cls_metrics.top5) * 100
print(f"  Top-1 Accuracy : {cls_top1:.2f}%")
print(f"  Top-5 Accuracy : {cls_top5:.2f}%")

# Test set pour Precision/Recall/F1
random.seed(42)
all_samples    = []
class_dirs     = sorted([d for d in os.listdir(CLASS_TRAIN)
                         if os.path.isdir(os.path.join(CLASS_TRAIN, d))])
cls_names_list = class_dirs
print(f"  Classes : {cls_names_list}")

print("  Utilisation du dossier test/ existant")
for label_idx, cls_name in enumerate(cls_names_list):
    test_dir = os.path.join(CLASS_TEST, cls_name)
    if os.path.exists(test_dir):
        imgs = [os.path.join(test_dir, f) for f in os.listdir(test_dir)
                if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        for img_path in imgs:
            all_samples.append((img_path, label_idx))

print(f"  Test set : {len(all_samples)} images")

preds_cls, labels_cls = [], []
for img_path, true_lbl in all_samples:
    res      = yolo_cls_best(img_path, verbose=False)[0]
    pred_lbl = int(res.probs.top1)
    preds_cls.append(pred_lbl)
    labels_cls.append(true_lbl)

print("\n" + "="*50)
print("RAPPORT CLASSIFICATION YOLOv11")
print("="*50)
cls_report = classification_report(labels_cls, preds_cls, target_names=cls_names_list, digits=4)
print(cls_report)
cls_acc = sum(p == l for p, l in zip(preds_cls, labels_cls)) / len(labels_cls) * 100
print(f"Accuracy globale : {cls_acc:.2f}%")

with open(os.path.join(OUTPUT_DIR, "yolo11_cls_report.txt"), "w") as f:
    f.write("RAPPORT CLASSIFICATION YOLOv11\n" + "="*50 + "\n")
    f.write(cls_report)
    f.write(f"\nAccuracy: {cls_acc:.2f}%\n")

# Confusion Matrix Classification
cm = confusion_matrix(labels_cls, preds_cls)
plt.figure(figsize=(7, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=cls_names_list, yticklabels=cls_names_list,
            annot_kws={"size": 14})
plt.title("YOLOv11 Classification — Matrice de Confusion", fontsize=14, fontweight='bold')
plt.xlabel("Prédiction", fontsize=12)
plt.ylabel("Réalité", fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "yolo11_cls_confusion_matrix.png"), dpi=200, bbox_inches='tight')
plt.show()
print("  Confusion matrix sauvegardée!")

# Learning Curves
results_csv_list = glob.glob(os.path.join(OUTPUT_DIR, "cls_runs/**/results.csv"), recursive=True)
if results_csv_list:
    import csv
    epochs_list, train_loss_list, val_acc_list = [], [], []
    with open(results_csv_list[0], newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            row = {k.strip(): v.strip() for k, v in row.items()}
            try:
                epochs_list.append(int(float(row.get('epoch', 0))))
                train_loss_list.append(float(row.get('train/loss', row.get('loss', 0))))
                val_acc_list.append(float(row.get('metrics/accuracy_top1',
                                    row.get('val/acc_top1', 0))) * 100)
            except (ValueError, KeyError):
                continue
    if epochs_list:
        fig_lc, axes_lc = plt.subplots(1, 2, figsize=(13, 5))
        fig_lc.suptitle("YOLOv11 Classification — Courbes d'Apprentissage", fontsize=14, fontweight='bold')
        axes_lc[0].plot(epochs_list, val_acc_list,    'r-o', markersize=4, label="Val Accuracy")
        axes_lc[0].set_title("Accuracy"); axes_lc[0].set_xlabel("Epoch")
        axes_lc[0].set_ylabel("Accuracy (%)"); axes_lc[0].legend(); axes_lc[0].grid(True, alpha=0.4)
        axes_lc[1].plot(epochs_list, train_loss_list, 'b-o', markersize=4, label="Train Loss")
        axes_lc[1].set_title("Loss"); axes_lc[1].set_xlabel("Epoch")
        axes_lc[1].set_ylabel("Loss"); axes_lc[1].legend(); axes_lc[1].grid(True, alpha=0.4)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "yolo11_cls_learning_curves.png"), dpi=200, bbox_inches='tight')
        plt.show()
        print("  Learning curves sauvegardées!")


# ============================================================
# PARTIE C — DETECTION FINE-TUNING
# ============================================================
print("\n[4/6] Fine-tuning YOLOv11 Détection (100 epochs)...")

yolo_det = YOLO(DET_PRETRAINED)
yolo_det.train(
    task          = "detect",
    data          = DETECT_YAML,
    epochs        = 1,
    imgsz         = 640,
    batch         = 16,
    lr0           = 0.001,
    freeze        = 10,
    name          = "yolo11_detect",
    project       = os.path.join(OUTPUT_DIR, "det_runs"),
    patience      = 10,
    exist_ok      = True,
    device        = DEVICE,
    save          = True,
    warmup_epochs = 3,
)

best_det_list = glob.glob(os.path.join(OUTPUT_DIR, "det_runs/**/best.pt"), recursive=True)
if not best_det_list:
    raise FileNotFoundError("Detection best.pt introuvable!")
best_det_path = best_det_list[0]
print(f"  best.pt détection : {best_det_path}")


# ============================================================
# PARTIE D — EVALUATION DETECTION
# ============================================================
print("\n[5/6] Évaluation détection YOLOv11...")

yolo_det_best = YOLO(best_det_path)
det_metrics   = yolo_det_best.val(data=DETECT_YAML, split="test", device=DEVICE)

yolo_map50   = float(det_metrics.box.map50) * 100
yolo_map5095 = float(det_metrics.box.map)   * 100
yolo_prec    = float(det_metrics.box.mp)    * 100
yolo_rec     = float(det_metrics.box.mr)    * 100

print(f"mAP@0.5      : {yolo_map50:.2f}%")
print(f"mAP@0.5:0.95 : {yolo_map5095:.2f}%")
print(f"Precision    : {yolo_prec:.2f}%")
print(f"Recall       : {yolo_rec:.2f}%")

with open(os.path.join(OUTPUT_DIR, "yolo11_det_report.txt"), "w") as f:
    f.write("RAPPORT DÉTECTION YOLOv11\n" + "="*50 + "\n")
    f.write(f"mAP@0.5      : {yolo_map50:.2f}%\n")
    f.write(f"mAP@0.5:0.95 : {yolo_map5095:.2f}%\n")
    f.write(f"Precision    : {yolo_prec:.2f}%\n")
    f.write(f"Recall       : {yolo_rec:.2f}%\n")

# Confusion Matrix Detection
confusion_det = np.zeros((2, 4), dtype=int)
conf_all      = []
counts        = defaultdict(int)
no_detection  = 0

img_files = sorted(
    glob.glob(os.path.join(DETECT_TEST_IMG, "*.jpg")) +
    glob.glob(os.path.join(DETECT_TEST_IMG, "*.png"))
)

for img_path in img_files:
    lbl_path = os.path.join(
        DETECT_TEST_LBL,
        os.path.splitext(os.path.basename(img_path))[0] + ".txt"
    )
    has_annotation = os.path.exists(lbl_path) and os.path.getsize(lbl_path) > 0
    gt_row = 0 if has_annotation else 1
    result = yolo_det_best(img_path, verbose=False)[0]
    if result.boxes is None or len(result.boxes) == 0:
        confusion_det[gt_row][3] += 1
        no_detection += 1
    else:
        for box in result.boxes:
            conf = float(box.conf[0])
            conf_all.append(conf)
            idx, _, _ = conf_to_intensity(conf)
            confusion_det[gt_row][idx] += 1
            counts[idx] += 1

total_boxes = sum(counts.values())
print(f"Total boxes  : {total_boxes}")
print(f"Strong (>70%): {counts[0]}")
print(f"Medium 30-70%: {counts[1]}")
print(f"Weak   (<30%): {counts[2]}")

col_labels = ["Strong Fire\n(conf>70%)", "Medium Fire\n(30-70%)", "Weak Fire\n(conf<30%)", "No\nDetection"]
row_labels = ["Fire\n(annote)", "Background\n(non annote)"]
row_sums   = confusion_det.sum(axis=1, keepdims=True).astype(float)
row_sums[row_sums == 0] = 1
confusion_pct = confusion_det / row_sums * 100
annot_det     = np.empty_like(confusion_det, dtype=object)
for i in range(2):
    for j in range(4):
        annot_det[i, j] = f"{confusion_det[i,j]}\n({confusion_pct[i,j]:.1f}%)"

fig_cm2, ax_cm2 = plt.subplots(figsize=(11, 5))
sns.heatmap(confusion_pct, annot=annot_det, fmt="", cmap="YlOrRd",
            linewidths=0.8, linecolor="white", ax=ax_cm2,
            xticklabels=col_labels, yticklabels=row_labels,
            vmin=0, vmax=100, cbar_kws={"label": "% des images", "shrink": 0.8})
ax_cm2.set_title("Matrice de Confusion - Détection YOLOv11\nApproche YOLO11 Only",
                 fontsize=13, fontweight="bold", pad=14)
ax_cm2.set_xlabel("Niveau de Confiance Prédit", fontsize=12, labelpad=10)
ax_cm2.set_ylabel("Ground Truth", fontsize=12, labelpad=10)
legend_det = [
    mpatches.Patch(color="#c0392b", label="Strong Fire  - conf > 70%"),
    mpatches.Patch(color="#e67e22", label="Medium Fire - conf 30-70%"),
    mpatches.Patch(color="#f1c40f", label="Weak Fire    - conf < 30%"),
    mpatches.Patch(color="#95a5a6", label="No Detection"),
]
ax_cm2.legend(handles=legend_det, loc="upper right",
              bbox_to_anchor=(1.0, -0.22), ncol=2, fontsize=9, frameon=False)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "yolo11_det_confusion_matrix.png"), dpi=200, bbox_inches="tight")
plt.show()

# Distribution confidence
if conf_all:
    fig_dist, ax_dist = plt.subplots(figsize=(10, 5))
    n_hist, bins_hist, hist_patches = ax_dist.hist(conf_all, bins=30, edgecolor="white", linewidth=0.5)
    for patch, left in zip(hist_patches, bins_hist[:-1]):
        _, _, color = conf_to_intensity(left)
        patch.set_facecolor(color)
    ax_dist.axvline(STRONG_THRESH, color="#7B0000", linestyle="--", linewidth=1.8, label=f"Seuil Strong = {STRONG_THRESH}")
    ax_dist.axvline(WEAK_THRESH,   color="#5D3800", linestyle="--", linewidth=1.8, label=f"Seuil Weak   = {WEAK_THRESH}")
    y_max = n_hist.max()
    ax_dist.text(0.15, y_max * 0.85, "Weak Fire",   color="#5D3800", fontsize=11, fontweight="bold")
    ax_dist.text(0.43, y_max * 0.85, "Medium Fire", color="#7A4200", fontsize=11, fontweight="bold")
    ax_dist.text(0.73, y_max * 0.85, "Strong Fire", color="#7B0000", fontsize=11, fontweight="bold")
    ax_dist.set_xlabel("Confidence Score", fontsize=12)
    ax_dist.set_ylabel("Nombre de détections", fontsize=12)
    ax_dist.set_title("Distribution des Confidence Scores\nApproche YOLO11 Only",
                      fontsize=13, fontweight="bold")
    ax_dist.legend(fontsize=10)
    ax_dist.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "yolo11_confidence_distribution.png"), dpi=200, bbox_inches="tight")
    plt.show()
    avg_conf = np.mean(conf_all) * 100
else:
    avg_conf = 0.0


# ============================================================
# PARTIE E — PIPELINE VISUALISATION
# ============================================================
print("\n[6/6] Visualisation pipeline Classification → Détection...")

fire_class_idx = cls_names_list.index('fire') if 'fire' in cls_names_list else 0
fire_samples   = [(p, l) for p, l in all_samples if l == fire_class_idx][:8]

fig, axes = plt.subplots(2, 4, figsize=(18, 9))
fig.suptitle("Pipeline YOLOv11 : Classification → Détection (YOLO11 Only)",
             fontsize=14, fontweight='bold')

for i, (img_path, _) in enumerate(fire_samples):
    ax      = axes[i // 4][i % 4]
    pil_img = Image.open(img_path).convert("RGB")
    cls_res    = yolo_cls_best(img_path, verbose=False)[0]
    pred_class = int(cls_res.probs.top1)
    conf_cls   = float(cls_res.probs.top1conf)
    ax.imshow(pil_img)

    if pred_class == fire_class_idx:
        det_res = yolo_det_best(img_path, verbose=False)[0]
        if det_res.boxes is not None and len(det_res.boxes) > 0:
            n_strong, n_medium, n_weak = 0, 0, 0
            for box in det_res.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf_det = float(box.conf[0])
                _, intensity_label, color = conf_to_intensity(conf_det)
                if conf_det >= STRONG_THRESH:   n_strong += 1
                elif conf_det >= WEAK_THRESH:   n_medium += 1
                else:                           n_weak   += 1
                w_s = pil_img.width  / det_res.orig_shape[1]
                h_s = pil_img.height / det_res.orig_shape[0]
                rect = patches.Rectangle(
                    (x1 * w_s, y1 * h_s),
                    (x2 - x1) * w_s, (y2 - y1) * h_s,
                    linewidth=2.5, edgecolor=color, facecolor='none')
                ax.add_patch(rect)
                ax.text(x1 * w_s, y1 * h_s - 5,
                        f"{intensity_label} {conf_det:.2f}", color='white',
                        fontsize=7, fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.2', facecolor=color, alpha=0.85))
            parts = []
            if n_strong: parts.append(f"S:{n_strong}")
            if n_medium: parts.append(f"M:{n_medium}")
            if n_weak:   parts.append(f"W:{n_weak}")
            ax.set_title(f"Cls:{conf_cls*100:.0f}% | " + " | ".join(parts),
                         fontsize=7, color='darkred', fontweight='bold')
        else:
            ax.set_title(f"Fire {conf_cls*100:.0f}% | 0 bbox", fontsize=8, color='orange')
    else:
        ax.set_title("No Fire", fontsize=8, color='gray')
    ax.axis('off')

legend_vis = [
    mpatches.Patch(color="#c0392b", label="Strong Fire (>70%)"),
    mpatches.Patch(color="#e67e22", label="Medium Fire (30-70%)"),
    mpatches.Patch(color="#f1c40f", label="Weak Fire (<30%)"),
]
fig.legend(handles=legend_vis, loc='lower center', ncol=3,
           fontsize=10, bbox_to_anchor=(0.5, -0.02), frameon=False)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "pipeline_yolo11_visualizations.png"), dpi=200, bbox_inches='tight')
plt.show()
print("  Visualisations sauvegardées!")

# Benchmark Timing
t_cls_list, t_det_list, t_tot_list = [], [], []
for img_path, _ in fire_samples[:min(50, len(fire_samples))]:
    t0 = time.time()
    cls_res    = yolo_cls_best(img_path, verbose=False)[0]
    pred_class = int(cls_res.probs.top1)
    t_cls = time.time() - t0
    t1 = time.time()
    if pred_class == fire_class_idx:
        _ = yolo_det_best(img_path, verbose=False)
    t_det = time.time() - t1
    t_cls_list.append(t_cls * 1000)
    t_det_list.append(t_det * 1000)
    t_tot_list.append((t_cls + t_det) * 1000)

avg_cls = np.mean(t_cls_list)
avg_det = np.mean(t_det_list)
avg_tot = np.mean(t_tot_list)
print(f"Temps Classification : {avg_cls:.1f} ms")
print(f"Temps Détection      : {avg_det:.1f} ms")
print(f"Temps total          : {avg_tot:.1f} ms  ({1000/avg_tot:.1f} FPS)")

# JSON Final
strong_pct = counts[0] / total_boxes * 100 if total_boxes > 0 else 0
medium_pct = counts[1] / total_boxes * 100 if total_boxes > 0 else 0
weak_pct   = counts[2] / total_boxes * 100 if total_boxes > 0 else 0

results_summary = {
    "Approche"         : "YOLO11 Only — Classification + Détection",
    "Pretrained_model" : DET_MODEL_NAME,
    "Source"           : "https://github.com/sayedgamal99/Real-Time-Smoke-Fire-Detection-YOLO11",
    "Epochs"           : 100,
    "Classification": {
        "Modele"       : "YOLOv11n-cls (yolo11n-cls.pt)",
        "Top1_Accuracy": round(cls_top1, 2),
        "Top5_Accuracy": round(cls_top5, 2),
        "Test_Accuracy": round(cls_acc,  2),
    },
    "Detection": {
        "Modele"        : f"YOLOv11 Detect ({DET_MODEL_NAME})",
        "mAP_50"        : round(yolo_map50,   2),
        "mAP_50_95"     : round(yolo_map5095, 2),
        "Precision"     : round(yolo_prec,    2),
        "Recall"        : round(yolo_rec,     2),
        "Avg_Confidence": round(avg_conf,      2),
    },
    "Fire_Intensity": {
        "Strong_Fire_pct": round(strong_pct, 1),
        "Medium_Fire_pct": round(medium_pct, 1),
        "Weak_Fire_pct"  : round(weak_pct,   1),
        "No_Detection"   : no_detection,
        "Total_Boxes"    : total_boxes,
    },
    "Pipeline_Timing_ms": {
        "Classify_avg" : round(avg_cls, 2),
        "Detect_avg"   : round(avg_det, 2),
        "Total_avg"    : round(avg_tot, 2),
        "FPS_approx"   : round(1000 / avg_tot, 1),
    }
}

json_path = os.path.join(OUTPUT_DIR, "yolo11_results.json")
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(results_summary, f, indent=2, ensure_ascii=False)

print("\n" + "=" * 60)
print("  APPROCHE YOLO11 ONLY TERMINÉE")
print("=" * 60)
print(json.dumps(results_summary, indent=2, ensure_ascii=False))
print(f"\nRésultats : {OUTPUT_DIR}")
