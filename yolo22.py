# ============================================================
#  APPROCHE 2 : YOLOv11s (Classification) -> YOLOv11 (Detection)
#  Classification : fine-tune 100 epochs sur Shamta & Demir 2024
#  Detection      : fire_detector.pt PRE-ENTRAINE (sans re-entrainement)
#  Split          : 70% Train / 15% Val / 15% Test
# ============================================================

import os, glob, time, json, random, csv, shutil
from ultralytics import YOLO
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.patches as mpatches
import seaborn as sns
import numpy as np
from PIL import Image
from collections import defaultdict
import torch

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
BASE_OUT = find_real_path("MEMOIRE/YOLO2_Results_final_epochs100")

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
# PATHS
# ---------------------------------------------
CLASS_TRAIN     = f"{SPLIT_DIR}/train"
CLASS_VAL       = f"{SPLIT_DIR}/valid"
CLASS_TEST      = f"{SPLIT_DIR}/test"
DETECT_YAML     = f"{BASE_DET}/data.yaml"
DETECT_TEST_IMG = f"{BASE_DET}/test/images"
DETECT_TEST_LBL = f"{BASE_DET}/test/labels"
OUTPUT_DIR      = BASE_OUT
PRETRAINED_DET  = "/content/fire-detection-using-yolov11/models/fire_detector.pt"

os.makedirs(OUTPUT_DIR, exist_ok=True)

STRONG_THRESH = 0.50
WEAK_THRESH   = 0.30
DEVICE = 0 if torch.cuda.is_available() else 'cpu'

print("=" * 60)
print("  APPROCHE 2 : YOLOv11s Classify -> YOLOv11 Detect")
print("  Classification : fine-tune 20 epochs")
print("  Detection      : fire_detector.pt pre-entraine")
print("  Split          : 70% Train / 15% Val / 15% Test")
print("=" * 60)
print(f"Device           : {'GPU' if torch.cuda.is_available() else 'CPU'}")
print(f"fire_detector.pt : {os.path.exists(PRETRAINED_DET)}")
print(f"CLASS_TRAIN      : {CLASS_TRAIN}")
print(f"DETECT_YAML      : {DETECT_YAML}")
print(f"OUTPUT_DIR       : {OUTPUT_DIR}")
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


# ============================================================
# PARTIE A — CLASSIFICATION
# ============================================================

print("\n[1/6] Fine-tuning YOLO-Classify (20 epochs)...")
yolo_cls = YOLO("yolo11s-cls.pt")
yolo_cls.train(
    data          = SPLIT_DIR,   # ✅ dossier
    epochs        = 100,          # ✅ 100 epochs
    imgsz         = 224,
    batch         = 32,
    name          = "approche2_classify",
    project       = os.path.join(OUTPUT_DIR, "cls_runs"),
    patience      = 10,
    exist_ok      = True,
    device        = DEVICE,
    save          = True,
    lr0           = 0.001,
    warmup_epochs = 3,
)
print("Fine-tuning Classification termine!")

# Evaluation
print("\n[2/6] Evaluation YOLO-Classify...")
cls_best_list = glob.glob(os.path.join(OUTPUT_DIR, "cls_runs/**/best.pt"), recursive=True)
if not cls_best_list:
    raise FileNotFoundError("best.pt classify introuvable!")
cls_best_path = cls_best_list[0]
yolo_cls_best = YOLO(cls_best_path)

cls_metrics  = yolo_cls_best.val(
    data   = SPLIT_DIR,   # ✅ dossier
    split  = "test",
    imgsz  = 224,
    batch  = 32,
    device = DEVICE,
)
cls_accuracy = float(cls_metrics.top1) * 100
cls_top5     = float(cls_metrics.top5) * 100
print(f"Top-1 Accuracy : {cls_accuracy:.2f}%")
print(f"Top-5 Accuracy : {cls_top5:.2f}%")

# Test set pour Precision/Recall/F1
random.seed(42)
all_samples = []
class_dirs  = sorted([d for d in os.listdir(CLASS_TRAIN)
                      if os.path.isdir(os.path.join(CLASS_TRAIN, d))])
CLASS_NAMES = class_dirs
print(f"Classes: {CLASS_NAMES}")

print("Utilisation du dossier test/ existant")
for label_idx, cls_name in enumerate(CLASS_NAMES):
    test_dir = os.path.join(CLASS_TEST, cls_name)
    if os.path.exists(test_dir):
        imgs = [os.path.join(test_dir, f) for f in os.listdir(test_dir)
                if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        for img_path in imgs:
            all_samples.append((img_path, label_idx))

print(f"Test set: {len(all_samples)} images")

preds_cls, labels_cls = [], []
for img_path, true_label in all_samples:
    result     = yolo_cls_best(img_path, verbose=False, imgsz=224)[0]
    pred_label = int(result.probs.top1)
    preds_cls.append(pred_label)
    labels_cls.append(true_label)

report_cls     = classification_report(labels_cls, preds_cls, target_names=CLASS_NAMES, digits=4)
print(report_cls)
cls_acc_manual = sum(p == l for p, l in zip(preds_cls, labels_cls)) / len(labels_cls) * 100
print(f"Accuracy manuelle: {cls_acc_manual:.2f}%")

with open(os.path.join(OUTPUT_DIR, "yolo_cls_classification_report.txt"), "w") as f:
    f.write("RAPPORT YOLO-Classify - Approche 2\n" + "="*50 + "\n")
    f.write(f"Split: 70% Train / 15% Val / 15% Test\n\n")
    f.write(report_cls)
    f.write(f"\nAccuracy: {cls_acc_manual:.2f}%\n")

# Confusion Matrix
cm_cls = confusion_matrix(labels_cls, preds_cls)
plt.figure(figsize=(7, 6))
sns.heatmap(cm_cls, annot=True, fmt='d', cmap='Oranges',
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, annot_kws={"size": 14})
plt.title("YOLO-Classify - Matrice de Confusion (Approche 2)\nSplit 70/15/15",
          fontsize=14, fontweight='bold')
plt.xlabel("Prediction", fontsize=12)
plt.ylabel("Realite", fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "yolo_cls_confusion_matrix.png"), dpi=200, bbox_inches='tight')
plt.show()
print("Confusion matrix classify sauvegardee!")

# Learning Curves
results_csv_list = glob.glob(os.path.join(OUTPUT_DIR, "cls_runs/**/results.csv"), recursive=True)
if results_csv_list:
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
        fig_lc.suptitle("YOLO-Classify - Courbes d'Apprentissage (Approche 2)",
                         fontsize=14, fontweight='bold')
        axes_lc[0].plot(epochs_list, val_acc_list,    'r-o', markersize=4, label="Val Accuracy")
        axes_lc[0].set_title("Accuracy"); axes_lc[0].set_xlabel("Epoch")
        axes_lc[0].set_ylabel("Accuracy (%)"); axes_lc[0].legend(); axes_lc[0].grid(True, alpha=0.4)
        axes_lc[1].plot(epochs_list, train_loss_list, 'b-o', markersize=4, label="Train Loss")
        axes_lc[1].set_title("Loss"); axes_lc[1].set_xlabel("Epoch")
        axes_lc[1].set_ylabel("Loss"); axes_lc[1].legend(); axes_lc[1].grid(True, alpha=0.4)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "yolo_cls_learning_curves.png"), dpi=200, bbox_inches='tight')
        plt.show()
        print("Learning curves sauvegardees!")


# ============================================================
# PARTIE B — DETECTION (fire_detector.pt SANS re-entrainement)
# ============================================================

print("\n[3/6] Chargement modele detection PRE-ENTRAINE...")
yolo_det_best = YOLO(PRETRAINED_DET)   # ✅ chargement direct sans train
print("Modele detection charge!")

print("\n[4/6] Evaluation YOLO-Detect sur dataset Shamta & Demir...")
det_metrics   = yolo_det_best.val(data=DETECT_YAML, split="test")
det_map50     = float(det_metrics.box.map50) * 100
det_map5095   = float(det_metrics.box.map)   * 100
det_precision = float(det_metrics.box.mp)    * 100
det_recall    = float(det_metrics.box.mr)    * 100

print(f"mAP@0.5      : {det_map50:.2f}%")
print(f"mAP@0.5:0.95 : {det_map5095:.2f}%")
print(f"Precision    : {det_precision:.2f}%")
print(f"Recall       : {det_recall:.2f}%")

with open(os.path.join(OUTPUT_DIR, "yolo_det_report.txt"), "w") as f:
    f.write("RAPPORT YOLO-Detect - Approche 2\n" + "="*50 + "\n")
    f.write(f"Modele : fire_detector.pt (pre-entraine, sans re-entrainement)\n")
    f.write(f"mAP@0.5      : {det_map50:.2f}%\n")
    f.write(f"mAP@0.5:0.95 : {det_map5095:.2f}%\n")
    f.write(f"Precision    : {det_precision:.2f}%\n")
    f.write(f"Recall       : {det_recall:.2f}%\n")


# ============================================================
# PARTIE C — CONFUSION MATRIX + DISTRIBUTION
# ============================================================

print("\n[5/6] Matrice de confusion detection...")
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

col_labels    = ["Strong Fire\n(conf>70%)", "Medium Fire\n(30-70%)", "Weak Fire\n(conf<30%)", "No\nDetection"]
row_labels    = ["Fire\n(annote)", "Background\n(non annote)"]
row_sums      = confusion_det.sum(axis=1, keepdims=True).astype(float)
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
ax_cm2.set_title("Matrice de Confusion - Detection YOLO\nApproche 2 : YOLOv11s Classify + YOLOv11 Detect",
                 fontsize=13, fontweight="bold", pad=14)
ax_cm2.set_xlabel("Niveau de Confiance Predit", fontsize=12, labelpad=10)
ax_cm2.set_ylabel("Ground Truth", fontsize=12, labelpad=10)
ax_cm2.tick_params(axis="x", labelsize=10)
ax_cm2.tick_params(axis="y", labelsize=10, rotation=0)
legend_det = [
    mpatches.Patch(color="#c0392b", label="Strong Fire  - conf > 70%"),
    mpatches.Patch(color="#e67e22", label="Medium Fire - conf 30-70%"),
    mpatches.Patch(color="#f1c40f", label="Weak Fire    - conf < 30%"),
    mpatches.Patch(color="#95a5a6", label="No Detection"),
]
ax_cm2.legend(handles=legend_det, loc="upper right",
              bbox_to_anchor=(1.0, -0.22), ncol=2, fontsize=9, frameon=False)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "yolo_detection_confusion_matrix.png"), dpi=200, bbox_inches="tight")
plt.show()

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
    ax_dist.set_ylabel("Nombre de detections", fontsize=12)
    ax_dist.set_title("Distribution des Confidence Scores\nApproche 2",
                      fontsize=13, fontweight="bold")
    ax_dist.legend(fontsize=10)
    ax_dist.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "yolo_confidence_distribution.png"), dpi=200, bbox_inches="tight")
    plt.show()
    avg_conf = np.mean(conf_all) * 100
else:
    avg_conf = 0.0


# ============================================================
# PARTIE D — PIPELINE VISUALISATION
# ============================================================

print("\n[6/6] Visualisation pipeline...")
fire_class_idx    = CLASS_NAMES.index('fire') if 'fire' in CLASS_NAMES else 0
fire_test_samples = [(p, l) for p, l in all_samples if l == fire_class_idx][:8]

fig_vis, axes_vis = plt.subplots(2, 4, figsize=(18, 9))
fig_vis.suptitle("Pipeline YOLO-Classify -> YOLO-Detect Pre-entraine (Approche 2)",
                 fontsize=13, fontweight='bold')

for i, (img_path, _) in enumerate(fire_test_samples):
    ax      = axes_vis[i // 4][i % 4]
    pil_img = Image.open(img_path).convert("RGB")
    cls_result  = yolo_cls_best(img_path, verbose=False, imgsz=224)[0]
    pred_class  = int(cls_result.probs.top1)
    cls_conf    = float(cls_result.probs.top1conf)
    ax.imshow(pil_img)
    if pred_class == fire_class_idx:
        det_result = yolo_det_best(img_path, verbose=False)[0]
        if det_result.boxes is not None and len(det_result.boxes) > 0:
            n_strong, n_medium, n_weak = 0, 0, 0
            for box in det_result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0])
                _, intensity_label, color = conf_to_intensity(conf)
                if conf >= STRONG_THRESH:   n_strong += 1
                elif conf >= WEAK_THRESH:   n_medium += 1
                else:                       n_weak   += 1
                w_scale = pil_img.width  / det_result.orig_shape[1]
                h_scale = pil_img.height / det_result.orig_shape[0]
                rect = patches.Rectangle(
                    (x1 * w_scale, y1 * h_scale),
                    (x2 - x1) * w_scale, (y2 - y1) * h_scale,
                    linewidth=2.5, edgecolor=color, facecolor='none')
                ax.add_patch(rect)
                ax.text(x1 * w_scale, y1 * h_scale - 5,
                        f"{intensity_label} {conf:.2f}", color='white',
                        fontsize=7, fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.2', facecolor=color, alpha=0.85))
            parts = []
            if n_strong: parts.append(f"S:{n_strong}")
            if n_medium: parts.append(f"M:{n_medium}")
            if n_weak:   parts.append(f"W:{n_weak}")
            ax.set_title(f"Cls:{cls_conf*100:.0f}% | " + " | ".join(parts),
                         fontsize=7, color='darkred', fontweight='bold')
        else:
            ax.set_title(f"Fire {cls_conf*100:.0f}% | 0 bbox", fontsize=8, color='orange')
    else:
        ax.set_title("No Fire", fontsize=8, color='gray')
    ax.axis('off')

legend_vis = [
    mpatches.Patch(color="#c0392b", label="Strong Fire (>70%)"),
    mpatches.Patch(color="#e67e22", label="Medium Fire (30-70%)"),
    mpatches.Patch(color="#f1c40f", label="Weak Fire (<30%)"),
]
fig_vis.legend(handles=legend_vis, loc='lower center', ncol=3,
               fontsize=10, bbox_to_anchor=(0.5, -0.02), frameon=False)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "pipeline_yolo_cls_det_visualizations.png"), dpi=200, bbox_inches='tight')
plt.show()

# Benchmark
times_cls, times_det, times_total = [], [], []
for img_path, _ in fire_test_samples[:min(50, len(fire_test_samples))]:
    t0    = time.time()
    cls_r = yolo_cls_best(img_path, verbose=False, imgsz=224)[0]
    pred  = int(cls_r.probs.top1)
    t_cls = time.time() - t0
    t1    = time.time()
    if pred == fire_class_idx:
        _ = yolo_det_best(img_path, verbose=False)
    t_det = time.time() - t1
    times_cls.append(t_cls * 1000)
    times_det.append(t_det * 1000)
    times_total.append((t_cls + t_det) * 1000)

avg_cls   = np.mean(times_cls)
avg_det   = np.mean(times_det)
avg_total = np.mean(times_total)
print(f"Temps YOLO-Cls  : {avg_cls:.1f} ms")
print(f"Temps YOLO-Det  : {avg_det:.1f} ms")
print(f"Temps total     : {avg_total:.1f} ms  ({1000/avg_total:.1f} FPS)")

# JSON
strong_pct = counts[0] / total_boxes * 100 if total_boxes > 0 else 0
medium_pct = counts[1] / total_boxes * 100 if total_boxes > 0 else 0
weak_pct   = counts[2] / total_boxes * 100 if total_boxes > 0 else 0

results_summary = {
    "Approche" : "Approche 2 - YOLOv11s Classify + YOLOv11 Detect (Pre-entraine)",
    "Split"    : "70% Train / 15% Val / 15% Test",
    "Epochs_classification" : 100,
    "Classification": {
        "Modele"        : "YOLOv11s-cls (fine-tune Shamta & Demir 2024)",
        "Accuracy"      : round(cls_acc_manual, 2),
        "Top1_YOLO_val" : round(cls_accuracy,   2),
    },
    "Detection": {
        "Modele"        : "fire_detector.pt (pre-entraine, sans re-entrainement)",
        "mAP_50"        : round(det_map50,      2),
        "mAP_50_95"     : round(det_map5095,    2),
        "Precision"     : round(det_precision,  2),
        "Recall"        : round(det_recall,     2),
        "Avg_Confidence": round(avg_conf,        2),
    },
    "Fire_Intensity": {
        "Strong_Fire_pct": round(strong_pct, 1),
        "Medium_Fire_pct": round(medium_pct, 1),
        "Weak_Fire_pct"  : round(weak_pct,   1),
        "No_Detection"   : no_detection,
        "Total_Boxes"    : total_boxes,
    },
    "Pipeline_Timing_ms": {
        "YOLO_Cls_avg"  : round(avg_cls,   2),
        "YOLO_Det_avg"  : round(avg_det,   2),
        "Total_avg"     : round(avg_total, 2),
        "FPS_approx"    : round(1000 / avg_total, 1),
    }
}

json_path = os.path.join(OUTPUT_DIR, "approche2_results.json")
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(results_summary, f, indent=2, ensure_ascii=False)

print("\n" + "=" * 60)
print("  APPROCHE 2 TERMINEE")
print("=" * 60)
print(json.dumps(results_summary, indent=2, ensure_ascii=False))
print(f"\nResultats : {OUTPUT_DIR}")
