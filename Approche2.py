# ============================================================
#  APPROCHE 2 : YOLOv11s (Classification) -> YOLOv11s (Detection)
#  Pipeline: Image -> YOLO Classify (fire/no fire?) -> If fire -> YOLO Detect (where?)
#  Dataset: Shamta & Demir 2024
# ============================================================
#
#  RESULTATS POUR LA MEMOIRE:
#  YOLO-Classify Accuracy, Precision, Recall, F1
#  Confusion Matrix YOLO-Classify (fire / nofire)
#  Learning Curves YOLO-Classify
#  YOLO-Detect mAP@0.5, Precision, Recall
#  Confusion Matrix YOLO Detection (Strong / Medium / Weak Fire)
#       Strong Fire : confidence > 70%
#       Medium Fire : confidence 30% - 70%
#       Weak Fire   : confidence < 30%
#  Distribution des confidence scores YOLO-Detect
#  Bounding Box visualizations avec couleur par intensite
#  Pipeline complet avec temps de traitement
#  JSON pour comparaison avec Approche 1
# ============================================================

import os, glob, time, json
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
# 1. PATHS
# ---------------------------------------------
CLASS_TRAIN  = "/content/drive/MyDrive/MEMOIRE/ForestFireDataset(Classifications)/ForestFireDataset/train"
CLASS_YAML   = "/content/drive/MyDrive/MEMOIRE/ForestFireDataset(Classifications)/ForestFireDataset/data.yaml"
DETECT_YAML  = "/content/drive/MyDrive/MEMOIRE/ForesFireDataset(ObjectDetection)/data.yaml"
OUTPUT_DIR   = "/content/drive/MyDrive/MEMOIRE/Approche2_Results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Seuils confidence intensite
STRONG_THRESH = 0.70
WEAK_THRESH   = 0.30

print("=" * 60)
print("  APPROCHE 2 : YOLOv11s Classify -> YOLOv11s Detect")
print("=" * 60)
print(f"Classification YAML : {CLASS_YAML}")
print(f"Detection YAML      : {DETECT_YAML}")
print(f"Resultats           : {OUTPUT_DIR}")
DEVICE = 0 if torch.cuda.is_available() else 'cpu'
print(f"Device              : {'cuda' if torch.cuda.is_available() else 'cpu'}")


# ---------------------------------------------
# HELPER: confidence -> intensite
# ---------------------------------------------
def conf_to_intensity(conf):
    """Retourne (index, label, couleur) selon confidence score."""
    if conf >= STRONG_THRESH:
        return 0, "Strong Fire", "#c0392b"
    elif conf >= WEAK_THRESH:
        return 1, "Medium Fire", "#e67e22"
    else:
        return 2, "Weak Fire",   "#f1c40f"


# ============================================================
# PARTIE A — YOLO CLASSIFICATION
# ============================================================

# ---------------------------------------------
# 2. YOLO CLASSIFY TRAINING
# ---------------------------------------------
print("\n[1/6] Entrainement YOLO-Classify (yolo11s-cls)...")

# NOTE: si data.yaml n existe pas pour la classification,
# on peut pointer directement vers le dossier train
yolo_cls = YOLO("yolo11s-cls.pt")
yolo_cls.train(
    data     = CLASS_TRAIN,   # dossier avec sous-dossiers fire/ et nofire/
    epochs   = 1,
    imgsz    = 224,
    batch    = 32,
    name     = "approche2_classify",
    project  = os.path.join(OUTPUT_DIR, "cls_runs"),
    patience = 10,
    exist_ok = True,
    device   = DEVICE,
    save     = True,
)
print("Entrainement YOLO-Classify termine!")


# ---------------------------------------------
# 3. YOLO CLASSIFY EVALUATION
# ---------------------------------------------
print("\n[2/6] Evaluation YOLO-Classify...")

# Recherche best.pt classify
cls_best_list = glob.glob(os.path.join(OUTPUT_DIR, "cls_runs/**/best.pt"), recursive=True)
if not cls_best_list:
    raise FileNotFoundError("best.pt classify introuvable!")
cls_best_path = cls_best_list[0]
print(f"best.pt classify: {cls_best_path}")

yolo_cls_best = YOLO(cls_best_path)

# Validation sur le test set
cls_metrics = yolo_cls_best.val(
    data    = CLASS_TRAIN,
    split   = "test",
    imgsz   = 224,
    batch   = 32,
    device  = DEVICE,
)

# Extraire les metriques
cls_accuracy  = float(cls_metrics.top1)   * 100
cls_top5      = float(cls_metrics.top5)   * 100

print(f"Top-1 Accuracy : {cls_accuracy:.2f}%")
print(f"Top-5 Accuracy : {cls_top5:.2f}%")

# --- Inference manuelle pour Precision/Recall/F1 et Confusion Matrix ---
print("\nCalcul Precision/Recall/F1 sur le test set...")

# Reconstruire test set manuellement (split 70/15/15 comme Approche 1)
import random
random.seed(42)
all_samples = []
class_dirs = sorted([d for d in os.listdir(CLASS_TRAIN)
                     if os.path.isdir(os.path.join(CLASS_TRAIN, d))])
CLASS_NAMES = class_dirs
print(f"Classes: {CLASS_NAMES}")

for label_idx, cls_name in enumerate(CLASS_NAMES):
    cls_dir = os.path.join(CLASS_TRAIN, cls_name)
    imgs = [os.path.join(cls_dir, f) for f in os.listdir(cls_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    random.shuffle(imgs)
    n_total = len(imgs)
    n_train = int(0.70 * n_total)
    n_valid = int(0.15 * n_total)
    test_imgs = imgs[n_train + n_valid:]
    for img_path in test_imgs:
        all_samples.append((img_path, label_idx))

print(f"Test set: {len(all_samples)} images")

preds_cls, labels_cls = [], []
for img_path, true_label in all_samples:
    result = yolo_cls_best(img_path, verbose=False, imgsz=224)[0]
    pred_label = int(result.probs.top1)
    preds_cls.append(pred_label)
    labels_cls.append(true_label)

# Rapport classification
print("\n" + "="*50)
print("RAPPORT YOLO-Classify (Classification)")
print("="*50)
report_cls = classification_report(labels_cls, preds_cls,
                                   target_names=CLASS_NAMES, digits=4)
print(report_cls)
cls_acc_manual = sum(p == l for p, l in zip(preds_cls, labels_cls)) / len(labels_cls) * 100
print(f"Accuracy manuelle: {cls_acc_manual:.2f}%")

with open(os.path.join(OUTPUT_DIR, "yolo_cls_classification_report.txt"), "w") as f:
    f.write("RAPPORT YOLO-Classify - Approche 2\n" + "="*50 + "\n")
    f.write(report_cls)
    f.write(f"\nAccuracy: {cls_acc_manual:.2f}%\n")

# --- Confusion Matrix Classification ---
cm_cls = confusion_matrix(labels_cls, preds_cls)
plt.figure(figsize=(7, 6))
sns.heatmap(cm_cls, annot=True, fmt='d', cmap='Oranges',
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
            annot_kws={"size": 14})
plt.title("YOLO-Classify - Matrice de Confusion (Approche 2)",
          fontsize=14, fontweight='bold')
plt.xlabel("Prediction", fontsize=12)
plt.ylabel("Realite", fontsize=12)
plt.tight_layout()
path_cm_cls = os.path.join(OUTPUT_DIR, "yolo_cls_confusion_matrix.png")
plt.savefig(path_cm_cls, dpi=200, bbox_inches='tight')
plt.show()
print(f"Confusion matrix classify sauvegardee: {path_cm_cls}")

# --- Learning Curves (depuis les resultats YOLO) ---
# YOLO sauvegarde results.csv dans le dossier du run
results_csv_list = glob.glob(
    os.path.join(OUTPUT_DIR, "cls_runs/**/results.csv"), recursive=True)

if results_csv_list:
    import csv
    epochs_list, train_loss_list, val_acc_list = [], [], []
    with open(results_csv_list[0], newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Nettoyer les noms de colonnes (espaces)
            row = {k.strip(): v.strip() for k, v in row.items()}
            try:
                epochs_list.append(int(float(row.get('epoch', 0))))
                # Colonnes YOLO classify
                train_loss_list.append(float(row.get('train/loss', row.get('loss', 0))))
                val_acc_list.append(float(row.get('metrics/accuracy_top1',
                                   row.get('val/acc_top1', 0))) * 100)
            except (ValueError, KeyError):
                continue

    if epochs_list:
        fig_lc, axes_lc = plt.subplots(1, 2, figsize=(13, 5))
        fig_lc.suptitle("YOLO-Classify - Courbes d'Apprentissage (Approche 2)",
                         fontsize=14, fontweight='bold')

        axes_lc[0].plot(epochs_list, val_acc_list, 'r-o', markersize=4, label="Val Accuracy")
        axes_lc[0].set_title("Accuracy"); axes_lc[0].set_xlabel("Epoch")
        axes_lc[0].set_ylabel("Accuracy (%)"); axes_lc[0].legend()
        axes_lc[0].grid(True, alpha=0.4)

        axes_lc[1].plot(epochs_list, train_loss_list, 'b-o', markersize=4, label="Train Loss")
        axes_lc[1].set_title("Loss"); axes_lc[1].set_xlabel("Epoch")
        axes_lc[1].set_ylabel("Loss"); axes_lc[1].legend()
        axes_lc[1].grid(True, alpha=0.4)

        plt.tight_layout()
        path_lc = os.path.join(OUTPUT_DIR, "yolo_cls_learning_curves.png")
        plt.savefig(path_lc, dpi=200, bbox_inches='tight')
        plt.show()
        print(f"Learning curves sauvegardees: {path_lc}")
    else:
        print("Donnees insuffisantes pour learning curves")
else:
    print("results.csv introuvable - learning curves non generees")


# ============================================================
# PARTIE B — YOLO DETECTION
# ============================================================

# ---------------------------------------------
# 4. YOLO DETECT TRAINING
# ---------------------------------------------
print("\n[3/6] Entrainement YOLO-Detect (yolo11s)...")

yolo_det = YOLO("yolo11s.pt")
yolo_det.train(
    data     = DETECT_YAML,
    epochs   = 50,
    imgsz    = 640,
    batch    = 16,
    name     = "approche2_detect",
    project  = os.path.join(OUTPUT_DIR, "det_runs"),
    patience = 10,
    exist_ok = True,
    device   = DEVICE,
    save     = True,
)
print("Entrainement YOLO-Detect termine!")


# ---------------------------------------------
# 5. YOLO DETECT EVALUATION
# ---------------------------------------------
print("\n[4/6] Evaluation YOLO-Detect...")

det_best_list = glob.glob(os.path.join(OUTPUT_DIR, "det_runs/**/best.pt"), recursive=True)
if not det_best_list:
    raise FileNotFoundError("best.pt detect introuvable!")
det_best_path = det_best_list[0]
print(f"best.pt detect: {det_best_path}")

yolo_det_best = YOLO(det_best_path)
det_metrics   = yolo_det_best.val(data=DETECT_YAML, split="test")

det_map50     = float(det_metrics.box.map50) * 100
det_map5095   = float(det_metrics.box.map)   * 100
det_precision = float(det_metrics.box.mp)    * 100
det_recall    = float(det_metrics.box.mr)    * 100

print("\n" + "="*50)
print("RAPPORT YOLO-Detect")
print("="*50)
print(f"mAP@0.5      : {det_map50:.2f}%")
print(f"mAP@0.5:0.95 : {det_map5095:.2f}%")
print(f"Precision    : {det_precision:.2f}%")
print(f"Recall       : {det_recall:.2f}%")


# ---------------------------------------------
# 6. CONFUSION MATRIX DETECTION (Strong/Medium/Weak)
# ---------------------------------------------
print("\n[5/6] Matrice de confusion detection (Strong/Medium/Weak)...")

DETECT_TEST_IMG = "/content/drive/MyDrive/MEMOIRE/ForesFireDataset(ObjectDetection)/test/images"
DETECT_TEST_LBL = "/content/drive/MyDrive/MEMOIRE/ForesFireDataset(ObjectDetection)/test/labels"

# Matrice 2x4: lignes=GT(Fire/Background), colonnes=Strong/Medium/Weak/NoDetect
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
print(f"Total bounding boxes : {total_boxes}")
print(f"Images sans detection: {no_detection}")
print(f"Strong Fire (>70%)   : {counts[0]}")
print(f"Medium Fire (30-70%) : {counts[1]}")
print(f"Weak Fire   (<30%)   : {counts[2]}")

# --- Figure A : Confusion Matrix Detection ---
col_labels = ["Strong Fire\n(conf>70%)", "Medium Fire\n(30-70%)",
              "Weak Fire\n(conf<30%)", "No\nDetection"]
row_labels  = ["Fire\n(annote)", "Background\n(non annote)"]

row_sums = confusion_det.sum(axis=1, keepdims=True).astype(float)
row_sums[row_sums == 0] = 1
confusion_pct = confusion_det / row_sums * 100

annot_det = np.empty_like(confusion_det, dtype=object)
for i in range(2):
    for j in range(4):
        annot_det[i, j] = f"{confusion_det[i,j]}\n({confusion_pct[i,j]:.1f}%)"

fig_cm2, ax_cm2 = plt.subplots(figsize=(11, 5))
sns.heatmap(
    confusion_pct,
    annot      = annot_det,
    fmt        = "",
    cmap       = "YlOrRd",
    linewidths = 0.8,
    linecolor  = "white",
    ax         = ax_cm2,
    xticklabels= col_labels,
    yticklabels= row_labels,
    vmin=0, vmax=100,
    cbar_kws   = {"label": "% des images", "shrink": 0.8}
)
ax_cm2.set_title(
    "Matrice de Confusion - Detection YOLO par Niveau de Confiance\n"
    "Approche 2 : YOLOv11s Classify + YOLOv11s Detect",
    fontsize=13, fontweight="bold", pad=14
)
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
path_cm_det = os.path.join(OUTPUT_DIR, "yolo_detection_confusion_matrix.png")
plt.savefig(path_cm_det, dpi=200, bbox_inches="tight")
plt.show()
print(f"Confusion matrix detection sauvegardee: {path_cm_det}")

# --- Figure B : Distribution confidence ---
if conf_all:
    fig_dist, ax_dist = plt.subplots(figsize=(10, 5))
    n_hist, bins_hist, hist_patches = ax_dist.hist(
        conf_all, bins=30, edgecolor="white", linewidth=0.5)
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
        "Distribution des Confidence Scores - YOLOv11s Detect\n"
        "Approche 2 : YOLOv11s Classify + YOLOv11s Detect",
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


# ============================================================
# PARTIE C — PIPELINE COMPLET + VISUALISATION
# ============================================================

# ---------------------------------------------
# 7. PIPELINE VISUALISATION (Bounding Boxes colores)
# ---------------------------------------------
print("\n[6/6] Visualisation pipeline YOLO-Classify -> YOLO-Detect...")

# Recuperer images fire du test set
fire_class_idx = CLASS_NAMES.index('fire') if 'fire' in CLASS_NAMES else 0
fire_test_samples = [(p, l) for p, l in all_samples if l == fire_class_idx][:8]

fig_vis, axes_vis = plt.subplots(2, 4, figsize=(18, 9))
fig_vis.suptitle(
    "Pipeline YOLO-Classify -> YOLO-Detect : Detection par Niveau d'Intensite (Approche 2)",
    fontsize=13, fontweight='bold'
)

for i, (img_path, _) in enumerate(fire_test_samples):
    ax = axes_vis[i // 4][i % 4]
    pil_img = Image.open(img_path).convert("RGB")

    # Etape 1: YOLO Classification
    t0 = time.time()
    cls_result = yolo_cls_best(img_path, verbose=False, imgsz=224)[0]
    pred_class  = int(cls_result.probs.top1)
    cls_conf    = float(cls_result.probs.top1conf)
    t_cls = (time.time() - t0) * 1000

    ax.imshow(pil_img)

    if pred_class == fire_class_idx:
        # Etape 2: YOLO Detection
        t1 = time.time()
        det_result = yolo_det_best(img_path, verbose=False)[0]
        t_det = (time.time() - t1) * 1000

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
                    linewidth=2.5, edgecolor=color, facecolor='none'
                )
                ax.add_patch(rect)
                ax.text(x1 * w_scale, y1 * h_scale - 5,
                        f"{intensity_label} {conf:.2f}",
                        color='white', fontsize=7, fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.2',
                                  facecolor=color, alpha=0.85))

            parts = []
            if n_strong: parts.append(f"S:{n_strong}")
            if n_medium: parts.append(f"M:{n_medium}")
            if n_weak:   parts.append(f"W:{n_weak}")
            ax.set_title(
                f"YOLO-Cls:{cls_conf*100:.0f}% | " + " | ".join(parts),
                fontsize=7, color='darkred', fontweight='bold'
            )
        else:
            ax.set_title(f"YOLO-Cls: Fire {cls_conf*100:.0f}%\nYOLO-Det: 0 bbox",
                         fontsize=8, color='orange')
    else:
        ax.set_title(f"YOLO-Cls: No Fire\nYOLO-Det ignore", fontsize=8, color='gray')

    ax.axis('off')

legend_vis = [
    mpatches.Patch(color="#c0392b", label="Strong Fire (conf>70%)"),
    mpatches.Patch(color="#e67e22", label="Medium Fire (30-70%)"),
    mpatches.Patch(color="#f1c40f", label="Weak Fire (<30%)"),
]
fig_vis.legend(handles=legend_vis, loc='lower center', ncol=3,
               fontsize=10, bbox_to_anchor=(0.5, -0.02), frameon=False)

plt.tight_layout()
path_vis = os.path.join(OUTPUT_DIR, "pipeline_yolo_cls_det_visualizations.png")
plt.savefig(path_vis, dpi=200, bbox_inches='tight')
plt.show()
print(f"Visualisations pipeline sauvegardees: {path_vis}")


# ---------------------------------------------
# 8. BENCHMARK TIMING
# ---------------------------------------------
print("\nMesure du temps de traitement du pipeline...")
times_cls, times_det, times_total = [], [], []

for img_path, _ in fire_test_samples[:min(50, len(fire_test_samples))]:
    t0 = time.time()
    cls_r = yolo_cls_best(img_path, verbose=False, imgsz=224)[0]
    pred  = int(cls_r.probs.top1)
    t_cls = time.time() - t0

    t1 = time.time()
    if pred == fire_class_idx:
        _ = yolo_det_best(img_path, verbose=False)
    t_det = time.time() - t1

    times_cls.append(t_cls * 1000)
    times_det.append(t_det * 1000)
    times_total.append((t_cls + t_det) * 1000)

avg_cls   = np.mean(times_cls)
avg_det   = np.mean(times_det)
avg_total = np.mean(times_total)
print(f"Temps moyen YOLO-Cls   : {avg_cls:.1f} ms")
print(f"Temps moyen YOLO-Det   : {avg_det:.1f} ms")
print(f"Temps total pipeline   : {avg_total:.1f} ms  ({1000/avg_total:.1f} FPS)")


# ---------------------------------------------
# 9. RÉSUMÉ FINAL JSON
# ---------------------------------------------
strong_pct = counts[0] / total_boxes * 100 if total_boxes > 0 else 0
medium_pct = counts[1] / total_boxes * 100 if total_boxes > 0 else 0
weak_pct   = counts[2] / total_boxes * 100 if total_boxes > 0 else 0

results_summary = {
    "Approche": "Approche 2 - YOLOv11s Classify + YOLOv11s Detect",
    "Classification": {
        "Modele"        : "YOLOv11s-cls",
        "Accuracy"      : round(cls_acc_manual, 2),
        "Top1_YOLO_val" : round(cls_accuracy, 2),
    },
    "Detection": {
        "Modele"        : "YOLOv11s",
        "mAP_50"        : round(det_map50, 2),
        "mAP_50_95"     : round(det_map5095, 2),
        "Precision"     : round(det_precision, 2),
        "Recall"        : round(det_recall, 2),
        "Avg_Confidence": round(avg_conf, 2),
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
print("  APPROCHE 2 TERMINEE - RESULTATS FINAUX")
print("=" * 60)
print(json.dumps(results_summary, indent=2, ensure_ascii=False))
print(f"\nTous les fichiers sauvegardes dans: {OUTPUT_DIR}")
print("\nFichiers generes:")
print("  yolo_cls_confusion_matrix.png          -> (Classification)")
print("  yolo_cls_learning_curves.png           -> (Classification)")
print("  yolo_detection_confusion_matrix.png    -> (Detection)")
print("  yolo_confidence_distribution.png       -> (Detection)")
print("  pipeline_yolo_cls_det_visualizations.png -> (Implementation)")
print("  approche2_results.json                 -> Comparaison avec Approche 1")
