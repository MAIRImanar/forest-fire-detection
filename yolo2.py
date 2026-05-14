import os, glob, torch, time, json, shutil
from ultralytics import YOLO
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import numpy as np
from tqdm import tqdm
from PIL import Image


# ─────────────────────────────────────────────
# 1. PATHS & CONFIG
# ─────────────────────────────────────────────

CLASS_TRAIN  = "/content/drive/MyDrive/MEMOIRE/ForestFireDataset(Classifications)/ForestFireDataset/train"
DETECT_YAML  = "/content/drive/MyDrive/MEMOIRE/ForesFireDataset(ObjectDetection)/data.yaml"
OUTPUT_DIR   = "/content/drive/MyDrive/MEMOIRE/Approche_YOLO11_OnlyYYYY"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 60)
print("  APPROCHE YOLO11 ONLY : Classification + Détection")
print("=" * 60)
print(f"Classification dataset : {CLASS_TRAIN}")
print(f"Detection YAML         : {DETECT_YAML}")
print(f"Résultats sauvegardés  : {OUTPUT_DIR}")

DEVICE = 0 if torch.cuda.is_available() else "cpu"
print(f"Device                 : {'CUDA' if torch.cuda.is_available() else 'CPU'}")


# ─────────────────────────────────────────────
# 2. CHARGEMENT DES MODÈLES
#    - Classification : yolo11n-cls.pt  (modèle officiel Ultralytics)
#    - Détection      : best_nano_111.pt (pré-entraîné feu/fumée)
# ─────────────────────────────────────────────

print("\n[1/6] Chargement des modèles...")

REPO_PATH        = "/content/smoke-fire-yolo"
DET_MODEL_NAME   = "best_nano_111.pt"
REPO_MODEL       = os.path.join(REPO_PATH, "models", DET_MODEL_NAME)
DET_PRETRAINED   = os.path.join(OUTPUT_DIR, DET_MODEL_NAME)

# Cloner le repo si pas déjà présent
if not os.path.exists(REPO_PATH):
    print("  Clonage du repo sayedgamal99...")
    os.system("git clone https://github.com/sayedgamal99/Real-Time-Smoke-Fire-Detection-YOLO11.git /content/smoke-fire-yolo")

# Copier best_nano_111.pt vers Drive
if not os.path.exists(DET_PRETRAINED):
    shutil.copy(REPO_MODEL, DET_PRETRAINED)
    print(f"  Modèle détection copié : {DET_PRETRAINED}")
else:
    print(f"  Modèle détection OK    : {DET_PRETRAINED}")

if not os.path.exists(DET_PRETRAINED) or os.path.getsize(DET_PRETRAINED) < 1000:
    raise FileNotFoundError(f"Modèle détection introuvable : {DET_PRETRAINED}")

# Modèle classification : yolo11n-cls.pt (Ultralytics officiel, téléchargé automatiquement)
CLS_BASE_MODEL = "yolo11n-cls.pt"

print(f"  Modèle classification  : {CLS_BASE_MODEL} (Ultralytics officiel)")
print(f"  Modèle détection       : {DET_MODEL_NAME} ({os.path.getsize(DET_PRETRAINED)/1e6:.1f} MB)")
print(f"  Source détection       : sayedgamal99/Real-Time-Smoke-Fire-Detection-YOLO11")


# ─────────────────────────────────────────────
# 3. YOLOv11 CLASSIFICATION FINE-TUNING
#    Using the pretrained weights as starting point
# ─────────────────────────────────────────────

print("\n[2/6] Fine-tuning YOLOv11 Classification...")

# yolo11n-cls.pt = modèle officiel Ultralytics pour classification d'images
# best_nano_111.pt est un modèle de DÉTECTION → utilisé uniquement en section 5
yolo_cls = YOLO(CLS_BASE_MODEL)

yolo_cls.train(
    task     = "classify",
    data     = CLASS_TRAIN,
    epochs   = 20,
    imgsz    = 224,
    batch    = 32,
    lr0      = 0.001,
    name     = "yolo11_classify",
    project  = os.path.join(OUTPUT_DIR, "cls_runs"),
    patience = 7,
    exist_ok = True,
    device   = DEVICE,
    save     = True,
    augment  = True,
    degrees  = 15,
    fliplr   = 0.5,
    hsv_v    = 0.3,
    hsv_s    = 0.3,
)

# Locate best classification checkpoint
best_cls_list = glob.glob(
    os.path.join(OUTPUT_DIR, "cls_runs/**/best.pt"), recursive=True
)
if not best_cls_list:
    raise FileNotFoundError("Classification best.pt not found!")
best_cls_path = best_cls_list[0]
print(f"  Best classification model: {best_cls_path}")


# ─────────────────────────────────────────────
# 4. EVALUATE YOLOv11 CLASSIFICATION
# ─────────────────────────────────────────────

print("\n[3/6] Évaluation de la classification YOLOv11...")

yolo_cls_best = YOLO(best_cls_path)

# Run validation on the val split
cls_metrics = yolo_cls_best.val(
    task    = "classify",
    data    = CLASS_TRAIN,
    split   = "val",
    imgsz   = 224,
    device  = DEVICE,
)

cls_top1 = float(cls_metrics.top1) * 100   # Top-1 accuracy
cls_top5 = float(cls_metrics.top5) * 100   # Top-5 accuracy
CLASS_NAMES = yolo_cls_best.names           # dict {0: 'fire', 1: 'no_fire'}

print(f"\n  Top-1 Accuracy : {cls_top1:.2f}%")
print(f"  Top-5 Accuracy : {cls_top5:.2f}%")

# ── Manual inference on test folder for confusion matrix ──
from torchvision import datasets as tvd, transforms as tvt
from torch.utils.data import DataLoader, random_split, Subset

tf_test = tvt.Compose([
    tvt.Resize((224, 224)),
    tvt.ToTensor(),
    tvt.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
full_ds   = tvd.ImageFolder(CLASS_TRAIN)
total     = len(full_ds)
n_test    = int(0.15 * total)
n_train   = total - n_test
_, test_ds = random_split(
    full_ds, [n_train, n_test],
    generator=torch.Generator().manual_seed(42)
)

test_img_paths  = [full_ds.samples[i][0] for i in test_ds.indices]
test_img_labels = [full_ds.samples[i][1] for i in test_ds.indices]
cls_names_list  = full_ds.classes   # ['fire', 'no_fire']

preds_cls, labels_cls = [], []
for img_path, true_lbl in tqdm(
    zip(test_img_paths, test_img_labels),
    total=len(test_img_paths),
    desc="Classifying test set"
):
    res = yolo_cls_best(img_path, verbose=False)[0]
    pred_lbl = int(res.probs.top1)
    preds_cls.append(pred_lbl)
    labels_cls.append(true_lbl)

print("\n" + "=" * 50)
print("RAPPORT CLASSIFICATION YOLOv11")
print("=" * 50)
cls_report = classification_report(
    labels_cls, preds_cls, target_names=cls_names_list, digits=4
)
print(cls_report)
cls_acc = sum(p == l for p, l in zip(preds_cls, labels_cls)) / len(labels_cls) * 100
print(f"Accuracy globale : {cls_acc:.2f}%")

with open(os.path.join(OUTPUT_DIR, "yolo11_cls_report.txt"), "w") as f:
    f.write("RAPPORT CLASSIFICATION YOLOv11\n" + "=" * 50 + "\n")
    f.write(cls_report)
    f.write(f"\nAccuracy: {cls_acc:.2f}%\n")

# Confusion matrix
cm = confusion_matrix(labels_cls, preds_cls)
plt.figure(figsize=(7, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=cls_names_list, yticklabels=cls_names_list,
            annot_kws={"size": 14})
plt.title("YOLOv11 Classification — Matrice de Confusion", fontsize=14, fontweight='bold')
plt.xlabel("Prédiction", fontsize=12); plt.ylabel("Réalité", fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "yolo11_cls_confusion_matrix.png"), dpi=200, bbox_inches='tight')
plt.show()
print("  Confusion matrix sauvegardée")


# ─────────────────────────────────────────────
# 5. YOLOv11 DETECTION FINE-TUNING
#    Continue from the same pretrained weights
# ─────────────────────────────────────────────

print("\n[4/6] Fine-tuning YOLOv11 Détection...")

yolo_det = YOLO(DET_PRETRAINED)   # modèle pré-entraîné feu/fumée

yolo_det.train(
    task     = "detect",
    data     = DETECT_YAML,
    epochs   = 20,
    imgsz    = 640,
    batch    = 16,
    lr0      = 0.001,
    name     = "yolo11_detect",
    project  = os.path.join(OUTPUT_DIR, "det_runs"),
    patience = 10,
    exist_ok = True,
    device   = DEVICE,
    save     = True,
)

best_det_list = glob.glob(
    os.path.join(OUTPUT_DIR, "det_runs/**/best.pt"), recursive=True
)
if not best_det_list:
    raise FileNotFoundError("Detection best.pt not found!")
best_det_path = best_det_list[0]
print(f"  Best detection model: {best_det_path}")


# ─────────────────────────────────────────────
# 6. EVALUATE YOLOv11 DETECTION
# ─────────────────────────────────────────────

print("\n[5/6] Évaluation de la détection YOLOv11...")

yolo_det_best = YOLO(best_det_path)
det_metrics   = yolo_det_best.val(data=DETECT_YAML, split="test", device=DEVICE)

yolo_map50    = float(det_metrics.box.map50) * 100
yolo_map5095  = float(det_metrics.box.map)   * 100
yolo_prec     = float(det_metrics.box.mp)    * 100
yolo_rec      = float(det_metrics.box.mr)    * 100

print("\n" + "=" * 50)
print("RAPPORT DÉTECTION YOLOv11")
print("=" * 50)
print(f"mAP@0.5      : {yolo_map50:.2f}%")
print(f"mAP@0.5:0.95 : {yolo_map5095:.2f}%")
print(f"Precision    : {yolo_prec:.2f}%")
print(f"Recall       : {yolo_rec:.2f}%")


# ─────────────────────────────────────────────
# 7. PIPELINE VISUALIZATION: Classify → Detect
# ─────────────────────────────────────────────

print("\n[6/6] Visualisation du pipeline Classification → Détection...")

fire_class_idx = cls_names_list.index('fire') if 'fire' in cls_names_list else 0
fire_samples   = [
    (p, l) for p, l in zip(test_img_paths, test_img_labels)
    if l == fire_class_idx
][:8]

fig, axes = plt.subplots(2, 4, figsize=(18, 9))
fig.suptitle(
    "Pipeline YOLOv11 : Classification → Détection (Approche YOLO11 Only)",
    fontsize=14, fontweight='bold'
)

for i, (img_path, _) in enumerate(fire_samples):
    ax = axes[i // 4][i % 4]
    pil_img = Image.open(img_path).convert("RGB")

    # Step 1 — Classification
    cls_res    = yolo_cls_best(img_path, verbose=False)[0]
    pred_class = int(cls_res.probs.top1)
    conf_cls   = float(cls_res.probs.top1conf)

    # Step 2 — Detection only if classified as fire
    if pred_class == fire_class_idx:
        det_res = yolo_det_best(img_path, verbose=False)[0]
        ax.imshow(pil_img)
        if det_res.boxes is not None and len(det_res.boxes) > 0:
            for box in det_res.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf_det = float(box.conf[0])
                w_s = pil_img.width  / det_res.orig_shape[1]
                h_s = pil_img.height / det_res.orig_shape[0]
                rect = patches.Rectangle(
                    (x1 * w_s, y1 * h_s),
                    (x2 - x1) * w_s, (y2 - y1) * h_s,
                    linewidth=2, edgecolor='red', facecolor='none'
                )
                ax.add_patch(rect)
                ax.text(
                    x1 * w_s, y1 * h_s - 5,
                    f"Fire {conf_det:.2f}", color='red',
                    fontsize=8, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='yellow', alpha=0.7)
                )
            title = f"✅ FIRE\nCls: {conf_cls*100:.1f}% | {len(det_res.boxes)} bbox"
            color = 'red'
        else:
            title = f"✅ Cls=Fire ({conf_cls*100:.1f}%)\nDet: 0 bbox"
            color = 'orange'
    else:
        ax.imshow(pil_img)
        title = f"🟢 No Fire ({conf_cls*100:.1f}%)\nDet: skipped"
        color = 'gray'

    ax.set_title(title, fontsize=8, color=color)
    ax.axis('off')

plt.tight_layout()
viz_path = os.path.join(OUTPUT_DIR, "pipeline_yolo11_visualizations.png")
plt.savefig(viz_path, dpi=200, bbox_inches='tight')
plt.show()
print(f"  Visualisations sauvegardées: {viz_path}")


# ─────────────────────────────────────────────
# 8. BENCHMARK TIMING
# ─────────────────────────────────────────────

print("\nMesure du temps de traitement du pipeline...")
n_bench = min(50, len(fire_samples))
t_cls_list, t_det_list, t_tot_list = [], [], []

for img_path, _ in fire_samples[:n_bench]:
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
print(f"Temps moyen Classification : {avg_cls:.1f} ms")
print(f"Temps moyen Détection      : {avg_det:.1f} ms")
print(f"Temps total pipeline       : {avg_tot:.1f} ms  ({1000/avg_tot:.1f} FPS)")


# ─────────────────────────────────────────────
# 9. RÉSUMÉ FINAL — JSON
# ─────────────────────────────────────────────

results_summary = {
    "Approche"         : "YOLO11 Only — Classification + Détection",
    "Pretrained_model" : DET_MODEL_NAME,
    "Source"           : "https://github.com/sayedgamal99/Real-Time-Smoke-Fire-Detection-YOLO11",
    "Classification": {
        "Modele"       : f"YOLOv11n-cls (yolo11n-cls.pt)",
        "Top1_Accuracy": round(cls_top1, 2),
        "Top5_Accuracy": round(cls_top5, 2),
        "Test_Accuracy": round(cls_acc, 2),
    },
    "Detection": {
        "Modele"       : f"YOLOv11 Detect ({DET_MODEL_NAME})",
        "mAP_50"       : round(yolo_map50, 2),
        "mAP_50_95"    : round(yolo_map5095, 2),
        "Precision"    : round(yolo_prec, 2),
        "Recall"       : round(yolo_rec, 2),
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
print("  APPROCHE YOLO11 ONLY TERMINÉE — RÉSULTATS FINAUX")
print("=" * 60)
print(json.dumps(results_summary, indent=2, ensure_ascii=False))
print(f"\nTous les fichiers sauvegardés dans: {OUTPUT_DIR}")
