#  APPROCHE 1 : R-CNN (Classification) → YOLOv11s (Détection)
#  Pipeline: Image → R-CNN (fire/no fire?) → If fire → YOLO (where?)

import os, glob, torch, torch.nn as nn, torch.optim as optim, time,  yaml
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split, Subset
from ultralytics import YOLO
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import json
import numpy as np
from tqdm import tqdm
from PIL import Image


# 1. PATHS

CLASS_TRAIN = "/content/drive/MyDrive/MEMOIRE/ForestFireDataset(Classifications)/ForestFireDataset/train"
DETECT_YAML = "/content/drive/MyDrive/MEMOIRE/ForesFireDataset(ObjectDetection)/data.yaml"


OUTPUT_DIR = "/content/drive/MyDrive/MEMOIRE/Approche1_Results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("  APPROCHE 1 : R-CNN → YOLOv11s  ")
print(f"Classification dataset : {CLASS_TRAIN}")
print(f"Detection YAML         : {DETECT_YAML}")
print(f"Résultats sauvegardés  : {OUTPUT_DIR}")
print(f"Classes trouvées       : {os.listdir(CLASS_TRAIN)}")


# 2. CONFIG

BATCH_SIZE  = 32
NUM_EPOCHS  = 15
LR          = 0.001
NUM_CLASSES = 2
IMG_SIZE    = 224
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nDevice: {DEVICE}")


# 3. DATASET — Split 70 / 15 / 15

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
CLASS_NAMES  = full_dataset.classes   # ex: ['fire', 'no_fire']
total        = len(full_dataset)
n_train      = int(0.70 * total)
n_valid      = int(0.15 * total)
n_test       = total - n_train - n_valid

print(f"\nDataset: {total} images | Classes: {CLASS_NAMES}")

train_ds, valid_ds, test_ds = random_split(
    full_dataset, [n_train, n_valid, n_test],
    generator=torch.Generator().manual_seed(42)
)

valid_ds_proper = Subset(
    datasets.ImageFolder(CLASS_TRAIN, tf_val),
    valid_ds.indices
)
test_ds_proper = Subset(
    datasets.ImageFolder(CLASS_TRAIN, tf_val),
    test_ds.indices
)

train_dl = DataLoader(train_ds,        BATCH_SIZE, shuffle=True,  num_workers=2, pin_memory=True)
valid_dl = DataLoader(valid_ds_proper, BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
test_dl  = DataLoader(test_ds_proper,  BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

print(f"Train: {n_train} | Valid: {n_valid} | Test: {n_test}")



# 4. R-CNN MODEL (ResNet50 fine-tuned)

print("\n[1/5] Création du modèle R-CNN (ResNet50)...")

model = models.resnet50(weights="IMAGENET1K_V1")

# Dégeler les 2 derniers blocs pour un meilleur fine-tuning
for name, param in model.named_parameters():
    param.requires_grad = False
for name, param in model.layer4.named_parameters():
    param.requires_grad = True

# Nouvelle tête de classification
model.fc = nn.Sequential(
    nn.Linear(model.fc.in_features, 512),
    nn.BatchNorm1d(512),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(512, NUM_CLASSES)
)
model = model.to(DEVICE)
print("R-CNN prêt!")

criterion = nn.CrossEntropyLoss()
# Optimizer différentié: lr faible pour layer4, lr normal pour fc
optimizer = optim.Adam([
    {'params': model.layer4.parameters(), 'lr': LR * 0.1},
    {'params': model.fc.parameters(),     'lr': LR}
])
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)


# 5. TRAINING R-CNN

print("\n[2/5] Entraînement R-CNN...")
history  = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
best_acc = 0.0
best_epoch = 0

for epoch in range(NUM_EPOCHS):
    # --- Train ---
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

    # --- Validation ---
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

print(f"\nMeilleur modèle: Epoch {best_epoch} avec Val Acc = {best_acc:.1f}%")



# 6. EVALUATION R-CNN (TEST SET)

print("\n[3/5] Évaluation R-CNN sur le test set...")
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

# Rapport complet
print("\n" + "="*50)
print("RAPPORT R-CNN (Classification)")
print("="*50)
report = classification_report(labels, preds, target_names=CLASS_NAMES, digits=4)
print(report)
rcnn_acc = sum(p == l for p, l in zip(preds, labels)) / len(labels) * 100
print(f"Accuracy globale: {rcnn_acc:.2f}%")

# Sauvegarde du rapport texte
with open(os.path.join(OUTPUT_DIR, "rcnn_classification_report.txt"), "w") as f:
    f.write("RAPPORT R-CNN - Approche 1\n")
    f.write("="*50 + "\n")
    f.write(report)
    f.write(f"\nAccuracy: {rcnn_acc:.2f}%\n")

# --- Confusion Matrix ---
cm = confusion_matrix(labels, preds)
plt.figure(figsize=(7, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
            annot_kws={"size": 14})
plt.title("R-CNN — Matrice de Confusion (Approche 1)", fontsize=14, fontweight='bold')
plt.xlabel("Prédiction", fontsize=12)
plt.ylabel("Réalité", fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "rcnn_confusion_matrix.png"), dpi=200, bbox_inches='tight')
plt.show()
print(" Confusion matrix sauvegardée")

# --- Learning Curves ---

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("R-CNN — Courbes d'Apprentissage (Approche 1)", fontsize=14, fontweight='bold')

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
print(" Learning curves sauvegardées")



# 7. YOLO TRAINING (yolo11s = small)

print("\n[4/5] Entraînement YOLOv11s...")
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


# 8. YOLO EVALUATION

print("\n[5/5] Évaluation YOLOv11s...")

# Recherche automatique de best.pt
best_pt_list = glob.glob(os.path.join(OUTPUT_DIR, "yolo_runs/**/best.pt"), recursive=True)
if not best_pt_list:
    best_pt_list = glob.glob("/content/**/best.pt", recursive=True)

if not best_pt_list:
    raise FileNotFoundError("best.pt introuvable! Vérifiez le chemin du projet YOLO.")

best_pt_path = best_pt_list[0]
print(f"best.pt trouvé: {best_pt_path}")

yolo_best = YOLO(best_pt_path)
metrics   = yolo_best.val(data=DETECT_YAML, split="test")

yolo_map50    = float(metrics.box.map50)   * 100
yolo_map5095  = float(metrics.box.map)     * 100
yolo_precision= float(metrics.box.mp)      * 100
yolo_recall   = float(metrics.box.mr)      * 100

print("\n" + "="*50)
print("RAPPORT YOLO (Détection)")
print("="*50)
print(f"mAP@0.5      : {yolo_map50:.2f}%")
print(f"mAP@0.5:0.95 : {yolo_map5095:.2f}%")
print(f"Precision    : {yolo_precision:.2f}%")
print(f"Recall       : {yolo_recall:.2f}%")


# 8b. MATRICE DE CONFUSION DÉTECTION — Strong / Medium / Weak Fire
    print("\nGénération de la matrice de confusion détection (niveaux de feu)...")

    # Catégorisation basée sur mAP@0.5
    def categorize_fire(map50_score):
        if map50_score >= 70:
            return "Strong\nFire\n(mAP≥70%)"
        elif map50_score >= 30:
            return "Medium\nFire\n(30%≤mAP<70%)"
        else:
            return "Weak\nFire\n(mAP<30%)"

    # Construire la matrice de confusion détection par image
    # On évalue chaque image du test set YOLO individuellement
    yolo_test_results = yolo_best.val(data=DETECT_YAML, split="test", verbose=False)

    # Récupérer les stats par image via predict sur le dossier test
    with open(DETECT_YAML, 'r') as f:
        yaml_data = yaml.safe_load(f)
    test_images_dir = yaml_data.get('test', '')
    if not os.path.isabs(test_images_dir):
        test_images_dir = os.path.join(os.path.dirname(DETECT_YAML), test_images_dir)

    # Prédictions sur chaque image test
    all_img_paths = glob.glob(os.path.join(test_images_dir, "*.jpg")) + \
                    glob.glob(os.path.join(test_images_dir, "*.png"))

    per_image_conf = []  # confidence moyenne par image détectée
    n_detected     = 0
    n_missed       = 0   # images sans détection (faux négatifs)
    n_false_alarm  = 0   # sera estimé via les images nofire si disponibles

    for img_path in all_img_paths[:200]:  # limiter pour la vitesse
        res = yolo_best(img_path, verbose=False)[0]
        if res.boxes is not None and len(res.boxes) > 0:
            confs = [float(b.conf[0]) for b in res.boxes]
            avg_conf = np.mean(confs) * 100
            per_image_conf.append(avg_conf)
            n_detected += 1
        else:
            per_image_conf.append(0)
            n_missed += 1

    # Classer chaque détection en Strong/Medium/Weak selon conf moyenne
    strong = sum(1 for c in per_image_conf if c >= 70)
    medium = sum(1 for c in per_image_conf if 30 <= c < 70)
    weak   = sum(1 for c in per_image_conf if 0 < c < 30)
    missed = n_missed

    # Matrice : Prédit (colonnes) vs Réel (lignes)
    # Ici on construit une matrice de performance par niveau
    # Lignes = niveau réel estimé par mAP global
    # Colonnes = niveau prédit par confidence image

    map_val = yolo_map50  # déjà calculé (en %)

    # Matrice 3x3 : Strong / Medium / Weak
    # On distribue les détections selon leur conf vs le niveau global mAP
    det_matrix = np.zeros((3, 3), dtype=int)
    categories = ['Strong\nFire', 'Medium\nFire', 'Weak\nFire']

    for conf in per_image_conf:
        if conf == 0:
            continue  # non détecté, ignoré ici
        # Niveau prédit (par conf image)
        if conf >= 70:   pred_lvl = 0  # Strong
        elif conf >= 30: pred_lvl = 1  # Medium
        else:            pred_lvl = 2  # Weak

        # Niveau réel (par mAP global du modèle)
        if map_val >= 70:   true_lvl = 0
        elif map_val >= 30: true_lvl = 1
        else:               true_lvl = 2

        det_matrix[true_lvl][pred_lvl] += 1

    # Affichage
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Analyse de Détection YOLO — Niveaux de Feu (Approche 1)",
                 fontsize=14, fontweight='bold')

    # --- Subplot 1 : Matrice de confusion détection ---
    ax1 = axes[0]
    sns.heatmap(det_matrix, annot=True, fmt='d', cmap='YlOrRd',
                xticklabels=categories, yticklabels=categories,
                annot_kws={"size": 13, "weight": "bold"},
                ax=ax1, linewidths=0.5, linecolor='gray')
    ax1.set_title("Matrice de Confusion — Détection\n(Niveau Réel vs Niveau Prédit)",
                  fontsize=12, fontweight='bold')
    ax1.set_xlabel("Niveau Prédit\n(par confidence image)", fontsize=11)
    ax1.set_ylabel("Niveau Réel\n(par mAP@0.5 global)", fontsize=11)

    # --- Subplot 2 : Distribution des niveaux détectés ---
    ax2 = axes[1]
    labels_bar  = ['Strong\nFire\n(conf≥70%)', 'Medium\nFire\n(30-70%)',
                   'Weak\nFire\n(conf<30%)', 'Non\nDétecté']
    values_bar  = [strong, medium, weak, missed]
    colors_bar  = ['#d32f2f', '#f57c00', '#fbc02d', '#90a4ae']

    bars = ax2.bar(labels_bar, values_bar, color=colors_bar, edgecolor='black', linewidth=0.8)
    for bar, val in zip(bars, values_bar):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                 str(val), ha='center', va='bottom', fontweight='bold', fontsize=12)

    # Ligne mAP
    map_color = '#d32f2f' if map_val >= 70 else ('#f57c00' if map_val >= 30 else '#fbc02d')
    fire_level = categorize_fire(map_val).replace('\n', ' ')
    ax2.set_title(f"Distribution des Détections par Niveau\nmAP@0.5 = {map_val:.1f}% → {fire_level}",
                  fontsize=12, fontweight='bold', color=map_color)
    ax2.set_ylabel("Nombre d'images", fontsize=11)
    ax2.set_xlabel("Niveau de Feu Détecté", fontsize=11)
    ax2.grid(axis='y', alpha=0.4)

    # Légende mAP
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#d32f2f', label='Strong Fire : mAP ≥ 70%'),
        Patch(facecolor='#f57c00', label='Medium Fire : 30% ≤ mAP < 70%'),
        Patch(facecolor='#fbc02d', label='Weak Fire   : mAP < 30%'),
    ]
    ax2.legend(handles=legend_elements, loc='upper right', fontsize=9)

    plt.tight_layout()
    det_matrix_path = os.path.join(OUTPUT_DIR, "yolo_detection_fire_levels.png")
    plt.savefig(det_matrix_path, dpi=200, bbox_inches='tight')
    plt.show()
    print(f"Matrice détection niveaux de feu sauvegardée: {det_matrix_path}")

    # Résumé textuel
    print(f"\n{'='*50}")
    print(f"ANALYSE NIVEAUX DE FEU (mAP@0.5 = {map_val:.1f}%)")
    print(f"{'='*50}")
    print(f"Niveau global du modèle : {categorize_fire(map_val).replace(chr(10), ' ')}")
    print(f"Images Strong Fire (conf≥70%) : {strong}")
    print(f"Images Medium Fire (30-70%)   : {medium}")
    print(f"Images Weak Fire   (conf<30%) : {weak}")
    print(f"Images Non détectées          : {missed}")



# 9. PIPELINE COMPLET: R-CNN → YOLO
#    Visualisation des Bounding Boxes 

print("\nVisualisation du pipeline R-CNN → YOLO sur des exemples de test...")

# Récupération des images du test set
test_img_paths = []
test_img_labels = []
base_dataset = datasets.ImageFolder(CLASS_TRAIN)

for idx in test_ds.indices:
    path, label = base_dataset.samples[idx]
    test_img_paths.append(path)
    test_img_labels.append(label)

# On prend 8 images fire pour visualiser le pipeline
fire_class_idx = CLASS_NAMES.index('fire') if 'fire' in CLASS_NAMES else 0
fire_samples = [(p, l) for p, l in zip(test_img_paths, test_img_labels) if l == fire_class_idx][:8]

tf_inference = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

fig, axes = plt.subplots(2, 4, figsize=(18, 9))
fig.suptitle("Pipeline R-CNN → YOLO : Exemples de Détection (Approche 1)",
             fontsize=14, fontweight='bold')

model.eval()

for i, (img_path, true_label) in enumerate(fire_samples):
    ax = axes[i // 4][i % 4]

    # Étape 1: R-CNN classification
    pil_img = Image.open(img_path).convert("RGB")
    t0 = time.time()
    inp = tf_inference(pil_img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        out   = model(inp)
        probs = torch.softmax(out, dim=1)[0]
        pred_class = out.argmax(1).item()
    rcnn_time = (time.time() - t0) * 1000

    # Étape 2: YOLO détection (SEULEMENT si R-CNN dit "fire")
    if pred_class == fire_class_idx:
        t1 = time.time()
        results = yolo_best(img_path, verbose=False)[0]
        yolo_time = (time.time() - t1) * 1000

        # Affichage image avec bounding boxes
        ax.imshow(pil_img)
        if results.boxes is not None and len(results.boxes) > 0:
            for box in results.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0])
                w_scale = pil_img.width  / results.orig_shape[1]
                h_scale = pil_img.height / results.orig_shape[0]
                rect = patches.Rectangle(
                    (x1 * w_scale, y1 * h_scale),
                    (x2 - x1) * w_scale, (y2 - y1) * h_scale,
                    linewidth=2, edgecolor='red', facecolor='none'
                )
                ax.add_patch(rect)
                ax.text(x1 * w_scale, y1 * h_scale - 5,
                        f"Fire {conf:.2f}", color='red',
                        fontsize=8, fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.2', facecolor='yellow', alpha=0.7))
            status = f" FIRE détecté\nR-CNN: {probs[fire_class_idx]*100:.1f}%\nYOLO: {len(results.boxes)} bbox"
        else:
            status = f" R-CNN=Fire\nmais YOLO: 0 bbox"

        ax.set_title(status, fontsize=8, color='red')
    else:
        ax.imshow(pil_img)
        ax.set_title(f" R-CNN: No Fire\n({probs[1-fire_class_idx]*100:.1f}%)\nYOLO ignoré", fontsize=8, color='gray')

    ax.axis('off')

plt.tight_layout()
pipeline_fig_path = os.path.join(OUTPUT_DIR, "pipeline_rcnn_yolo_visualizations.png")
plt.savefig(pipeline_fig_path, dpi=200, bbox_inches='tight')
plt.show()
print(f" Visualisations pipeline sauvegardées: {pipeline_fig_path}")


# 10. BENCHMARK: temps de traitement du pipeline

print("\nMesure du temps de traitement du pipeline...")
n_benchmark = min(50, len(fire_samples))
times_rcnn, times_yolo, times_total = [], [], []

for img_path, _ in fire_samples[:n_benchmark]:
    pil_img = Image.open(img_path).convert("RGB")
    inp = tf_inference(pil_img).unsqueeze(0).to(DEVICE)

    t0 = time.time()
    with torch.no_grad():
        out = model(inp)
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



# 11. RÉSUMÉ FINAL — JSON (pour comparaison avec Approche 2)

results_summary = {
    "Approche"              : "Approche 1 — R-CNN + YOLOv11s",
    "Classification": {
        "Modele"            : "ResNet50 (R-CNN)",
        "Accuracy"          : round(rcnn_acc, 2),
        "Best_Val_Accuracy" : round(best_acc, 2),
        "Best_Epoch"        : best_epoch,
    },
    "Detection": {
        "Modele"            : "YOLOv11s",
        "mAP_50"            : round(yolo_map50, 2),
        "mAP_50_95"         : round(yolo_map5095, 2),
        "Precision"         : round(yolo_precision, 2),
        "Recall"            : round(yolo_recall, 2),
    },
    "Pipeline_Timing_ms": {
        "R_CNN_avg"         : round(avg_rcnn, 2),
        "YOLO_avg"          : round(avg_yolo, 2),
        "Total_avg"         : round(avg_total, 2),
        "FPS_approx"        : round(1000 / avg_total, 1),
    }
}

json_path = os.path.join(OUTPUT_DIR, "approche1_results.json")
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(results_summary, f, indent=2, ensure_ascii=False)

print("\n" + "=" * 60)
print("  APPROCHE 1 TERMINÉE — RÉSULTATS FINAUX")
print("=" * 60)
print(json.dumps(results_summary, indent=2, ensure_ascii=False))
print(f"\nTous les fichiers sauvegardés dans: {OUTPUT_DIR}")
