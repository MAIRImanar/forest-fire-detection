#  APPROCHE 1 - ÉTAPE 3 : YOLOv11 DÉTECTION (bounding boxes)
#  Dataset : ForestFireDataset(ObjectDetection)
#  Modèle  : YOLOv11n 

from ultralytics import YOLO
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import json


# 1. CONFIGURATION

#  MODIFIEZ CE CHEMIN selon votre Google Drive
DATASET_YAML = "/content/drive/MyDrive/ForestFireDataset_Detection/data.yaml"

# Hyperparamètres YOLOv11
EPOCHS      = 50
IMG_SIZE    = 640
BATCH_SIZE  = 16
MODEL_SIZE  = "yolo11n.pt"   # n=nano (léger), s=small, m=medium
PROJECT_DIR = "yolo_results"
RUN_NAME    = "approche1_detection"

print(" YOLOv11 - Détection d'incendies de forêts")
print("=" * 60)
print(f"   Modèle    : {MODEL_SIZE}")
print(f"   Epochs    : {EPOCHS}")
print(f"   Image size: {IMG_SIZE}x{IMG_SIZE}")
print(f"   Dataset   : {DATASET_YAML}")

# 2. CHARGEMENT DU MODÈLE YOLOv11

print("\n Chargement du modèle YOLOv11...")
model = YOLO(MODEL_SIZE)
print(" Modèle YOLOv11 chargé !")

# 3. ENTRAÎNEMENT

print("\n Début de l'entraînement YOLOv11...")
print("=" * 60)

results = model.train(
    data    = DATASET_YAML,
    epochs  = EPOCHS,
    imgsz   = IMG_SIZE,
    batch   = BATCH_SIZE,
    name    = RUN_NAME,
    project = PROJECT_DIR,
    patience= 10,           # Early stopping après 10 epochs sans amélioration
    save    = True,
    plots   = True,
    verbose = True,
    device  = 0 if __import__('torch').cuda.is_available() else 'cpu'
)

print("\n Entraînement YOLOv11 terminé !")

# 4. ÉVALUATION SUR TEST SET

print("\n Évaluation YOLOv11 sur le Test Set...")

best_model_path = f"{PROJECT_DIR}/{RUN_NAME}/weights/best.pt"
model_best = YOLO(best_model_path)

metrics = model_best.val(
    data   = DATASET_YAML,
    split  = "test",
    imgsz  = IMG_SIZE,
    batch  = BATCH_SIZE,
    plots  = True,
    save_json = True
)


# 5. AFFICHAGE DES MÉTRIQUES

print("\n Résultats YOLOv11 :")
print("=" * 60)
print(f"   mAP@0.5     : {metrics.box.map50:.4f}  ({metrics.box.map50*100:.2f}%)")
print(f"   mAP@0.5:0.95: {metrics.box.map:.4f}  ({metrics.box.map*100:.2f}%)")
print(f"   Precision   : {metrics.box.mp:.4f}  ({metrics.box.mp*100:.2f}%)")
print(f"   Recall      : {metrics.box.mr:.4f}  ({metrics.box.mr*100:.2f}%)")


# 6. VISUALISATION DES COURBES YOLO

results_dir = f"{PROJECT_DIR}/{RUN_NAME}"
curves_to_show = ["results.png", "confusion_matrix.png", "PR_curve.png", "F1_curve.png"]

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
axes = axes.flatten()

for i, curve in enumerate(curves_to_show):
    curve_path = os.path.join(results_dir, curve)
    if os.path.exists(curve_path):
        img = mpimg.imread(curve_path)
        axes[i].imshow(img)
        axes[i].set_title(curve.replace(".png", "").replace("_", " ").title())
        axes[i].axis("off")

plt.suptitle("YOLOv11 - Résultats d'Entraînement", fontsize=16, fontweight="bold")
plt.tight_layout()
plt.savefig("yolo_training_results.png", dpi=150)
plt.show()
print(" Résultats YOLO sauvegardés : yolo_training_results.png")

# 7. TEST SUR UNE IMAGE (Prédiction visuelle)

print("\n  Test de prédiction visuelle...")

test_image_path = "/content/drive/MyDrive/ForestFireDataset_Detection/test/images"
test_images = [f for f in os.listdir(test_image_path) if f.endswith(('.jpg', '.png', '.jpeg'))]

if test_images:
    sample_images = test_images[:4]
    results_pred = model_best.predict(
        source=[os.path.join(test_image_path, img) for img in sample_images],
        conf=0.25,
        save=True,
        project="predictions",
        name="sample_predictions"
    )

    fig, axes = plt.subplots(1, len(sample_images), figsize=(20, 5))
    for i, r in enumerate(results_pred):
        img_with_boxes = r.plot()
        axes[i].imshow(img_with_boxes[:, :, ::-1])
        axes[i].set_title(f"Image {i+1}")
        axes[i].axis("off")

    plt.suptitle("YOLOv11 - Prédictions (Détection du feu)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig("yolo_sample_predictions.png", dpi=150)
    plt.show()
    print("📊 Prédictions sauvegardées : yolo_sample_predictions.png")

# Sauvegarde des résultats pour la comparaison finale
yolo_results = {
    "model": "YOLOv11n (Détection)",
    "mAP50": round(float(metrics.box.map50) * 100, 2),
    "mAP50_95": round(float(metrics.box.map) * 100, 2),
    "precision": round(float(metrics.box.mp) * 100, 2),
    "recall": round(float(metrics.box.mr) * 100, 2),
    "best_model_path": best_model_path
}
with open("yolo_detection_results.json", "w") as f:
    json.dump(yolo_results, f)

print("\n Résultats YOLOv11 sauvegardés dans yolo_detection_results.json")
print(f" Meilleur modèle : {best_model_path}")
