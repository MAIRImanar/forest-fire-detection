# ============================================================
#  APPROCHE 1 : R-CNN (Classification) + YOLOv11s (Détection)
#  Dataset: Shamta & Demir 2024
# ============================================================

import os, glob, torch, torch.nn as nn, torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
from ultralytics import YOLO
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import json
from tqdm import tqdm


# 1. PATHS

CLASS_TRAIN = "/content/drive/MyDrive/MEMOIRE/ForestFireDataset(Classifications)/ForestFireDataset/train"
DETECT_YAML = "/content/drive/MyDrive/MEMOIRE/ForesFireDataset(ObjectDetection)/data.yaml"

print("Paths:")
print(f"   Classification : {CLASS_TRAIN}")
print(f"   Detection YAML : {DETECT_YAML}")
print(f"   Classes        : {os.listdir(CLASS_TRAIN)}")


# 2. CONFIG

BATCH_SIZE  = 32
NUM_EPOCHS  = 15
LR          = 0.001
NUM_CLASSES = 2
IMG_SIZE    = 224
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n Device: {DEVICE}")


# 3. DATASET - Split 70/15/15

tf_train = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(0.3, 0.3, 0.3),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])
tf_val = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

full_dataset = datasets.ImageFolder(CLASS_TRAIN, tf_train)
CLASS_NAMES  = full_dataset.classes
total        = len(full_dataset)
n_train      = int(0.70 * total)
n_valid      = int(0.15 * total)
n_test       = total - n_train - n_valid

print(f"\n Dataset total : {total} images")
print(f"   Classes : {CLASS_NAMES}")

train_ds, valid_ds, test_ds = random_split(
    full_dataset, [n_train, n_valid, n_test],
    generator=torch.Generator().manual_seed(42)
)
valid_ds.dataset.transform = tf_val
test_ds.dataset.transform  = tf_val

train_dl = DataLoader(train_ds, BATCH_SIZE, shuffle=True,  num_workers=2)
valid_dl = DataLoader(valid_ds, BATCH_SIZE, shuffle=False, num_workers=2)
test_dl  = DataLoader(test_ds,  BATCH_SIZE, shuffle=False, num_workers=2)

print(f"   Train:{n_train} | Valid:{n_valid} | Test:{n_test}")


# 4. R-CNN MODEL (ResNet50)

print("\n Creation R-CNN (ResNet50)...")
model = models.resnet50(weights="IMAGENET1K_V1")
for p in model.parameters():
    p.requires_grad = False
model.fc = nn.Sequential(
    nn.Linear(model.fc.in_features, 512),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(512, NUM_CLASSES)
)
model = model.to(DEVICE)
print("R-CNN pret!")

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=LR)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)


# 5. TRAINING R-CNN

print("\n Entrainement R-CNN...")
history  = {"train_loss":[],"train_acc":[],"val_loss":[],"val_acc":[]}
best_acc = 0.0

for epoch in range(NUM_EPOCHS):
    model.train()
    tl, tc = 0.0, 0
    for imgs, lbls in tqdm(train_dl, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}"):
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

    ta = tc/n_train*100
    va = vc/n_valid*100
    history["train_loss"].append(tl/n_train)
    history["train_acc"].append(ta)
    history["val_loss"].append(vl/n_valid)
    history["val_acc"].append(va)
    print(f"Epoch {epoch+1:02d} | Train: {ta:.1f}% | Val: {va:.1f}%")

    if va > best_acc:
        best_acc = va
        torch.save(model.state_dict(), "rcnn_best.pth")
        print(f"   Best saved! ({best_acc:.1f}%)")

    scheduler.step()


# 6. TEST R-CNN

print("\n Evaluation R-CNN...")
model.load_state_dict(torch.load("rcnn_best.pth"))
model.eval()
preds, labels = [], []
with torch.no_grad():
    for imgs, lbls in test_dl:
        out = model(imgs.to(DEVICE))
        preds.extend(out.argmax(1).cpu().numpy())
        labels.extend(lbls.numpy())

print(classification_report(labels, preds, target_names=CLASS_NAMES))
rcnn_acc = sum(p==l for p,l in zip(preds,labels))/len(labels)*100
print(f"R-CNN Accuracy: {rcnn_acc:.2f}%")

# Confusion Matrix
cm = confusion_matrix(labels, preds)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
plt.title("R-CNN - Matrice de Confusion (Approche 1)")
plt.tight_layout()
plt.savefig("rcnn_confusion.png", dpi=150)
plt.show()

# Learning Curves
fig, axes = plt.subplots(1, 2, figsize=(12,4))
axes[0].plot(history["train_acc"],  label="Train", color="blue")
axes[0].plot(history["val_acc"],    label="Val",   color="orange")
axes[0].set_title("R-CNN - Accuracy"); axes[0].legend(); axes[0].grid(True)
axes[1].plot(history["train_loss"], label="Train", color="blue")
axes[1].plot(history["val_loss"],   label="Val",   color="orange")
axes[1].set_title("R-CNN - Loss"); axes[1].legend(); axes[1].grid(True)
plt.tight_layout()
plt.savefig("rcnn_curves.png", dpi=150)
plt.show()
print("Courbes sauvegardees!")


# 7. YOLO TRAINING (yolo11s = small)

print("\n Entrainement YOLOv11s...")
yolo = YOLO("yolo11s.pt")
yolo.train(
    data     = DETECT_YAML,
    epochs   = 50,
    imgsz    = 640,
    batch    = 16,
    name     = "approche1",
    project  = "yolo_results",
    patience = 10,
    exist_ok = True,
    device   = 0 if torch.cuda.is_available() else 'cpu'
)


# 8. YOLO EVALUATION

print("\n Evaluation YOLOv11s...")

# البحث التلقائي عن best.pt
best_pt_list = glob.glob("/content/forest-fire-detection/runs/detect/**/best.pt", recursive=True)
if not best_pt_list:
    best_pt_list = glob.glob("/content/**/best.pt", recursive=True)

print(f"   best.pt trouve : {best_pt_list[0]}")
yolo_best = YOLO(best_pt_list[0])
metrics   = yolo_best.val(data=DETECT_YAML, split="test")

print(f"mAP@0.5   : {metrics.box.map50*100:.2f}%")
print(f"Precision : {metrics.box.mp*100:.2f}%")
print(f"Recall    : {metrics.box.mr*100:.2f}%")


# 9. RESULTATS FINAUX

results = {
    "Approche"      : "R-CNN + YOLOv11s",
    "RCNN_Accuracy" : round(rcnn_acc, 2),
    "YOLO_mAP50"    : round(float(metrics.box.map50)*100, 2),
    "YOLO_Precision": round(float(metrics.box.mp)*100, 2),
    "YOLO_Recall"   : round(float(metrics.box.mr)*100, 2),
}
with open("approche1_results.json","w") as f:
    json.dump(results, f, indent=2)

print("\n APPROCHE 1 TERMINEE!")
print(json.dumps(results, indent=2))
