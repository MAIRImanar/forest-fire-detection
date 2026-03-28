
#  R-CNN + YOLOv11 - APPROCHE 1 COMPLET
#  Auto-detect dataset paths


import os, torch, torch.nn as nn, torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from ultralytics import YOLO
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import json
from tqdm import tqdm


# 1. PATHS - AUTO DETECT

BASE = "/content/drive/MyDrive/MEMOIRE"

# Classification dataset
CLASS_BASE = f"{BASE}/ForestFireDataset(Classifications)/ForestFireDataset"

# Detection dataset  
DETECT_BASE = f"{BASE}/ForesFireDataset(ObjectDetection)"
DETECT_YAML = f"{DETECT_BASE}/data.yaml"

# Auto-detect train/valid/test
def find_split(base, split):
    for name in [split, split.capitalize()]:
        p = os.path.join(base, name)
        if os.path.exists(p):
            return p
    # Si valid n'existe pas, utiliser test
    if split == "valid":
        return find_split(base, "test")
    raise FileNotFoundError(f"Split '{split}' non trouvé dans {base}")

TRAIN_DIR = find_split(CLASS_BASE, "train")
VALID_DIR = find_split(CLASS_BASE, "valid")
TEST_DIR  = find_split(CLASS_BASE, "test")

print(" Paths détectés automatiquement:")
print(f"   Train : {TRAIN_DIR}")
print(f"   Valid : {VALID_DIR}")
print(f"   Test  : {TEST_DIR}")
print(f"   YAML  : {DETECT_YAML}")


# 2. CONFIG

BATCH_SIZE    = 32
NUM_EPOCHS    = 15
LR            = 0.001
NUM_CLASSES   = 2
IMG_SIZE      = 224
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n  Device: {DEVICE}")


# 3. DATA TRANSFORMS

train_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(0.3, 0.3, 0.3),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])
val_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

# ─────────────────────────────────────────
# 4. DATASETS
# ─────────────────────────────────────────
print("\n Chargement dataset...")
train_ds = datasets.ImageFolder(TRAIN_DIR, train_tf)
valid_ds = datasets.ImageFolder(VALID_DIR, val_tf)
test_ds  = datasets.ImageFolder(TEST_DIR,  val_tf)

train_dl = DataLoader(train_ds, BATCH_SIZE, shuffle=True,  num_workers=2)
valid_dl = DataLoader(valid_ds, BATCH_SIZE, shuffle=False, num_workers=2)
test_dl  = DataLoader(test_ds,  BATCH_SIZE, shuffle=False, num_workers=2)

CLASS_NAMES = train_ds.classes
print(f" Classes: {CLASS_NAMES}")
print(f"   Train:{len(train_ds)} | Valid:{len(valid_ds)} | Test:{len(test_ds)}")


# 5. R-CNN MODEL (ResNet50)

print("\n Création R-CNN...")
model = models.resnet50(pretrained=True)
for p in model.parameters():
    p.requires_grad = False
model.fc = nn.Sequential(
    nn.Linear(model.fc.in_features, 512),
    nn.ReLU(), nn.Dropout(0.5),
    nn.Linear(512, NUM_CLASSES)
)
model = model.to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=LR)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)


# 6. TRAINING R-CNN

print("\n Entraînement R-CNN...")
history = {"train_loss":[],"train_acc":[],"val_loss":[],"val_acc":[]}
best_acc = 0.0

for epoch in range(NUM_EPOCHS):
    # Train
    model.train()
    tl, tc = 0.0, 0
    for imgs, lbls in tqdm(train_dl, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}"):
        imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
        optimizer.zero_grad()
        out = model(imgs)
        loss = criterion(out, lbls)
        loss.backward()
        optimizer.step()
        tl += loss.item() * imgs.size(0)
        tc += (out.argmax(1) == lbls).sum().item()

    # Validation
    model.eval()
    vl, vc = 0.0, 0
    with torch.no_grad():
        for imgs, lbls in valid_dl:
            imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
            out = model(imgs)
            vl += criterion(out, lbls).item() * imgs.size(0)
            vc += (out.argmax(1) == lbls).sum().item()

    ta = tc/len(train_ds)*100
    va = vc/len(valid_ds)*100
    history["train_loss"].append(tl/len(train_ds))
    history["train_acc"].append(ta)
    history["val_loss"].append(vl/len(valid_ds))
    history["val_acc"].append(va)

    print(f"Epoch {epoch+1:02d} | Train:{ta:.1f}% | Val:{va:.1f}%")

    if va > best_acc:
        best_acc = va
        torch.save(model.state_dict(), "rcnn_best.pth")
        print(f"    Saved! Best Val: {best_acc:.1f}%")

    scheduler.step()


# 7. TEST R-CNN

print("\n Test R-CNN...")
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
print(f" R-CNN Test Accuracy: {rcnn_acc:.2f}%")

# Plot confusion matrix
cm = confusion_matrix(labels, preds)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
plt.title("R-CNN Confusion Matrix")
plt.savefig("rcnn_confusion.png", dpi=150)
plt.show()


# 8. YOLO TRAINING

print("\n Entraînement YOLOv11...")
yolo = YOLO("yolo11n.pt")
yolo.train(
    data    = DETECT_YAML,
    epochs  = 50,
    imgsz   = 640,
    batch   = 16,
    name    = "approche1",
    project = "yolo_results",
    patience= 10,
    device  = 0 if torch.cuda.is_available() else 'cpu'
)

# YOLO Evaluation
print("\n Test YOLOv11...")
yolo_best = YOLO("yolo_results/approche1/weights/best.pt")
metrics = yolo_best.val(data=DETECT_YAML, split="test")

print(f" mAP@0.5    : {metrics.box.map50*100:.2f}%")
print(f" Precision  : {metrics.box.mp*100:.2f}%")
print(f" Recall     : {metrics.box.mr*100:.2f}%")


# 9. SAVE RESULTS

results = {
    "RCNN_Accuracy": round(rcnn_acc, 2),
    "YOLO_mAP50":    round(metrics.box.map50*100, 2),
    "YOLO_Precision":round(metrics.box.mp*100, 2),
    "YOLO_Recall":   round(metrics.box.mr*100, 2),
}
with open("approche1_results.json","w") as f:
    json.dump(results, f, indent=2)

print("\n APPROCHE 1 TERMINÉE!")
print(json.dumps(results, indent=2))
