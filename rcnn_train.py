#  APPROCHE 1 - ÉTAPE 2 : R-CNN CLASSIFICATION (fire / nofire)
#  Dataset : ForestFireDataset(Classifications)
#  Modèle  : ResNet50 pré-entraîné (Transfer Learning)

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import time
from tqdm import tqdm

#  MODIFIEZ CE CHEMIN selon votre Google Drive
DATASET_PATH = "/content/drive/MyDrive/MEMOIRE/ForestFireDataset(Classifications)/ForestFireDataset"  # Chemin Google Colab

TRAIN_DIR = os.path.join(DATASET_PATH, "train")
VALID_DIR = os.path.join(DATASET_PATH, "test")
TEST_DIR  = os.path.join(DATASET_PATH, "test")

# Hyperparamètres
BATCH_SIZE   = 32
NUM_EPOCHS   = 15
LEARNING_RATE = 0.001
NUM_CLASSES  = 2          # fire, nofire
IMG_SIZE     = 224        # ResNet50 attend 224x224

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f" Device utilisé : {DEVICE}")

# 2. TRANSFORMATIONS DES IMAGES

train_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])  # Normalisation ImageNet
])

val_test_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


# 3. CHARGEMENT DU DATASET

print("\n Chargement du dataset...")

train_dataset = datasets.ImageFolder(root=TRAIN_DIR, transform=train_transforms)
valid_dataset = datasets.ImageFolder(root=VALID_DIR, transform=val_test_transforms)
test_dataset  = datasets.ImageFolder(root=TEST_DIR,  transform=val_test_transforms)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2)
valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

CLASS_NAMES = train_dataset.classes  # ['fire', 'nofire']
print(f" Classes détectées : {CLASS_NAMES}")
print(f"   Train  : {len(train_dataset)} images")
print(f"   Valid  : {len(valid_dataset)} images")
print(f"   Test   : {len(test_dataset)}  images")


# 4. MODÈLE R-CNN (ResNet50 + Transfer Learning)

print("\n Création du modèle R-CNN (ResNet50)...")

model = models.resnet50(pretrained=True)

# Geler les couches de base (Transfer Learning)
for param in model.parameters():
    param.requires_grad = False

# Remplacer la dernière couche (fully connected)
num_features = model.fc.in_features
model.fc = nn.Sequential(
    nn.Linear(num_features, 512),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(512, NUM_CLASSES)
)

model = model.to(DEVICE)
print(f" Modèle R-CNN prêt ! Paramètres entraînables : {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

# 5. LOSS & OPTIMISEUR

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=LEARNING_RATE)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)


# 6. ENTRAÎNEMENT

def train_model(model, train_loader, valid_loader, criterion, optimizer, scheduler, num_epochs):
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc = 0.0

    print("\n Début de l'entraînement R-CNN...")
    print("=" * 60)

    for epoch in range(num_epochs):
        start_time = time.time()

        # ── Phase TRAIN ──
        model.train()
        train_loss, train_correct = 0.0, 0

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]"):
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            train_correct += (preds == labels).sum().item()

        # ── Phase VALIDATION ──
        model.eval()
        val_loss, val_correct = 0.0, 0

        with torch.no_grad():
            for images, labels in valid_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == labels).sum().item()

        # ── Calcul des métriques ──
        epoch_train_loss = train_loss / len(train_dataset)
        epoch_train_acc  = train_correct / len(train_dataset) * 100
        epoch_val_loss   = val_loss / len(valid_dataset)
        epoch_val_acc    = val_correct / len(valid_dataset) * 100
        epoch_time       = time.time() - start_time

        history["train_loss"].append(epoch_train_loss)
        history["train_acc"].append(epoch_train_acc)
        history["val_loss"].append(epoch_val_loss)
        history["val_acc"].append(epoch_val_acc)

        print(f"\nEpoch [{epoch+1:02d}/{num_epochs}] | "
              f"Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc:.2f}% | "
              f"Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc:.2f}% | "
              f"Time: {epoch_time:.1f}s")

        # Sauvegarder le meilleur modèle
        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            torch.save(model.state_dict(), "rcnn_best_model.pth")
            print(f"   Meilleur modèle sauvegardé ! (Val Acc: {best_val_acc:.2f}%)")

        scheduler.step()

    print(f"\n Entraînement terminé ! Meilleure Val Accuracy : {best_val_acc:.2f}%")
    return history

history = train_model(model, train_loader, valid_loader, criterion, optimizer, scheduler, NUM_EPOCHS)


# 7. COURBES D'APPRENTISSAGE

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(history["train_loss"], label="Train Loss", color="blue")
axes[0].plot(history["val_loss"],   label="Val Loss",   color="orange")
axes[0].set_title("R-CNN - Loss par Epoch")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss")
axes[0].legend()
axes[0].grid(True)

axes[1].plot(history["train_acc"], label="Train Accuracy", color="green")
axes[1].plot(history["val_acc"],   label="Val Accuracy",   color="red")
axes[1].set_title("R-CNN - Accuracy par Epoch")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Accuracy (%)")
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.savefig("rcnn_training_curves.png", dpi=150)
plt.show()
print(" Courbes sauvegardées : rcnn_training_curves.png")


# 8. ÉVALUATION SUR TEST SET

print("\n Évaluation sur le Test Set...")

model.load_state_dict(torch.load("rcnn_best_model.pth"))
model.eval()

all_preds, all_labels = [], []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(DEVICE)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())

n
print("\n Rapport de Classification R-CNN :")
print("=" * 60)
print(classification_report(all_labels, all_preds, target_names=CLASS_NAMES))

# Matrice de confusion
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(7, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
plt.title("R-CNN - Matrice de Confusion (Test Set)")
plt.ylabel("Réel")
plt.xlabel("Prédit")
plt.tight_layout()
plt.savefig("rcnn_confusion_matrix.png", dpi=150)
plt.show()
print(" Matrice de confusion sauvegardée : rcnn_confusion_matrix.png")


import json
accuracy = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)
rcnn_results = {
    "model": "R-CNN (ResNet50)",
    "accuracy": round(accuracy * 100, 2),
    "history": history
}
with open("rcnn_results.json", "w") as f:
    json.dump(rcnn_results, f)

print(f"\n R-CNN Test Accuracy : {accuracy*100:.2f}%")
print(" Résultats sauvegardés dans rcnn_results.json")
