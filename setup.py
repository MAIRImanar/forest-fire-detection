#  APPROCHE 1 : R-CNN (Classification) + YOLOv11 (Détection)
#  Projet : AI-Based Forest Fire Detection
#  Auteur : MANAR mairi

# Installation des bibliothèques nécessaires
import subprocess
import sys

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package, "-q"])

print("📦 Installation des bibliothèques...")

install("torch")
install("torchvision")
install("ultralytics")       # YOLOv11
install("opencv-python")
install("matplotlib")
install("scikit-learn")
install("seaborn")
install("Pillow")
install("pandas")
install("numpy")
install("tqdm")

print("✅ Toutes les bibliothèques sont installées !")

# Vérification GPU
import torch
print(f"\n🖥️  Device disponible : {'GPU ✅ ' + torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU ⚠️ (lent)'}")
print(f"📌 PyTorch version : {torch.__version__}")
