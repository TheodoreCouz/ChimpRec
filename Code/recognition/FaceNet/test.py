import os
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets
from facenet_pytorch import InceptionResnetV1
from itertools import combinations
import random

# Dataset et Transformations
dataset_path = "./ChimpRec-Dataset/Chimpanzee_recognition_dataset"

transform = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

from torchvision import transforms
from PIL import Image
import torch

class TripletDataset(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform
        self.data = []
        self.labels = []
        for img, label in self.dataset:
            self.data.append(img)
            self.labels.append(label)
        self.labels = torch.tensor(self.labels)
        self.classes = torch.unique(self.labels)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        anchor_img = self.data[idx]
        anchor_label = self.labels[idx]

        # Trouver un positif (même classe) et un négatif (classe différente)
        positive_indices = torch.where(self.labels == anchor_label)[0]
        negative_indices = torch.where(self.labels != anchor_label)[0]

        positive_idx = random.choice(positive_indices)
        negative_idx = random.choice(negative_indices)

        positive_img = self.data[positive_idx]
        negative_img = self.data[negative_idx]

        # Si la transformation est définie, appliquez-la uniquement si l'image est un PIL.Image ou un ndarray
        if self.transform:
            # Convertir uniquement si l'image est déjà un tensor, sinon appliquer les transformations
            if isinstance(anchor_img, torch.Tensor):
                anchor_img = anchor_img
            else:
                anchor_img = self.transform(anchor_img)  # Appliquer les transformations

            if isinstance(positive_img, torch.Tensor):
                positive_img = positive_img
            else:
                positive_img = self.transform(positive_img)

            if isinstance(negative_img, torch.Tensor):
                negative_img = negative_img
            else:
                negative_img = self.transform(negative_img)

        return anchor_img, positive_img, negative_img




train_dataset = TripletDataset(
    datasets.ImageFolder(os.path.join(dataset_path, "train"), transform=transform),
    transform=transform
)
val_dataset = TripletDataset(
    datasets.ImageFolder(os.path.join(dataset_path, "val"), transform=transform),
    transform=transform
)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

# Charger le modèle pré-entraîné
facenet = InceptionResnetV1(pretrained='vggface2').eval()

# Débloquer les dernières couches
for param in facenet.parameters():
    param.requires_grad = False
for layer in list(facenet.children())[-5:]:
    for param in layer.parameters():
        param.requires_grad = True

# Déplacer le modèle sur GPU si disponible
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
facenet = facenet.to(device)

# Fonction de perte : TripletMarginLoss
criterion = nn.TripletMarginLoss(margin=1.0)

# Optimiseur
optimizer = optim.Adam(facenet.parameters(), lr=0.001)

# Entraînement avec Triplet Loss
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10):
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print("-----------------------")

        model.train()
        running_loss = 0.0

        for anchor, positive, negative in train_loader:
            anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)

            # Réinitialiser les gradients
            optimizer.zero_grad()

            # Calculer les embeddings
            anchor_output = model(anchor)
            positive_output = model(positive)
            negative_output = model(negative)

            # Calculer la perte
            loss = criterion(anchor_output, positive_output, negative_output)

            # Backpropagation
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)
        print(f"Train Loss: {epoch_loss:.4f}")

        # Phase de validation (facultative)
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for anchor, positive, negative in val_loader:
                anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)

                anchor_output = model(anchor)
                positive_output = model(positive)
                negative_output = model(negative)

                loss = criterion(anchor_output, positive_output, negative_output)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        print(f"Val Loss: {val_loss:.4f}")

# Appel du modèle dans le bloc `if __name__ == '__main__'` pour éviter l'erreur de `multiprocessing`
if __name__ == '__main__':
    train_model(facenet, train_loader, val_loader, criterion, optimizer, num_epochs=10)
