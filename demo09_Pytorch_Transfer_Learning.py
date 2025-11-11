import torch
import torch.nn as nn
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader

# Load pre‑trained ResNet18
model = models.resnet18(pretrained=True)

# Freeze all layers except the final fully connected layer
for param in model.parameters():
    param.requires_grad = False

# Replace the final layer to classify 2 classes (e.g., cats vs dogs)
model.fc = nn.Linear(model.fc.in_features, 2)

# Define loss function and optimizer (only train the new layer)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.fc.parameters(), lr=0.001)

# Assume we have a dataset of two classes in directory structure
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
train_dataset = datasets.ImageFolder('data/train', transform=transform)
train_loader  = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Training loop (single epoch for brevity)
model.train()
for images, labels in train_loader:
    optimizer.zero_grad()
    outputs = model(images)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()

print("Finished transfer learning training on custom dataset")
