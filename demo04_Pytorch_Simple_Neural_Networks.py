import torch
import torch.nn as nn

# Synthetic two‑class dataset
X = torch.randn(200, 2)
y = (X[:, 0] * X[:, 1] > 0).long()  # class 1 if product > 0, else class 0

# Define a simple feedforward network
class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 16)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(16, 2)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        return self.fc2(x)

model = SimpleNet()

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Training loop
for epoch in range(200):
    logits = model(X)
    loss = criterion(logits, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Evaluate accuracy
with torch.no_grad():
    predictions = model(X).argmax(dim=1)
    accuracy = (predictions == y).float().mean()
    print(f"Training accuracy: {accuracy.item():.2f}")
