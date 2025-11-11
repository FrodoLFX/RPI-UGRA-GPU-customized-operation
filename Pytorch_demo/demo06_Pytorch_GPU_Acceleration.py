import torch
import torch.nn as nn

# Check for CUDA availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Simple network
data_dim = 100
hidden_dim = 50
num_classes = 10

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(data_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

model = Net().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

# Dummy data
X = torch.randn(1024, data_dim).to(device)
y = torch.randint(0, num_classes, (1024,), dtype=torch.long).to(device)

# Forward and backward pass on GPU
logits = model(X)
loss = criterion(logits, y)

optimizer.zero_grad()
loss.backward()
optimizer.step()

print(f"Loss after one update: {loss.item():.4f}")
