import torch
from torch.utils.data import Dataset, DataLoader

# Create a custom dataset for y = 3x + 1 + noise
class RegressionDataset(Dataset):
    def __init__(self, n_samples=1000):
        self.x = torch.randn(n_samples, 1)
        self.y = 3 * self.x + 1 + 0.3 * torch.randn(n_samples, 1)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

# Instantiate dataset and dataloader
dataset = RegressionDataset()
loader = DataLoader(dataset, batch_size=32, shuffle=True)

# Simple linear regression model
model = torch.nn.Linear(1, 1)
criterion = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.05)

# Training using mini‑batches
for epoch in range(5):
    for batch_x, batch_y in loader:
        preds = model(batch_x)
        loss = criterion(preds, batch_y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

print(f"Learned weight: {model.weight.item():.2f}, bias: {model.bias.item():.2f}")
