import torch
import torch.nn as nn

# Define and train a simple model
model = nn.Linear(2, 1)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# Dummy training
for epoch in range(10):
    input_data = torch.randn(5, 2)
    target = torch.randn(5, 1)
    output = model(input_data)
    loss = ((output - target) ** 2).mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Save the model's state_dict
path = 'linear_model.pt'
torch.save(model.state_dict(), path)

# Load into a new model instance
new_model = nn.Linear(2, 1)
new_model.load_state_dict(torch.load(path))
new_model.eval()  # set to evaluation mode

# Test that the loaded model produces same output
with torch.no_grad():
    test_input = torch.tensor([[0.5, -1.0]])
    print("Original model output:", model(test_input))
    print("Loaded model output:", new_model(test_input))
