import torch

# Create tensors with gradient tracking
a = torch.tensor([2.0, 3.0], requires_grad=True)
b = torch.tensor([1.0, 4.0], requires_grad=True)

# Compute a simple function: f = (a * b).sum()
prod = a * b        # elementwise product
f = prod.sum()      # scalar output

# Compute gradients
df_da, df_db = torch.autograd.grad(f, [a, b])

print(f"Gradient df/da: {df_da}")
print(f"Gradient df/db: {df_db}")
