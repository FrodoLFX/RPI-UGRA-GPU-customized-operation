import torch

# Create tensors of different dimensions
scalar   = torch.tensor(3.14)              # 0‑D tensor (scalar)
vector   = torch.tensor([1.0, 2.0, 3.0])   # 1‑D tensor (vector)
matrix   = torch.tensor([[1, 2], [3, 4]])  # 2‑D tensor (matrix)
three_d  = torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])  # 3‑D

# Basic arithmetic
sum_vec = vector + torch.tensor([10., 20., 30.])  # elementwise addition
prod_mat = matrix @ matrix                        # matrix multiplication

# Reshaping
a = torch.arange(12)       # 1D tensor with values 0–11
b = a.reshape(3, 4)        # reshape to 3×4 matrix

# GPU support (if CUDA available)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
vector_gpu = vector.to(device)  # move tensor to GPU
print(f"Vector on device: {device}, values: {vector_gpu}")
