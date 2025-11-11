import torch

# Generate synthetic data: y = 2x + 3 + noise
X = torch.arange(0, 10, dtype=torch.float32)
y_true = 2 * X + 3 + torch.randn(X.size()) * 0.5

# Initialize parameters with gradients
a = torch.randn(1, requires_grad=True)  # slope
b = torch.randn(1, requires_grad=True)  # intercept

learning_rate = 0.01

for epoch in range(500):
    y_pred = a * X + b
    loss = ((y_pred - y_true) ** 2).mean()  # mean squared error
    
    # Compute gradients
    loss.backward()

    # Update parameters (gradient descent)
    with torch.no_grad():
        a -= learning_rate * a.grad
        b -= learning_rate * b.grad
        
        # Zero the gradients for next iteration
        a.grad.zero_()
        b.grad.zero_()

# Print learned parameters
print(f"Learned slope: {a.item():.2f}, intercept: {b.item():.2f}")
