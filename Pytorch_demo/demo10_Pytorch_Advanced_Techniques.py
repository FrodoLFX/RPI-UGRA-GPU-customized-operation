import torch
from torch.autograd import Function

# Custom autograd function for x^3
class CubeFunction(Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input ** 3

    @staticmethod
    def backward(ctx, grad_output):
        (input,) = ctx.saved_tensors
        grad_input = 3 * (input ** 2) * grad_output
        return grad_input

# Use the custom function
tensor = torch.tensor([2.0, 3.0], requires_grad=True)
output = CubeFunction.apply(tensor)
output.sum().backward()
print(f"Gradient of x^3 at [2,3]: {tensor.grad}")

# Define a small module and convert to TorchScript
class MyModule(torch.nn.Module):
    def forward(self, x):
        return x ** 2 + 1

scripted_module = torch.jit.script(MyModule())
print(scripted_module.code)  # show TorchScript code

# Save and load the scripted module
scripted_module.save('mymodule.pt')
loaded = torch.jit.load('mymodule.pt')
print("TorchScript module output for input 5:", loaded(torch.tensor(5)))
