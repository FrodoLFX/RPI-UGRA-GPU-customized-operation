#include <torch/extension.h>

// CUDA forward declaration
at::Tensor fused_gate_forward_cuda(at::Tensor x, at::Tensor v, double threshold);

torch::Tensor fused_gate_forward(torch::Tensor x, torch::Tensor v, double threshold) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(v.is_cuda(), "v must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32 for this demo");
    TORCH_CHECK(v.dtype() == torch::kFloat32, "v must be float32 for this demo");
    TORCH_CHECK(x.dim() == 2, "x must be 2D [M,N]");
    TORCH_CHECK(v.dim() == 1, "v must be 1D [N]");
    TORCH_CHECK(x.size(1) == v.size(0), "x second dim must match v length");
    return fused_gate_forward_cuda(x, v, threshold);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &fused_gate_forward, "Fused gate forward (CUDA)");
}
