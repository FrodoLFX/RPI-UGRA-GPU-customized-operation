#include <torch/extension.h>

at::Tensor matmul_forward_cuda(at::Tensor A, at::Tensor B);
at::Tensor gate_forward_cuda(at::Tensor C, at::Tensor v, double threshold);
at::Tensor fused_forward_cuda(at::Tensor A, at::Tensor B, at::Tensor v, double threshold);

static void check_inputs_matmul(const at::Tensor& A, const at::Tensor& B) {
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "A and B must be CUDA tensors");
    TORCH_CHECK(A.dtype() == at::kFloat && B.dtype() == at::kFloat, "float32 only for Phase 2");
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "A and B must be 2D");
    TORCH_CHECK(A.is_contiguous() && B.is_contiguous(), "A and B must be contiguous");
    TORCH_CHECK(A.size(1) == B.size(0), "A is [M,K], B is [K,N]");
}

static void check_inputs_gate(const at::Tensor& C, const at::Tensor& v) {
    TORCH_CHECK(C.is_cuda() && v.is_cuda(), "C and v must be CUDA tensors");
    TORCH_CHECK(C.dtype() == at::kFloat && v.dtype() == at::kFloat, "float32 only for Phase 2");
    TORCH_CHECK(C.dim() == 2, "C must be 2D [M,N]");
    TORCH_CHECK(v.dim() == 1, "v must be 1D [N]");
    TORCH_CHECK(C.is_contiguous() && v.is_contiguous(), "C and v must be contiguous");
    TORCH_CHECK(C.size(1) == v.size(0), "C second dim must match v length");
}

at::Tensor matmul_forward(at::Tensor A, at::Tensor B) {
    check_inputs_matmul(A, B);
    return matmul_forward_cuda(A, B);
}

at::Tensor gate_forward(at::Tensor C, at::Tensor v, double threshold) {
    check_inputs_gate(C, v);
    return gate_forward_cuda(C, v, threshold);
}

at::Tensor fused_forward(at::Tensor A, at::Tensor B, at::Tensor v, double threshold) {
    check_inputs_matmul(A, B);
    TORCH_CHECK(v.is_cuda() && v.dtype() == at::kFloat && v.is_contiguous(), "v must be CUDA contiguous float32");
    TORCH_CHECK(v.dim() == 1 && v.size(0) == B.size(1), "v must be [N] where B is [K,N]");
    return fused_forward_cuda(A, B, v, threshold);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("matmul", &matmul_forward, "Tiled matmul (CUDA)");
    m.def("gate", &gate_forward, "Gate+threshold (CUDA)");
    m.def("fused", &fused_forward, "Fused matmul+gate (CUDA)");
}
