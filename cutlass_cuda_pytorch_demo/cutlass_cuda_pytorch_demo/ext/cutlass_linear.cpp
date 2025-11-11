#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>

// from .cu
extern "C" cutlass::Status cutlass_gemm_fp32_rowmajor(
    const float* A, const float* B, float* C,
    int M, int N, int K,
    cudaStream_t stream);

namespace {

torch::Tensor cutlass_linear_impl(torch::Tensor A, torch::Tensor B,
                                  c10::optional<torch::Tensor> bias_opt,
                                  bool relu) {
  TORCH_CHECK(A.is_cuda() && B.is_cuda(), "A and B must be CUDA tensors");
  TORCH_CHECK(A.scalar_type() == at::kFloat && B.scalar_type() == at::kFloat,
              "This demo expects float32 tensors (got ", A.scalar_type(), " and ", B.scalar_type(), ")");
  TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "A and B must be 2D");
  TORCH_CHECK(A.size(1) == B.size(0), "Inner dim mismatch: A(M,K) x B(K,N)");

  auto M = static_cast<int>(A.size(0));
  auto K = static_cast<int>(A.size(1));
  auto N = static_cast<int>(B.size(1));

  auto A_c = A.contiguous();
  auto B_c = B.contiguous();
  auto C   = torch::empty({M, N}, A.options());

  // Use current PyTorch CUDA stream
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  auto status = cutlass_gemm_fp32_rowmajor(
      A_c.data_ptr<float>(),
      B_c.data_ptr<float>(),
      C.data_ptr<float>(),
      M, N, K,
      stream
  );
  TORCH_CHECK(status == cutlass::Status::kSuccess, "CUTLASS GEMM failed with status=", int(status));

  if (bias_opt.has_value()) {
    auto bias = bias_opt.value();
    TORCH_CHECK(bias.is_cuda(), "bias must be CUDA tensor");
    TORCH_CHECK(bias.scalar_type() == at::kFloat, "bias must be float32");
    TORCH_CHECK(bias.dim() == 1 && bias.size(0) == N, "bias must be shape (N,)");

    // Add bias row-wise: C[i, j] += bias[j]
    C.add_(bias.view({1, N}));
  }

  if (relu) {
    C.clamp_min_(0);
  }

  return C;
}

} // namespace

TORCH_LIBRARY(cutlass_ext, m) {
  m.def("linear(Tensor A, Tensor B, Tensor? bias=None, bool relu=False) -> Tensor");
}

TORCH_LIBRARY_IMPL(cutlass_ext, CUDA, m) {
  m.impl("linear", cutlass_linear_impl);
}
