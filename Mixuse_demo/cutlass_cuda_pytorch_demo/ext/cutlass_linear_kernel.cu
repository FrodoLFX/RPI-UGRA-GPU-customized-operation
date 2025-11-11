#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm.h>
#include <cuda_runtime.h>

extern "C" cutlass::Status cutlass_gemm_fp32_rowmajor(
    const float* A, const float* B, float* C,
    int M, int N, int K,
    cudaStream_t stream) {

  using ElementInputA = float;
  using ElementInputB = float;
  using ElementOutput = float;
  using ElementAccumulator = float;
  using Layout = cutlass::layout::RowMajor;

  using Gemm = cutlass::gemm::device::Gemm<
      ElementInputA, Layout,
      ElementInputB, Layout,
      ElementOutput, Layout,
      ElementAccumulator>;

  float alpha = 1.0f;
  float beta  = 0.0f;

  typename Gemm::Arguments args(
      {M, N, K},
      {A, K},          // A: (M,K) row-major, lda = K
      {B, N},          // B: (K,N) row-major, ldb = N
      {C, N},          // C: (M,N) row-major, ldc = N
      {C, N},          // D: output (in-place on C), ldd = N
      {alpha, beta}
  );

  Gemm gemm_op;

  // Allocate the workspace if needed
  size_t workspace_bytes = Gemm::get_workspace_size(args);
  void* workspace = nullptr;
  if (workspace_bytes > 0) {
    cudaError_t err = cudaMalloc(&workspace, workspace_bytes);
    if (err != cudaSuccess) {
      return cutlass::Status::kErrorWorkspaceNull;
    }
  }

  // Initialize & run
  cutlass::Status status = gemm_op.initialize(args, workspace);
  if (status != cutlass::Status::kSuccess) {
    if (workspace) cudaFree(workspace);
    return status;
  }

  status = gemm_op.run(stream);

  if (workspace) cudaFree(workspace);
  return status;
}
