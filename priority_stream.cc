#include <chrono>
#include <thread>

#include <glog/logging.h>

#include "priority_stream.h"

#define CUDA_CHECK(condition) \
  /* Code block avoids redefinition of cudaError_t error */ \
  do { \
    cudaError_t error = condition; \
    CHECK_EQ(error, cudaSuccess) << " " << cudaGetErrorString(error); \
  } while (0)

#define CUBLAS_CHECK(condition) \
  do { \
    cublasStatus_t status = condition; \
    CHECK_EQ(status, CUBLAS_STATUS_SUCCESS) << status; \
  } while (0)

PriorityStreamSimulator::PriorityStreamSimulator(
    const int priority,
    const int size,
    const std::string& name ) : priority_(priority), size_(size), name_(name) {
  // Create the cuda stream.
  static int PRI_LOW = 0;
  static int PRI_HIGH = 0;
  CUDA_CHECK(cudaDeviceGetStreamPriorityRange(&PRI_LOW, &PRI_HIGH));
  // LOG(INFO) << "Lowest priority: " << PRI_LOW;
  // LOG(INFO) << "Highest priority:" << PRI_HIGH;
  CHECK_GE(priority, PRI_HIGH);
  CHECK_LE(priority, PRI_LOW);

  CUDA_CHECK(cudaStreamCreateWithPriority(&stream_, cudaStreamNonBlocking, priority));

  // Create the working memories;
  CUDA_CHECK(cudaMalloc(&A_, sizeof(float) * size_ * size_));
  CUDA_CHECK(cudaMalloc(&B_, sizeof(float) * size_ * size_));
  CUDA_CHECK(cudaMalloc(&C_, sizeof(float) * size_ * size_));

  // Create the cublas handle
  CUBLAS_CHECK(cublasCreate(&handle_));
  CUBLAS_CHECK(cublasSetStream(handle_, stream_));
}

PriorityStreamSimulator::~PriorityStreamSimulator() {
  CUDA_CHECK(cudaFree(A_));
  CUDA_CHECK(cudaFree(B_));
  CUDA_CHECK(cudaFree(C_));
  CUBLAS_CHECK(cublasDestroy(handle_));
  CUDA_CHECK(cudaStreamDestroy(stream_));
}

void PriorityStreamSimulator::Run(
    const int num_iters, const bool async, const int sleep_ms) {
  const float FLOAT_ONE = 1.0f;
  const float FLOAT_ZERO = 0.0f;
  for (int i = 0; i < num_iters; ++i) {
    CUBLAS_CHECK(cublasSgemm(
      handle_,
      CUBLAS_OP_N,
      CUBLAS_OP_N,
      size_,
      size_,
      size_,
      &FLOAT_ONE,
      A_,
      size_,
      B_,
      size_,
      &FLOAT_ZERO,
      C_,
      size_));
    if (!async) {
      CUDA_CHECK(cudaStreamSynchronize(stream_));
    }
    if (sleep_ms) {
      std::this_thread::sleep_for(std::chrono::milliseconds(sleep_ms));
    }
  }
}

void PriorityStreamSimulator::RunAndLog(
    const int num_iters, const bool async, const int sleep_ms) {
  auto start = std::chrono::steady_clock::now();
  Run(num_iters, async, sleep_ms);
  CUDA_CHECK(cudaStreamSynchronize(stream_));
  auto end = std::chrono::steady_clock::now();
  LOG(INFO) << name_ << " running average in nanoseconds: "
       << std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count() / num_iters
       << " ns";
}
