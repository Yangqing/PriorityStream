#pragma once

#include <cuda_runtime.h>
#include <cublas_v2.h>

class PriorityStreamSimulator {
 public:
  PriorityStreamSimulator(const int priority, const int size, const std::string& name);
  ~PriorityStreamSimulator();

  void Run(const int num_iters, const bool async, const int sleep_ms);

  void RunAndLog(const int num_iters, const bool async, const int sleep_ms);

 private:
  int priority_;
  unsigned int size_;
  std::string name_;

  // I know, raw pointers are bad...
  float* A_;
  float* B_;
  float* C_;
  cudaStream_t stream_;
  cublasHandle_t handle_;
};