#include <algorithm.cuh>

#include <thrust/reduce.h>

#include <config.hpp>
#include <filter_runner.cuh>
#include <warp_filter.cuh>

namespace filtering {

namespace cg = cooperative_groups;

#define gpuErrchk(ans)                                                         \
  { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line,
                      bool abort = true) {
  if (code != cudaSuccess) {
    fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file,
            line);
    if (abort)
      exit(code);
  }
}

__global__ void find_newlines(const char *text, const size_t size,
                              bool *is_newline) {
  auto grid = cg::this_grid();
  const size_t grid_size = grid.size();
  for (auto tid = grid.thread_rank(); tid < size; tid += grid_size) {
    is_newline[tid] = (text[tid] == '\n');
  }
}

// currently ASCII assumption
size_t filter(std::string &lines) {
  const auto byte_count = lines.size();
  const auto len = lines.size() + 1;
  const char *h_text = lines.c_str();
  char *d_text;
  bool *d_is_newline, *d_is_valid;
  size_t *d_indices;

  cudaMalloc(&d_text, len);
  cudaMemcpy(d_text, h_text, len, cudaMemcpyHostToDevice);
  cudaMalloc(&d_is_newline, len * sizeof(char));

  find_newlines<<<4096, 1024>>>(d_text, len, d_is_newline);

  const auto json_count =
      thrust::reduce(thrust::device, d_is_newline, d_is_newline + len, 0);

  cudaMalloc(&d_indices, json_count * sizeof(size_t));
  cudaMalloc(&d_is_valid, json_count * sizeof(bool));

  thrust::copy_if(thrust::device,
                  thrust::make_counting_iterator(static_cast<decltype(len)>(0)),
                  thrust::make_counting_iterator(len), d_is_newline, d_indices,
                  thrust::identity<bool>());

  std::cout << "JSON COUNT: " << json_count << std::endl;

  utils::bench<filter::warp_filter>(byte_count, d_text, json_count, d_indices,
                                    d_is_valid);

  const auto correct_count =
      thrust::reduce(thrust::device, d_is_valid, d_is_valid + json_count, 0);
  std::cout << " VALID: " << correct_count << std::endl;
  cudaFree(d_indices);
  cudaFree(d_text);
  cudaFree(d_is_newline);
  return 0;
}

} // namespace filtering
