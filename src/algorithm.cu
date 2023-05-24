#include <algorithm.cuh>
#include <cooperative_groups.h>
#include <thrust/reduce.h>

namespace filtering {

__device__ const char d_string1[] = "first";

template <int Index> __device__ constexpr const char *get_word() {
  return d_string1;
}
template <int Index> __device__ constexpr size_t get_len() { return 5; }

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

template <int WordIndex>
__global__ void filter_warp_per_json(const char *text, size_t num_jsons,
                                     size_t *delimiters, bool *out) {
  const auto FilterWord = get_word<WordIndex>();
  const auto word_len = get_len<WordIndex>();

  const auto grid = cg::this_grid();
  const size_t grid_size = grid.size();
  const auto tid = grid.thread_rank();
  const auto warp = cg::tiled_partition<32>(cg::this_thread_block());
  const auto lid = warp.thread_rank();

  const auto wid = tid / 32;
  if (wid >= num_jsons) {
    return;
  }

  out[wid] = false;
  const auto json_start = (wid == 0) ? 0 : delimiters[wid - 1];
  const auto json_end = delimiters[wid];

  bool result = true;
  int done;

  for (size_t i = json_start; i < json_end - word_len + 1; ++i) {
    result = (lid >= word_len) || (text[i + lid] == FilterWord[lid]);
    warp.sync();
    warp.match_all(result, done);
    if (done) {
      out[wid] = true;
      return;
    }
  }
}

// currently ASCII assumption
size_t filter(std::string &lines) {
  const auto len = lines.size() + 1;
  const char *h_text = lines.c_str();
  char *d_text;
  bool *d_is_newline, *d_is_valid;
  size_t *d_indices;
  cudaMalloc(&d_text, len);
  cudaMemcpy(d_text, h_text, len, cudaMemcpyHostToDevice);
  cudaMalloc(&d_is_newline, len * sizeof(bool));
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

  filter_warp_per_json<1><<<(json_count - 1) / 4 + 1, 128>>>(
      d_text, json_count, d_indices, d_is_valid);
  const auto correct_count =
      thrust::reduce(thrust::device, d_is_valid, d_is_valid + json_count, 0);
  std::cout << " VALID: " << correct_count << std::endl;
  cudaFree(d_indices);
  cudaFree(d_text);
  cudaFree(d_is_newline);
  return 0;
}

} // namespace filtering
