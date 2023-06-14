#pragma once

#include <cooperative_groups.h>

#include <config.hpp>

namespace cg = cooperative_groups;

namespace filter {

__host__ __device__ unsigned int constexpr upp2(unsigned int v) {
  v--;
  v |= v >> 1;
  v |= v >> 2;
  v |= v >> 4;
  v |= v >> 8;
  v |= v >> 16;
  v++;

  return v;
}

template<int WordSize>
__device__ __host__ __forceinline__ constexpr int find_group_size() {
  constexpr int up2v = upp2(WordSize);
  if constexpr(WordSize % 8 == 0) { 
    // double doesn't make sense
    // as the memory bus is saturated enough 
    // when loading floats
    return up2v / 4;
  } else if constexpr(WordSize % 4 == 0) {
    return up2v / 4;
  }
  return up2v;
}

template<int WordSize>
__device__ __host__ constexpr auto comp_type() {
    if constexpr(WordSize % 8 == 0) {
        // double doesn't make sense
        // as the memory bus is saturated enough 
        // when loading floats
        return float{};
    } else if constexpr(WordSize % 4 == 0) {
        return float{};
    } else {
        return char{};
    }
}

template <int WordLen>
__global__ void filter_warp_per_json(const char *text, size_t num_jsons,
                                     size_t *delimiters, bool *out) {
  constexpr auto FilterWord = config::full_filter;
  constexpr auto group_size = find_group_size<WordLen>();

  const auto grid = cg::this_grid();
  const size_t grid_size = grid.size();
  const auto tid = grid.thread_rank();
  const auto warp = cg::tiled_partition<group_size>(cg::this_thread_block());
  const auto lid = warp.thread_rank();

  const auto wid = tid / group_size;
  if (wid >= num_jsons) {
    return;
  }

  const auto json_start = (wid == 0) ? 0 : delimiters[wid - 1];
  const auto json_end = delimiters[wid];

  bool result = true;
  int done;

  using CT = decltype(comp_type<WordLen>());
  constexpr int comp_len = sizeof(CT);
  CT f{0}, t{0};
  memcpy((void*)&f, (void*)&config::full_filter[lid], comp_len);

  for (size_t i = json_start; i < json_end - WordLen + 1; ++i) {
    memcpy((void*)&t, (void*)&text[i + lid], comp_len);
    result = ((lid >= WordLen) || (t == f));
    warp.sync();
    warp.match_all(result, done);
    if (done && result) {
      out[wid] = true;
      return;
    }
  }
  out[wid] = false;
}

template<int WordLen>
struct warp_filter {
    constexpr static char name[] = "Cooperative Group Filter";
    int WarpsPerBlock = 8;
    auto operator()(const char* text, size_t num_jsons, size_t *delimiters, bool* out) {
        filter_warp_per_json<WordLen><<<(num_jsons + WarpsPerBlock - 1) / WarpsPerBlock, WarpsPerBlock * 32>>>(text, num_jsons, delimiters, out);
    }
};

}
