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
        // 64b doesn't make sense
        // as the memory bus is saturated enough 
        // when loading 32b
        return uint32_t{};
    } else if constexpr(WordSize % 4 == 0) {
        return uint32_t{};
    } else {
        return char{};
    }
}

template <int WordLen>
__global__ void filter_warp_per_json(const char *text, size_t num_jsons,
                                     char** addresses, bool *out) {
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

  const auto json_start = (wid == 0) ? text : addresses[wid - 1];
  const auto json_end = addresses[wid];

  bool result = true;
  int done = true;

  using CT = decltype(comp_type<WordLen>());
  constexpr int comp_len = sizeof(CT);
  CT f{0}, t{0};
  memcpy((void*)&f, (void*)&config::full_filter[comp_len * lid], comp_len);

  for (auto addr = json_start; addr < json_end - WordLen + 1; ++addr) {
    memcpy((void*)&t, (void*)&addr[comp_len*lid], comp_len);
    result = ((lid >= WordLen / comp_len) || (t == f));
    if constexpr(group_size > 1) {
        warp.sync();
        warp.match_all(result, done);
    } else {
        done = true;
    }
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
    auto operator()(const char* text, size_t num_jsons, char** addresses, bool* out) {
        filter_warp_per_json<WordLen><<<(num_jsons + WarpsPerBlock - 1) / WarpsPerBlock, WarpsPerBlock * 32>>>(text, num_jsons, addresses, out);
    }
};

}
