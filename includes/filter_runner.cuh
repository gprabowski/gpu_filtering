#pragma once

#include <iostream>

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

namespace utils {
template<typename Filter>
struct filter_runner {
    auto operator()(const char* text, size_t num_jsons, char** addresses, bool* out) {
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        gpuErrchk(cudaEventRecord(start));
        Filter filter;
        filter(text, num_jsons, addresses, out);
        gpuErrchk(cudaEventRecord(stop));
        gpuErrchk(cudaEventSynchronize(stop));

        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        return milliseconds;
    }
};

template<template<int> class Func, int Val, int End, typename...Args>
void static_iterate(double gb, Args...args) {
  auto t = utils::filter_runner<Func<Val>>()(args...);
  std::cout << Func<Val>::name << " GB/s : " << gb / t << std::endl;
  if constexpr(Val < End) {
    static_iterate<Func, Val + 1, End>(gb, args...);
  }
}

template<template<int> class Filter>
void bench(const std::size_t byte_count, const char* text, size_t num_jsons, char** addresses, bool* out) {
  auto gb = static_cast<double> (byte_count) / (1024 * 1024 * 1024);
  gb = gb * 1000.0;
  static_iterate<Filter, 1, 32>(gb, text, num_jsons, addresses, out);
  return;
}
}
