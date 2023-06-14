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
    auto operator()(const char* text, size_t num_jsons, size_t *delimiters, bool* out) {
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        gpuErrchk(cudaEventRecord(start));
        Filter filter;
        filter(text, num_jsons, delimiters, out);
        gpuErrchk(cudaEventRecord(stop));
        gpuErrchk(cudaEventSynchronize(stop));

        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        return milliseconds;
    }
};

template<template<int> class Filter>
void bench(const std::size_t byte_count, const char* text, size_t num_jsons, size_t *delimiters, bool* out) {
  auto gb = static_cast<double> (byte_count) / (1024 * 1024 * 1024);
  gb = gb * 1000.0;
  auto t = utils::filter_runner<Filter<1>>()(text, num_jsons, delimiters, out);
  std::cout << Filter<1>::name << " GB/s : " << gb / t << std::endl;
  t = utils::filter_runner<Filter<2>>()(text, num_jsons, delimiters, out);
  std::cout << Filter<2>::name << " GB/s : " << gb / t << std::endl;
  t = utils::filter_runner<Filter<3>>()(text, num_jsons, delimiters, out);
  std::cout << Filter<3>::name << " GB/s : " << gb / t << std::endl;
  t = utils::filter_runner<Filter<4>>()(text, num_jsons, delimiters, out);
  std::cout << Filter<4>::name << " GB/s : " << gb / t << std::endl;
  t = utils::filter_runner<Filter<5>>()(text, num_jsons, delimiters, out);
  std::cout << Filter<5>::name << " GB/s : " << gb / t << std::endl;
  t = utils::filter_runner<Filter<6>>()(text, num_jsons, delimiters, out);
  std::cout << Filter<6>::name << " GB/s : " << gb / t << std::endl;
  t = utils::filter_runner<Filter<7>>()(text, num_jsons, delimiters, out);
  std::cout << Filter<7>::name << " GB/s : " << gb / t << std::endl;
  t = utils::filter_runner<Filter<8>>()(text, num_jsons, delimiters, out);
  std::cout << Filter<8>::name << " GB/s : " << gb / t << std::endl;
  t = utils::filter_runner<Filter<9>>()(text, num_jsons, delimiters, out);
  std::cout << Filter<9>::name << " GB/s : " << gb / t << std::endl;
  t = utils::filter_runner<Filter<10>>()(text, num_jsons, delimiters, out);
  std::cout << Filter<10>::name << " GB/s : " << gb / t << std::endl;
  t = utils::filter_runner<Filter<11>>()(text, num_jsons, delimiters, out);
  std::cout << Filter<11>::name << " GB/s : " << gb / t << std::endl;
  t = utils::filter_runner<Filter<12>>()(text, num_jsons, delimiters, out);
  std::cout << Filter<12>::name << " GB/s : " << gb / t << std::endl;
  t = utils::filter_runner<Filter<13>>()(text, num_jsons, delimiters, out);
  std::cout << Filter<13>::name << " GB/s : " << gb / t << std::endl;
  t = utils::filter_runner<Filter<14>>()(text, num_jsons, delimiters, out);
  std::cout << Filter<14>::name << " GB/s : " << gb / t << std::endl;
  t = utils::filter_runner<Filter<15>>()(text, num_jsons, delimiters, out);
  std::cout << Filter<15>::name << " GB/s : " << gb / t << std::endl;
  t = utils::filter_runner<Filter<16>>()(text, num_jsons, delimiters, out);
  std::cout << Filter<16>::name << " GB/s : " << gb / t << std::endl;
  t = utils::filter_runner<Filter<17>>()(text, num_jsons, delimiters, out);
  std::cout << Filter<17>::name << " GB/s : " << gb / t << std::endl;
  t = utils::filter_runner<Filter<18>>()(text, num_jsons, delimiters, out);
  std::cout << Filter<18>::name << " GB/s : " << gb / t << std::endl;
  t = utils::filter_runner<Filter<19>>()(text, num_jsons, delimiters, out);
  std::cout << Filter<19>::name << " GB/s : " << gb / t << std::endl;
  t = utils::filter_runner<Filter<20>>()(text, num_jsons, delimiters, out);
  std::cout << Filter<20>::name << " GB/s : " << gb / t << std::endl;
  t = utils::filter_runner<Filter<21>>()(text, num_jsons, delimiters, out);
  std::cout << Filter<21>::name << " GB/s : " << gb / t << std::endl;
  t = utils::filter_runner<Filter<22>>()(text, num_jsons, delimiters, out);
  std::cout << Filter<22>::name << " GB/s : " << gb / t << std::endl;
  t = utils::filter_runner<Filter<23>>()(text, num_jsons, delimiters, out);
  std::cout << Filter<23>::name << " GB/s : " << gb / t << std::endl;
  t = utils::filter_runner<Filter<24>>()(text, num_jsons, delimiters, out);
  std::cout << Filter<24>::name << " GB/s : " << gb / t << std::endl;
  t = utils::filter_runner<Filter<25>>()(text, num_jsons, delimiters, out);
  std::cout << Filter<25>::name << " GB/s : " << gb / t << std::endl;
  t = utils::filter_runner<Filter<26>>()(text, num_jsons, delimiters, out);
  std::cout << Filter<26>::name << " GB/s : " << gb / t << std::endl;
  t = utils::filter_runner<Filter<27>>()(text, num_jsons, delimiters, out);
  std::cout << Filter<27>::name << " GB/s : " << gb / t << std::endl;
  t = utils::filter_runner<Filter<28>>()(text, num_jsons, delimiters, out);
  std::cout << Filter<28>::name << " GB/s : " << gb / t << std::endl;
  t = utils::filter_runner<Filter<29>>()(text, num_jsons, delimiters, out);
  std::cout << Filter<29>::name << " GB/s : " << gb / t << std::endl;
  t = utils::filter_runner<Filter<30>>()(text, num_jsons, delimiters, out);
  std::cout << Filter<30>::name << " GB/s : " << gb / t << std::endl;
  t = utils::filter_runner<Filter<31>>()(text, num_jsons, delimiters, out);
  std::cout << Filter<31>::name << " GB/s : " << gb / t << std::endl;
  t = utils::filter_runner<Filter<32>>()(text, num_jsons, delimiters, out);
  std::cout << Filter<32>::name << " GB/s : " << gb / t << std::endl;
  return;
}
}
