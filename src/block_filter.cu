//
// Created by placek on 30.05.23.
//

#include <string>
#include <thrust/execution_policy.h>
#include <thrust/device_vector.h>

#include <config.hpp>

#include <block_filter.cuh>
#include <common.cuh>
#include "kmp.cuh"
#include "utils.cuh"

namespace filtering {

//
//    template<int Index>
//    size_t block_filter(std::string &lines) {
//        const auto length = lines.size() + 1;
//        const char *h_text = lines.c_str();
//        char *d_text;
//        bool *d_is_newline;
//        size_t *d_newline_positions;
//        bool *filter_result;
//
//        // malloc
//        CUDA_CHECK(cudaMalloc(&d_text, length * sizeof(char)));
//        CUDA_CHECK(cudaMalloc(&d_is_newline, length * sizeof(bool)));
//
//        // copy
//        CUDA_CHECK(cudaMemcpy(d_text, h_text, length * sizeof(char), cudaMemcpyHostToDevice));
//        CUDA_CHECK(cudaMemset(d_is_newline, 0, length * sizeof(bool)));
//
//        // find newlines
//        auto block_size = (length - 1) / THREAD_SIZE + 1;
//        find_newlines_kernel<<<block_size, THREAD_SIZE>>>(d_text, length, d_is_newline);
//
//        CUDA_KERNEL_FINISH();
//
//        // count newlines
//        const auto num_newlines = thrust::reduce(thrust::device, d_is_newline, d_is_newline + length, 0);
//
//        CUDA_CHECK(cudaMalloc(&d_newline_positions, num_newlines * sizeof(size_t)));
//
//        // find newlines positions
//        thrust::copy_if(thrust::device,
//                        thrust::make_counting_iterator<unsigned long>(0),
//                        thrust::make_counting_iterator(length),
//                        d_is_newline,
//                        d_newline_positions,
//                        thrust::identity<bool>());
//
//        CUDA_CHECK(cudaMalloc(&filter_result, num_newlines * sizeof(bool)));
//        CUDA_CHECK(cudaMemset(filter_result, 0, num_newlines * sizeof(bool)));
//
//        // find largest jsonl size
//        size_t *d_differences;
//        CUDA_CHECK(cudaMalloc(&d_differences, num_newlines * sizeof(size_t)));
//
//        // set first difference as a value of first newline position
//        CUDA_CHECK(cudaMemcpy(d_differences, d_newline_positions, sizeof(size_t), cudaMemcpyDeviceToDevice));
//        thrust::transform(thrust::device,
//                          d_newline_positions + 1,
//                          d_newline_positions + num_newlines,
//                          d_newline_positions,
//                          d_differences + 1,
//                          thrust::minus<size_t>());
//        const auto max_jsonl_size = thrust::reduce(thrust::device, d_differences, d_differences + num_newlines, 0,
//                                                   thrust::maximum<size_t>());
//
//        // free
//        CUDA_CHECK(cudaFree(d_differences));
//
//        // filter
//        block_size = num_newlines;
//        filter_block_per_json_kernel<Index><<<block_size, THREAD_SIZE, max_jsonl_size * sizeof(char) + 1>>>(d_text,
//                                                                                                        num_newlines,
//                                                                                                        d_newline_positions,
//                                                                                                        filter_result);
//
//        CUDA_KERNEL_FINISH();
//
//        // count filtered
//        const auto num_filtered = thrust::reduce(thrust::device, filter_result, filter_result + num_newlines, 0);
//
//        // free
//        CUDA_CHECK(cudaFree(d_text));
//        CUDA_CHECK(cudaFree(d_is_newline));
//        CUDA_CHECK(cudaFree(d_newline_positions));
//        CUDA_CHECK(cudaFree(filter_result));
//
//        return num_filtered;
//    }


}

//template size_t filtering::block_filter<configuration::Index>(std::string &lines);