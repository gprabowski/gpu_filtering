//
// Created by placek on 30.05.23.
//

#include <thrust/reduce.h>
#include <thrust/device_vector.h>

#include <common.cuh>
#include "kmp.cuh"
#include "utils.cuh"


namespace filtering {

    constexpr int THREAD_SIZE = 1024;




//    template<int Index>
//    size_t thread_filter(std::string &lines) {
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
//
//        // filter
//        block_size = (num_newlines - 1) / THREAD_SIZE + 1;
//        filter_thread_per_json_kernel<Index><<<block_size, THREAD_SIZE>>>(d_text, num_newlines,
//                                                                          d_newline_positions,
//                                                                          filter_result);
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

//// instantiate template
//template size_t filtering::thread_filter<configuration::Index>(std::string &lines);