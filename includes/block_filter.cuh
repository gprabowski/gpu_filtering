//
// Created by placek on 30.05.23.
//

#ifndef JSON_FILTERING_BLOCK_FILTER_CUH
#define JSON_FILTERING_BLOCK_FILTER_CUH

#include <string>

#include <kmp.cuh>

namespace filtering {
//    template<int Index>
//    size_t block_filter(std::string &lines);

    constexpr int THREAD_SIZE = 1024;

    template<int Index, int WordLen>
    __global__ void filter_block_per_json_kernel(const char *text, size_t num_of_jsons, char **addresses,
                                                 bool *filter_result) {
        extern __shared__ char shared_json[];

        const auto json_start = blockIdx.x == 0 ? text : addresses[blockIdx.x - 1] + 1;
        const auto json_end = addresses[blockIdx.x];

        const auto json_length = json_end - json_start;

        const auto json_chunk_start = json_start + threadIdx.x * json_length / blockDim.x;
        auto json_chunk_end = json_start + (threadIdx.x + 1) * json_length / blockDim.x;
        json_chunk_end = json_chunk_end > json_end ? json_end : json_chunk_end;
        const auto json_chunk_length = json_chunk_end - json_chunk_start;

        // copy json chunk to shared memory
        for (auto ch = json_chunk_start; ch < json_chunk_end; ch++) {
            shared_json[ch - json_start] = *ch;
        }

        const auto shared_chunk_start = shared_json + (json_chunk_start - json_start);

        // check if pattern starts inside json chunk
        if (threadIdx.x == 0) {
            filter_result[blockIdx.x] = false;
        }

        if (is_pattern_present_in_text<Index, WordLen>(shared_chunk_start, json_chunk_length + WordLen - 1)) {
            filter_result[blockIdx.x] = true;
            return;
        }
    }

    template<int WordLength>
    struct block_filter {
        constexpr static char name[] = "Block filter";

        auto operator()(const char *text, size_t num_jsons, char **addresses, bool *out) {

            CUDA_CHECK(cudaFuncSetAttribute(filter_block_per_json_kernel<0, WordLength>,
                                            cudaFuncAttributeMaxDynamicSharedMemorySize,
                                            60 * 1024));

            filter_block_per_json_kernel<0, WordLength><<<num_jsons, THREAD_SIZE, 60 * 1024>>>(
                    text,
                    num_jsons,
                    addresses,
                    out);

            CUDA_KERNEL_FINISH();
        }
    };
}

#endif //JSON_FILTERING_BLOCK_FILTER_CUH
