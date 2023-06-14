//
// Created by placek on 24.05.23.
//

#ifndef JSON_FILTERING_ALGORITHM_CUH
#define JSON_FILTERING_ALGORITHM_CUH

#include "utils.cuh"
#include "kmp.cuh"

namespace filtering {
//    template<int Index>
//    size_t thread_filter(std::string &lines);

    template<int Index, int WordLen>
    __global__ void filter_thread_per_json_kernel(const char *text, size_t num_of_jsons, char** addresses,
                                                  bool *filter_result) {
        const auto tid = TID;
        if (tid >= num_of_jsons) {
            return;
        }

        const auto json_start = tid == 0 ? text : addresses[tid - 1] + 1;
        const auto json_end = addresses[tid];

        const auto json_length = json_end - json_start;

        filter_result[tid] = is_pattern_present_in_text<Index, WordLen>(json_start, json_length);
    }

    template<int WordLength>
    struct thread_filter {
        constexpr static char name[] = "Thread Filter";

        constexpr static int ThreadSize = 1024;

        auto operator()(const char *text, size_t num_jsons, char **addresses, bool *out) {
            filter_thread_per_json_kernel<1, WordLength><<<(num_jsons - 1) / ThreadSize + 1, ThreadSize>>>(text, num_jsons, addresses, out);
        }

    };

}

#endif //JSON_FILTERING_ALGORITHM_CUH
