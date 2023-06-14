#pragma once

namespace config {
constexpr int DNF = 4;
constexpr int RF = 4;
static __device__ constexpr char full_filter[] =
    // Pattern with high occurence for small filters (high bandwidth)
    //"2489651045\",\"type\":\"CreateEvent";

    // pattern with 0 occurence -> tests actual bandwidth
    "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx";

} // namespace config
