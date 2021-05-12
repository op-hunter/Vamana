//
// Created by cmli on 21-4-30.
//

#pragma once
#include <iostream>

namespace Vamana {

float L2SquareF(const void* pa, const void* pb, const void* pd) {
    float dis = 0;
    auto ppa = (float*)const_cast<void*>(pa);
    auto ppb = (float*)const_cast<void*>(pb);
    for (auto i = 0; i < *(size_t*)pd; i ++) {
        auto diff = ppa[i] - ppb[i];
        dis += diff * diff;
    }
    return dis;
}

int L2SquareI(const void* pa, const void* pb, const void* pd) {
    int dis = 0;
    for (auto i = 0; i < *(size_t*)pd; i ++) {
        auto diff = ((int*)pa)[i] - ((int*)pb)[i];
        dis += diff * diff;
    }
    return dis;
}

float InnerProductF(const void* pa, const void* pb, const void* pd) {
    float dis = 0;
    for (auto i = 0; i < *(size_t*)pd; i ++) {
        dis += ((float*)pa)[i] * ((float*)pb)[i];
    }
    return dis;
}

int InnerProductI(const void* pa, const void* pb, const void* pd) {
    int dis = 0;
    for (auto i = 0; i < *(size_t*)pd; i ++) {
        dis += ((int*)pa)[i] * ((int*)pb)[i];
    }
    return dis;
}

} // namespace Vamana

