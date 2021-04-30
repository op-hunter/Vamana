//
// Created by cmli on 21-4-30.
//

#include <iostream>
#include "Distance.h"

#pragma once
namespace Vamana {

enum MetricType {
    L2 = 100,
    IP = 200,
};

enum DataType {
    INT32 = 100,
    FLOAT = 200,
    UINT8 = 300,
};

template <typename METRICTYPE>
using DISTANCE = METRICTYPE(*)(const void*, const void*, const void*);

template <typename dist_t>
class MetricSpace {
 public:
    virtual dist_t full_dist(const void*, const void*, const void*) = 0;
    inline dist_t half_dist(const void* pa, const void* pb, const void* pd) {
        size_t dim = *((size_t*)pd);
        dim >>= 1;
        return full_dist(pa, pb, &dim);
    }

    DISTANCE<dist_t> dis_func_ = nullptr;
};



///// L2 Space
class L2SpaceF: public MetricSpace<float> {
 public:
    L2SpaceF(const size_t dim) {
        dis_func_ = L2SquareF;
        // todo: instruction optimization
    }
    float full_dist(const void* pa, const void* pb, const void* pd) {
        return dis_func_(pa, pb, pd);
    }
};



///// IP Sapce
class IPSapceF: public MetricSpace<float> {
 public:
    IPSapceF(const size_t dim) {
        dis_func_ = InnerProductF;
        // todo: instruction optimization
    }
    float full_dist(const void* pa, const void* pb, const void* pd) {
        return dis_func_(pa, pb, pd);
    }
};


} // namespace Vamana

