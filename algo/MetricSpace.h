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

    inline dist_t post_half(const void* pa, const void* pb, const void* pd) {
        size_t dim = *((size_t*)pd);
        auto ppa = (char*)const_cast<void*>(pa);
        auto ppb = (char*)const_cast<void*>(pb);
        return full_dist(ppa + (dim >> 1) * esize, ppb + (dim >> 1) * esize, dim - (dim >> 1));
    }

    DISTANCE<dist_t> dis_func_ = nullptr;
    short esize;
};



///// L2 Space
class L2SpaceF: public MetricSpace<float> {
 public:
    L2SpaceF(const size_t dim) {
        dis_func_ = L2SquareF;
        esize = sizeof(float);
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
        esize = sizeof(float);
        // todo: instruction optimization
    }
    float full_dist(const void* pa, const void* pb, const void* pd) {
        return dis_func_(pa, pb, pd);
    }
};


} // namespace Vamana

