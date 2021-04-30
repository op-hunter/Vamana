//
// Created by cmli on 21-4-29.
//
#include "MetricSpace.h"
#include <vector>
#include <cmath>
#include <cstring>

#pragma once

namespace Vamana {
typedef uint32_t idx_t;

template <typename dist_t>
class Vamana {
 public:
    Vamana(MetricType mt, DataType dt, size_t r, size_t l, float alp, size_t dim)
    : R_(r), L_(l), alpha_(alp) {
        if (MetricType::L2 == mt) {
            // todo: init by DataType, specified float temporarily
            ms_ = new L2SpaceF(dim);
        } else if (MetricType::IP == mt) {
            ms_ = new IPSapceF(dim);
        }
        // todo: default deal with float data
        data_size_ = dim * sizeof(float);
        link_size_ = R_ * sizeof(idx_t) + sizeof(idx_t);
        node_size_ = link_size_ + data_size_;
        index_built_ = false;
    }

    ~Vamana() {
        DropIndex();
    }

    void DropIndex() {
        if (graph_)
            free(graph_);
        graph_ = nullptr;
        if (sp_)
            free(sp_);
        sp_ = nullptr;
        index_built_ = false;
    }

    void CreateIndex(const void* pdata, size_t n) {
        if (index_built_) {
            std::cout << "FBI Warning: index already built, if want re-build, call DropIndex first" << std::endl;
            return;
        }
        graph_ = (char*) malloc(node_size_ * n);
        sp_ = (char*) malloc(data_size_);
        if (R_ < (size_t)std::ceil(log2(n))) {
            std::cout << "FBI Warning: the parameter is less than log2(n), maybe result in low recall" << std::endl;
        }
        addPoints(pdata, n);
        randomInit(n);
        buildIndex();
        index_built_ = true;
    }

 private:
    void addPoints(const void* pdata, size_t n) {
        float* pd = (float*)pdata;
        // todo: parallelize
        for (auto i = 0; i < n; i ++) {
            memcpy(graph_ + i * node_size_ + link_size_, pd + i * data_size_, data_size_);
        }
    }

    void randomInit(const size_t n) {
#pragma omp parallel
    {
    }
    }

    void buildIndex() {}

    void search(const void* qp, const size_t topk, const size_t search_length) {}

    void robustPrune(const idx_t p, std::vector<idx_t> cand_set, const float alpha, const size_t degree_ub) {}

 private:
    size_t R_;
    size_t L_;
    float alpha_;
    MetricSpace<dist_t>* ms_;
    size_t data_size_;
    size_t link_size_;
    size_t node_size_;
    char *graph_;
    char *sp_;
    bool index_built_;
};


} // namespace Vamana
