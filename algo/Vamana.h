//
// Created by cmli on 21-4-29.
//
#include "MetricSpace.h"
#include <vector>
#include <cmath>
#include <cstring>
#include <random>
#include <mutex>
#include <set>
#include <queue>
#include "omp.h"

#pragma once

namespace Vamana {
typedef uint32_t idx_t;

template <typename dist_t>
class Vamana {
 public:
    struct cmp {
        constexpr bool operator() (std::pair<dist_t, idx_t> const &a, std::pair<dist_t, idx_t> const &b) const noexcept {
            return a.first < b.first;
        }
    };
    using maxHeap = std::priority_queue<std::pair<dist_t, idx_t>, std::vector<std::pair<dist_t, idx_t>>, cmp>;
    Vamana(MetricType mt, DataType dt, size_t r, size_t l, float alp, size_t dim)
    : R_(r), L_(l), alpha_(alp), dim_(dim), link_list_locks_(1000000) {
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
        index_built_ = false;
    }

    void CreateIndex(const void* pdata, size_t n) {
        if (index_built_) {
            std::cout << "FBI Warning: index already built, if want re-build, call DropIndex first" << std::endl;
            return;
        }
        graph_ = (char*) malloc(node_size_ * n);
        memset(graph_, 0, node_size_ * n);
        if (R_ < (size_t)std::ceil(log2(n))) {
            std::cout << "FBI Warning: the parameter is less than log2(n), maybe result in low recall" << std::endl;
        }
        std::vector<std::mutex>(n).swap(link_list_locks_);
        addPoints(pdata, n);
        randomInit(n);
        buildIndex(pdata, n);
        index_built_ = true;
    }

 private:
    idx_t* getLinkByID(const idx_t id) {
        return (idx_t*)(graph_ + node_size_ * id);
    }

    char* getDataByID(const idx_t id) {
        return (char*)((graph_ + node_size_ * id) + link_size_);
    }

    void addPoints(const void* pdata, size_t n) {
        char* pd = (char*)(const_cast<void*>(pdata));
        // todo: parallelize
        for (auto i = 0; i < n; i ++) {
            memcpy(graph_ + i * node_size_ + link_size_, pd + i * data_size_, data_size_);
        }
    }

    void randomInit(const size_t n) {
        // todo: replace a better random generator
        srand(unsigned(time(NULL)));
#pragma omp parallel
    {
        auto nt = omp_get_num_threads(); // number of threads
        auto tn = omp_get_thread_num(); // thread number
        for (auto i = 0; i < n; i ++) {
            if (i % nt == tn) {
                std::set<idx_t> random_neighbors;
                do {
                    random_neighbors.insert((idx_t)(rand() % n));
                } while (random_neighbors.size() < R_);
                auto p_link = getLinkByID(i);
                assert(random_neighbors.size() <= R_);
                for (auto &chosen : random_neighbors) {
                    *p_link ++;
                    p_link[*p_link] = chosen;
                }
            }
        }
    }
    }

    void buildIndex(const void* pdata, size_t n) {
        float* pd = (float*)(const_cast<void*>(pdata));
        float* center = (float*) malloc(data_size_);
        // step1: calculate start point, i.e. navigate point in NSG
        for (size_t i = 0; i < n * dim_; i ++) {
            center[i % dim_] += pd[i];
        }
        for (auto i = 0; i < dim_; i ++)
            center[i] /= n;
        auto tpL = search(center, (idx_t)(random() % n), L_, n, true, nullptr);
        while (!tpL.empty()) {
            sp_ = tpL.top().second;
            tpL.pop();
        }

        // step2: do the first iteration with alpha = 1
#pragma omp parallel for schedule(dynamic)
        for (idx_t i = 0; i < n; i ++) {
            maxHeap candidates;
            search(pd + i * dim_, sp_, L_, n, false, candidates);
            robustPrune(i, candidates, 1.0);
            make_edge(i, 1.0);
        }

        // todo: need update sp_?

        // step3: do the second iteration with alpha = alpha_
#pragma omp parallel for schedule(dynamic)
        for (int i = (int)(n - 1); i >= 0; i --) {
            maxHeap candidates;
            search(pd + i * dim_, sp_, L_, n, false, candidates);
            robustPrune(i, candidates, alpha_);
            make_edge(i, alpha_);
        }

        // step4: update sp_
        tpL = search(center, sp_, L_, n, true, nullptr);
        while (!tpL.empty()) {
            sp_ = tpL.top().second;
            tpL.pop();
        }

    }

    maxHeap search(const void* qp, const idx_t sp, const size_t search_length, const size_t n, bool is_st, maxHeap& neighbor_candi) {
        std::vector<bool> vis(n, false);
        maxHeap resultset;
        maxHeap expandset;
        expandset.emplace(-ms_->full_dist(getDataByID(sp), qp, &dim_), sp);
        dist_t lowerBound = -expandset.top().first;
        while (expandset.size()) {
            auto cur = expandset.top();
            assert(cur.second < n);
            vis[cur.second] = true;
            if ((-cur.first) > lowerBound)
                break;
            expandset.pop();
            if (neighbor_candi != nullptr)
                neighbor_candi.emplace(cur);
            auto link = getLinkByID(cur.second);
            auto linksz = *link;
            std::unique_lock<std::mutex> lk(link_list_locks_[cur.second]);
            if (is_st)
                lk.unlock();
            for (auto i = 1; i <= linksz; i ++) {
                auto candi_id = link[i];
                // todo: prefetch
                if (vis[candi_id])
                    continue;
                auto candi_data = getDataByID(candi_id);
                auto dist = ms_->full_dist(qp, candi_data, &dim_);
                if (resultset.size() < search_length || dist < lowerBound) {
                    expandset.emplace(-dist, candi_id);
                    resultset.emplace(dist, candi_id);
                    if (resultset.size() > search_length)
                        resultset.pop();
                    if (!resultset.empty())
                        lowerBound = resultset.top().first;
                }
            }
        }
        return resultset;
    }

    void robustPrune(const idx_t p, maxHeap& cand_set, const float alpha) {
        std::unique_lock<std::mutex> lk(link_list_locks_[p]);
        auto link = getLinkByID(p);
        if (cand_set.size() < R_) {
            *link = (idx_t)cand_set.size();
            for (auto i = 1; i <= *link; i ++) {
                link[i] = cand_set.top().second;
                cand_set.pop();
            }
            return;
        }
        *link = 0;
        while (cand_set.size() > 0) {
            if (*link >= R_)
                break;
            auto cur = cand_set.top();
            cand_set.pop();
            bool good = true;
            for (auto j = 1; j <= *link; j ++) {
                auto dist = ms_->full_dist(getDataByID(cur.second), getDataByID(link[j]), &dim_);
                if (dist * alpha < -cur.first) {
                    good = false;
                    break;
                }
            }
            if (good) {
                (*link) ++;
                link[*link] = cur.second;
            }
        }
    }

    void make_edge(const idx_t p, const float alpha) {
        auto link = getLinkByID(p);
        for (auto i = 1; i <= *link; i ++) {
            auto neighbor_link = getLinkByID(link[i]);
            std::unique_lock<std::mutex> lk(link_list_locks_[link[i]]);
            if (*neighbor_link < R_) {
                (*neighbor_link) ++;
                neighbor_link[*neighbor_link] = p;
            } else {
                maxHeap pruneCandi;
                auto dist = ms_->full_dist(getDataByID(p), getDataByID(link[i]), &dim_);
                pruneCandi.emplace(-dist, p);
                for (auto j = 1; j <= *neighbor_link; j ++) {
                    pruneCandi.emplace(-ms_->full_dist(getDataByID(link[i]), getDataByID(neighbor_link[j]), &dim_), neighbor_link[j]);
                }
                robustPrune(link[i], pruneCandi, alpha);
            }
        }
    }

 private:
    size_t R_;
    size_t L_;
    float alpha_;
    MetricSpace<dist_t>* ms_;
    size_t data_size_;
    size_t link_size_;
    size_t node_size_;
    char *graph_;
    idx_t *sp_;
    bool index_built_;
    size_t dim_;
    std::vector<std::mutex> link_list_locks_;
};


} // namespace Vamana
