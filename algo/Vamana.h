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
#include <cassert>
#include <algorithm>
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
        sp_ = 0;
        graph_ = nullptr;
        ntotal_ = 0;
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
        ntotal_ = n;
        graph_ = (char*) malloc(node_size_ * n);
        memset(graph_, 0, node_size_ * ntotal_);
        if (R_ < (size_t)std::ceil(log2(ntotal_))) {
            std::cout << "FBI Warning: the parameter is less than log2(n), maybe result in low recall" << std::endl;
        }
        std::vector<std::mutex>(ntotal_).swap(link_list_locks_);
        addPoints(pdata, ntotal_);
        randomInit();
        HealthyCheck();
        buildIndex(pdata);
        index_built_ = true;
    }

    std::vector<std::pair<dist_t, idx_t>>
    Search(const void* pquery, const size_t topk) {
        std::vector<std::pair<dist_t, idx_t>> ret(topk);
        auto ans = search(pquery, sp_, topk);
        auto sz = ans.size();
        while (!ans.empty()) {
            sz --;
            ret[sz] = ans.top();
            ans.pop();
        }
        return ret;
    }

    std::vector<std::vector<std::pair<dist_t, idx_t>>>
    Search(const void* pquery, const size_t nq, const size_t topk) {
        std::vector<std::vector<std::pair<dist_t, idx_t>>> ret;
        ret.resize(nq);
        auto pq = (float*)pquery;
#pragma omp parallel for
        for (auto i = 0; i < nq; i ++) {
            ret[i] = Search(pq + i * dim_, topk);
        }
        return ret;
    }

    void HealthyCheck() {
        std::vector<size_t> degree_hist;
        scan_graph(degree_hist);
        std::cout << "show degree histogram of graph:" << std::endl;
        for (auto i = 0; i < degree_hist.size(); i ++) {
            std::cout << "degree = " << i << ": cnt = " << degree_hist[i] << std::endl;
        }
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

    void randomInit() {
        // todo: replace a better random generator
        srand(unsigned(time(NULL)));
#pragma omp parallel
    {
        auto nt = omp_get_num_threads(); // number of threads
        auto tn = omp_get_thread_num(); // thread number
        for (auto i = 0; i < ntotal_; i ++) {
            if (i % nt == tn) {
                std::set<idx_t> random_neighbors;
                do {
                    random_neighbors.insert((idx_t)(rand() % ntotal_));
                } while (random_neighbors.size() < R_);
                auto p_link = getLinkByID(i);
                assert(random_neighbors.size() <= R_);
                for (auto &chosen : random_neighbors) {
                    (*p_link) ++;
                    p_link[*p_link] = chosen;
                }
            }
        }
    }
    }

    void check_candidate_set(const idx_t cur, const maxHeap& candi) {
        std::vector<idx_t> candset;
        for (auto &pr : candi) {
            candset.emplace(pr.second);
        }
        if (candset.size() != candi.size()) {
            std::cout << "cur = " << cur << ", check_candidate_set failed" << std::endl;
        }
        assert(candi.size() == candset.size());
    }

    void buildIndex(const void* pdata) {
        assert(ntotal_ > 0);
        float* pd = (float*)(const_cast<void*>(pdata));
        float* center = (float*) malloc(data_size_);
        // step1: calculate start point, i.e. navigate point in NSG
        for (size_t i = 0; i < ntotal_ * dim_; i ++) {
            center[i % dim_] += pd[i];
        }
        for (auto i = 0; i < dim_; i ++)
            center[i] /= ntotal_;
        auto tstart = std::chrono::high_resolution_clock::now();
        auto tpL = search(center, (idx_t)(random() % ntotal_), L_);
        auto tend = std::chrono::high_resolution_clock::now();
        std::cout << "first search 4 sp_ finished in " << std::chrono::duration_cast<std::chrono::milliseconds>(tend - tstart).count() << " ms." << std::endl;
        while (!tpL.empty()) {
            sp_ = tpL.top().second;
//            std::cout << "sp_ = " << sp_ << std::endl;
            tpL.pop();
        }
        std::cout << "init sp_ = " << sp_ << std::endl;

        // step2: do the first iteration with alpha = 1
        tstart = std::chrono::high_resolution_clock::now();
#pragma omp parallel for schedule(dynamic, 100)
        for (idx_t i = 0; i < ntotal_; i ++) {
            maxHeap candidates;
            search(pd + i * dim_, candidates);
//            check_candidate_set(i, candidates);
//            std::cout << "search done." << std::endl;
            robustPrune(i, candidates, 1.0, 1);
//            std::cout << "robustPrune done." << std::endl;
            make_edge(i, 1.0);
//            std::cout << "make_edge done." << std::endl;
//            std::cout << "node " << i << " done." << std::endl;
//            if (i && (i % 10000 == 0))
//                std::cout << "done " << i << " nodes." << std::endl;
        }
        tend = std::chrono::high_resolution_clock::now();
        std::cout << "the first round iteration finished in " << std::chrono::duration_cast<std::chrono::milliseconds>(tend - tstart).count() << " ms." << std::endl;

        // todo: need update sp_?
        tpL = search(center, sp_, L_);
        while (!tpL.empty()) {
            sp_ = tpL.top().second;
            tpL.pop();
        }

        std::cout << "updated sp_ after 1st iteration: " << sp_ << std::endl;

        std::cout << "HealthyCheck after the 1st round iteration:" << std::endl;
        HealthyCheck();

        // step3: do the second iteration with alpha = alpha_
        tstart = std::chrono::high_resolution_clock::now();
#pragma omp parallel for schedule(dynamic, 100)
        for (int i = (int)(ntotal_ - 1); i >= 0; i --) {
            maxHeap candidates;
            search(pd + i * dim_, candidates);
//            check_candidate_set(i, candidates);
            robustPrune(i, candidates, alpha_, 2);
            make_edge(i, alpha_);
        }
        tend = std::chrono::high_resolution_clock::now();
        std::cout << "the second round iteration finished in " << std::chrono::duration_cast<std::chrono::milliseconds>(tend - tstart).count() << " ms." << std::endl;

        // step4: update sp_
        tpL = search(center, sp_, L_);
        while (!tpL.empty()) {
            sp_ = tpL.top().second;
            tpL.pop();
        }

        std::cout << "updated sp_ after 2nd iteration: " << sp_ << std::endl;

        std::cout << "HealthyCheck after the 2nd round iteration:" << std::endl;
        HealthyCheck();
    }

    maxHeap search(const void* qp, maxHeap& neighbor_candi) {
        while (neighbor_candi.size()) {
            neighbor_candi.pop();
            std::cout << "neighbor_candi not empty" << std::endl;
        }
        std::vector<bool> vis(ntotal_, false);
        maxHeap resultset;
        maxHeap expandset;
//        std::set<idx_t> ncset;
        expandset.emplace(-ms_->full_dist(getDataByID(sp_), qp, &dim_), sp_);
        vis[sp_] = true;
        dist_t lowerBound = -expandset.top().first;
        neighbor_candi.emplace(expandset.top());
//        ncset.emplace(sp_);
        while (expandset.size()) {
            auto cur = expandset.top();
            assert(cur.second < ntotal_);
            if ((-cur.first) > lowerBound)
                break;
            expandset.pop();
            auto link = getLinkByID(cur.second);
            auto linksz = *link;
            if (linksz > R_) {
                std::cout << "search: linksz = " << linksz << " which is > R_ = " << R_ << std::endl;
            }
            assert(linksz <= R_);
            std::unique_lock<std::mutex> lk(link_list_locks_[cur.second]);
            for (auto i = 1; i <= linksz; i ++) {
                auto candi_id = link[i];
                // todo: prefetch
                if (vis[candi_id])
                    continue;
                auto candi_data = getDataByID(candi_id);
                auto dist = ms_->full_dist(qp, candi_data, &dim_);
                if (resultset.size() < L_ || dist < lowerBound) {
                    expandset.emplace(-dist, candi_id);
                    vis[candi_id] = true;
                    neighbor_candi.emplace(-dist, candi_id);
//                    ncset.emplace(candi_id);
                    resultset.emplace(dist, candi_id);
                    if (resultset.size() > L_)
                        resultset.pop();
                    if (!resultset.empty())
                        lowerBound = resultset.top().first;
                }
            }
        }
//        if (ncset.size() != neighbor_candi.size()) {
//            std::cout << "after search, ncset.size = " << ncset.size() << " while neighbor_candi.size = " << neighbor_candi.size() << std::endl;
//        }
//        assert(ncset.size() == neighbor_candi.size());
        return resultset;
    }

     maxHeap search(const void* qp, const size_t sp, const size_t topk) {
        size_t ub = L_ < topk ? topk : L_;
        std::vector<bool> vis(ntotal_, false);
        maxHeap resultset;
        maxHeap expandset;
        expandset.emplace(-ms_->full_dist(getDataByID(sp), qp, &dim_), sp);
        vis[sp] = true;
        dist_t lowerBound = -expandset.top().first;
        while (expandset.size()) {
            auto cur = expandset.top();
            assert(cur.second < ntotal_);
            if ((-cur.first) > lowerBound)
                break;
            expandset.pop();
            auto link = getLinkByID(cur.second);
            auto linksz = *link;
            if (linksz > R_) {
                std::cout << "search_st: linksz = " << linksz << " which is > R_ = " << R_ << std::endl;
            }
            assert(linksz <= R_);
            for (auto i = 1; i <= linksz; i ++) {
                auto candi_id = link[i];
                // todo: prefetch
                if (vis[candi_id])
                    continue;
                auto candi_data = getDataByID(candi_id);
                auto dist = ms_->full_dist(qp, candi_data, &dim_);
                if (resultset.size() < ub || dist < lowerBound) {
                    expandset.emplace(-dist, candi_id);
                    vis[candi_id] = true;
                    resultset.emplace(dist, candi_id);
                    if (resultset.size() > ub)
                        resultset.pop();
                    if (!resultset.empty())
                        lowerBound = resultset.top().first;
                }
            }
        }
        return resultset;
    }


    void robustPrune(const idx_t p, maxHeap& cand_set, const float alpha, int flag) {
        std::unique_lock<std::mutex> lock(link_list_locks_[p]);
        auto link = getLinkByID(p);
        if (cand_set.size() <= R_) {
            *link = (idx_t)cand_set.size();
//            std::set<idx_t> ns;
            for (auto i = 1; i <= *link; i ++) {
                link[i] = cand_set.top().second;
//                ns.emplace(link[i]);
                cand_set.pop();
            }
            /*
            if (ns.size() != (*link)) {
                std::cout << "cur p = " << p << ", ns.size = " << ns.size() << " while *link = " << *link << std::endl;
                std::cout << "prune from " << flag << std::endl;
                std::vector<idx_t> sortlk;
                sortlk.resize(*link);
                std::cout << "show links:" << std::endl;
                for (auto i = 1; i <= *link; i ++) {
                    std::cout << link[i] << " ";
                    sortlk[i - 1] = link[i];
                }
                std::cout << std::endl;
                std::sort(sortlk.begin(), sortlk.end());
                std::cout << "show sorted links:" << std::endl;
                for (auto i = 0; i < sortlk.size(); i ++) {
                    std::cout << sortlk[i] << " ";
                }
                std::cout << std::endl;
                std::vector<idx_t> sortst;
                std::cout << "show ns:" << std::endl;
                for (auto &x : ns) {
                    std::cout << x << " ";
                    sortst.push_back(x);
                }
                std::cout << std::endl;
                std::sort(sortst.begin(), sortst.end());
                std::cout << "show sorted ns:" << std::endl;
                for (auto &x : sortst) {
                    std::cout << x << " ";
                }
                std::cout << std::endl;
            }
            */
//            assert(ns.size() == *link);
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
//                assert(link[j] != cur.second);
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

    bool isDuplicate(const idx_t p, const idx_t* link) {
        assert((*link) <= R_);
        for (auto i = 1; i <= *link; i ++) {
            if (p == link[i])
                return true;
        }
        return false;
    }

    void make_edge(const idx_t p, const float alpha) {
        auto link = getLinkByID(p);
        for (auto i = 1; i <= *link; i ++) {
            auto neighbor_link = getLinkByID(link[i]);
            std::unique_lock<std::mutex> lk(link_list_locks_[link[i]]);
            if (!isDuplicate(p, neighbor_link)) {
                if (*neighbor_link < R_) {
                    (*neighbor_link) ++;
                    neighbor_link[*neighbor_link] = p;
                } else {
                    lk.unlock();
                    maxHeap pruneCandi;
                    auto dist = ms_->full_dist(getDataByID(p), getDataByID(link[i]), &dim_);
                    pruneCandi.emplace(-dist, p);
                    for (auto j = 1; j <= *neighbor_link; j ++) {
                        pruneCandi.emplace(-ms_->full_dist(getDataByID(link[i]), getDataByID(neighbor_link[j]), &dim_), neighbor_link[j]);
                    }
                    robustPrune(link[i], pruneCandi, alpha, 3);
                }
            }
        }
    }

    void scan_graph(std::vector<size_t>& degree_histogram) {
        degree_histogram.resize(R_ + 1, 0);
        size_t total = 0;
#pragma omp parallel for reduction(+: total)
        for (auto i = 0; i < ntotal_; i ++) {
            auto link = getLinkByID(i);
            if ((*link) > R_) {
                std::cout << "scan_graph: *(link) = " << *link << " which is > R_ = " << R_ << std::endl;
            }
            assert((*link) <= R_);
#pragma omp critical
            {
                degree_histogram[*link] ++;
//                total ++;
            }
            std::set<idx_t> ns;
            for (auto j = 1; j <= *link; j ++) {
                assert(link[j] < ntotal_);
                ns.emplace(link[j]);
            }
            total += std::abs((int)(*link) - (int)(ns.size()));
//            assert(ns.size() == (*link));
        }
        std::cout << "scan_graph done, duplicate total = " << total << std::endl;
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
    idx_t sp_;
    bool index_built_;
    size_t dim_;
    size_t ntotal_;
    std::vector<std::mutex> link_list_locks_;
};


} // namespace Vamana
