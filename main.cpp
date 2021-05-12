#include <iostream>
#include <fstream>
#include <unistd.h>
#include "algo/Vamana.h"
#include "omp.h"

void test_omp() {

#pragma omp parallel
    {
        auto nt = omp_get_num_threads();
        auto tn = omp_get_thread_num();
#pragma omp critical
        {
            std::cout << "current tn = " << tn << ", total nt = " << nt << std::endl;
        };
        for (auto i = 0; i < 15; i ++) {
            if (i % nt == tn) {
#pragma omp critical
                {
                    std::cout << tn << " : " << i << std::endl;
                };
            }
        }
    }
    std::vector<int> cnt(15, 0);
#pragma omp parallel for schedule(dynamic)
    for (auto i = 0; i < 100000; i ++) {
        usleep(2000);
        cnt[omp_get_thread_num()] ++;
    }
    std::cout << "show histogram of omp:" << std::endl;
    for (auto i = 0; i < cnt.size(); i ++) {
        std::cout << "cnt[" << i << "] = " << cnt[i] << std::endl;
    }
}

int
main() {
//    test_omp();
    using idx_t = Vamana::idx_t;
    unsigned nbd;
    unsigned nb;
    unsigned nq;
    unsigned nqd;
    size_t k = 100;

    float* pdata = nullptr;
    float* pquery  = nullptr;

    std::ifstream fin("/home/zilliz/workspace/data/sift1m.bin", std::ios::binary);
    std::cout << "open file" << std::endl;
    fin.read((char*)&nb, 4);
    std::cout << "read nb = " << nb << std::endl;
    fin.read((char*)&nbd, 4);
    std::cout << "read nbd = " << nbd << std::endl;
    pdata = (float*)malloc(nb * nbd * 4);
    std::cout << "malloc space for base data" << std::endl;
    fin.read((char*)pdata, nb * nbd * 4);
    fin.close();
    std::cout << "read sift1m.bin done." << std::endl;
    std::ifstream finq("/home/zilliz/workspace/data/sift1m_query.bin", std::ios::binary);
    std::cout << "open query file" << std::endl;
    finq.read((char*)&nq, 4);
    std::cout << "read nq = " << nq << std::endl;
    finq.read((char*)&nqd, 4);
    std::cout << "read nqd = " << nqd << std::endl;
    pquery = (float*)malloc(nq * nqd * 4);
    std::cout << "malloc space for query data" << std::endl;
    finq.read((char*)pquery, nq * nqd * 4);
    finq.close();
    std::cout << "read sift1m_query.bin done." << std::endl;

    std::cout << "nb = " << nb << ", nq = " << nq << ", nq-dim = " << nqd << ", nb-dim = " << nbd << std::endl;
    if (nbd != nqd) {
        std::cout << "nqd = " << nqd << " != nbd = " << nbd << std::endl;
        return 1;
    }



    Vamana::Vamana<float>* alg_vamana = new Vamana::Vamana<float>(Vamana::MetricType::L2, Vamana::DataType::FLOAT, 70, 75, 1.2, nbd);

    auto tstart = std::chrono::high_resolution_clock::now();
    alg_vamana->CreateIndex(pdata, nb);
    auto tend = std::chrono::high_resolution_clock::now();
    std::cout << "create index done in " << std::chrono::duration_cast<std::chrono::milliseconds>( tend - tstart ).count() << " millseconds." << std::endl;

    std::vector<std::vector<std::pair<float, idx_t>>> resultset;
    resultset.resize(nq);
    tstart = std::chrono::high_resolution_clock::now();
#pragma omp parallel for
    for (size_t j = 0; j < nq; ++j) {
        const void* p = pquery + j * nqd;
        resultset[j] = alg_vamana->Search(p, k);
    }
    tend = std::chrono::high_resolution_clock::now();
    std::cout << "query done in " << std::chrono::duration_cast<std::chrono::milliseconds>( tend - tstart ).count() << " millseconds." << std::endl;

    // load ground truth
    size_t sz;
    std::vector<std::vector<std::pair<float, size_t>>> groundtruth;
    groundtruth.resize(nq);
    std::ifstream finn("ground_truth_100.bin.2", std::ios::binary);
    for (unsigned i = 0; i < nq; i ++) {
        finn.read((char*)&sz, 8);
        std::cout << "query " << i + 1 << " has " << sz << " groundtruth ans." << std::endl;
        groundtruth[i].resize(sz);
        finn.read((char*)groundtruth[i].data(), k * 16);
    }
    finn.close();

    std::cout << "show groundtruth:" << std::endl;
    for (size_t i = 0; i < groundtruth.size(); i ++) {
        for (size_t j = 0; j < k; j ++) {
            std::cout << "(" << groundtruth[i][j].second << ", " << groundtruth[i][j].first << ") ";
        }
        std::cout << std::endl;
    }

    std::cout << "--------------------------------------------------------------------" << std::endl;

    std::cout << "show resultset:" << std::endl;
    for (size_t i = 0; i < resultset.size(); i ++) {
        for (size_t j = 0; j < k; j ++) {
            std::cout << "(" << resultset[i][j].second << ", " << resultset[i][j].first << ") ";
        }
        std::cout << std::endl;
    }

    // calculate recall@k
    int tot_cnt = 0;
    std::cout << "recall@" << k << ":" << std::endl;
    for (unsigned i = 0; i < nq; i ++) {
        int cnt = 0;
        // std::cout << "groundtruth[i][k - 1].first = " << groundtruth[i][k - 1].first << std::endl;
        for (size_t j = 0; j < k; j ++) {
            if (resultset[i][j].first <= groundtruth[i][k - 1].first)
                cnt ++;
        }
        // std::cout << "cnt = " << cnt << std::endl;
        tot_cnt += cnt;
        std::cout << "query " << i + 1 << " recall@" << k << " is: " << ((double)(cnt)) / k * 100 << "%." << std::endl;
    }
    std::cout << "avg recall@100 = " << ((double)tot_cnt) / k / nq * 100 << "%." << std::endl;

    delete alg_vamana;
    free(pdata);
    free(pquery);
    return 0;
}