// Standalone scale/quality benchmark for VectorForge's in-repo ANN structures.
// Include the implementation directly so this test exercises the actual classes.
#define main vectorforge_server_entrypoint
#include "../main.cpp"
#undef main

#include <numeric>

static std::vector<float> makeVector(std::mt19937& rng, int dims) {
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> v(dims);
    for (float& x : v) x = dist(rng);
    return v;
}

static double mean(const std::vector<double>& values) {
    if (values.empty()) return 0.0;
    return std::accumulate(values.begin(), values.end(), 0.0) / values.size();
}

int main() {
    constexpr int D = 16;
    constexpr int K = 10;
    constexpr int QUERY_COUNT = 50;
    const std::vector<int> sizes = {1000, 5000, 10000};
    std::mt19937 rng(20261110);

    std::cout << "dataset_size,build_ms,queries,k,hnsw_recall_at_10,"
                 "bruteforce_avg_us,kdtree_cosine_avg_us,hnsw_avg_us\n";

    for (int n : sizes) {
        BruteForce brute;
        KDTree kd(D);
        HNSW hnsw(16, 200);
        std::vector<VectorItem> items;
        items.reserve(n);

        auto buildStart = std::chrono::steady_clock::now();
        for (int i = 0; i < n; ++i) {
            VectorItem item{i + 1, "benchmark", "synthetic", makeVector(rng, D)};
            items.push_back(item);
            brute.insert(item);
            kd.insert(item);
            hnsw.insert(item, cosine);
        }
        const double buildMs = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - buildStart).count();

        std::vector<double> bfTimes, kdTimes, hnswTimes;
        int relevantFound = 0;
        for (int qi = 0; qi < QUERY_COUNT; ++qi) {
            const auto q = makeVector(rng, D);
            auto timed = [](auto&& fn, std::vector<std::pair<float,int>>& out) {
                const auto start = std::chrono::steady_clock::now();
                out = fn();
                return std::chrono::duration<double, std::micro>(
                    std::chrono::steady_clock::now() - start).count();
            };

            std::vector<std::pair<float,int>> exact, kdResult, ann;
            bfTimes.push_back(timed([&]{ return brute.knn(q, K, cosine); }, exact));
            kdTimes.push_back(timed([&]{ return kd.knn(q, K, cosine, false); }, kdResult));
            hnswTimes.push_back(timed([&]{ return hnsw.knn(q, K, 50, cosine); }, ann));

            std::set<int> exactIds;
            for (const auto& hit : exact) exactIds.insert(hit.second);
            for (const auto& hit : ann)
                if (exactIds.count(hit.second)) ++relevantFound;

            if (kdResult.size() != exact.size()) {
                std::cerr << "KD-Tree returned a different result count at N=" << n << "\n";
                return 2;
            }
            for (size_t i = 0; i < exact.size(); ++i) {
                if (exact[i].second != kdResult[i].second ||
                    std::abs(exact[i].first - kdResult[i].first) > 1e-5f) {
                    std::cerr << "KD-Tree cosine exactness mismatch at N=" << n << "\n";
                    return 3;
                }
            }
        }

        const double recall = static_cast<double>(relevantFound) /
                              (QUERY_COUNT * K);
        std::cout << n << ',' << std::fixed << std::setprecision(2) << buildMs
                  << ',' << QUERY_COUNT << ',' << K << ','
                  << std::setprecision(4) << recall << ','
                  << std::setprecision(2) << mean(bfTimes) << ','
                  << mean(kdTimes) << ',' << mean(hnswTimes) << '\n';
    }
    return 0;
}
