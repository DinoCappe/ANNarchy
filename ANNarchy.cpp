#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <queue>
#include <random>
#include <unordered_set>
#include <vector>

namespace nb = nanobind;

using float2d = nb::ndarray<float, nb::ndim<2>, nb::c_contig, nb::device::cpu>;
using float1d = nb::ndarray<float, nb::ndim<1>, nb::c_contig, nb::device::cpu>;

static int64_t g_distances_count = 0;

static inline float sq_dist(const float* __restrict__ a,
                             const float* __restrict__ b,
                             size_t dim) {
    float d = 0.0f;
    for (size_t i = 0; i < dim; i++) {
        float diff = a[i] - b[i];
        d += diff * diff;
    }
    return d;
}

// ===========================================================================
// IVF
// ===========================================================================
struct IVFIndex {
    std::vector<float>   points;
    std::vector<float>   centroids;
    std::vector<std::vector<int32_t>> lists;
    size_t n = 0, dim = 0, nlist = 0;

    static size_t assign_points(const std::vector<float>& pts,
                                const std::vector<float>& cents,
                                std::vector<int32_t>& assignments,
                                size_t n, size_t dim, size_t nlist) {
        size_t changes = 0;
        #pragma omp parallel for reduction(+:changes) schedule(static)
        for (size_t i = 0; i < n; i++) {
            const float* pt = pts.data() + i * dim;
            float best_d = std::numeric_limits<float>::infinity();
            int32_t best_c = 0;
            for (size_t c = 0; c < nlist; c++) {
                float d = sq_dist(pt, cents.data() + c * dim, dim);
                if (d < best_d) { best_d = d; best_c = static_cast<int32_t>(c); }
            }
            if (assignments[i] != best_c) { assignments[i] = best_c; changes++; }
        }
        return changes;
    }

    static void update_centroids(const std::vector<float>& pts,
                                 std::vector<float>& cents,
                                 const std::vector<int32_t>& assignments,
                                 size_t n, size_t dim, size_t nlist) {
        std::vector<float>   sums(nlist * dim, 0.0f);
        std::vector<int32_t> counts(nlist, 0);
        for (size_t i = 0; i < n; i++) {
            int32_t c = assignments[i];
            const float* pt = pts.data() + i * dim;
            float* s = sums.data() + c * dim;
            for (size_t j = 0; j < dim; j++) s[j] += pt[j];
            counts[c]++;
        }
        for (size_t c = 0; c < nlist; c++) {
            if (counts[c] == 0) continue;
            float inv = 1.0f / counts[c];
            float* centroid = cents.data() + c * dim;
            float* s = sums.data() + c * dim;
            for (size_t j = 0; j < dim; j++) centroid[j] = s[j] * inv;
        }
    }

    void fit(float2d train, int nlist_, int niter) {
        n     = train.shape(0);
        dim   = train.shape(1);
        nlist = static_cast<size_t>(nlist_);
        points.assign(train.data(), train.data() + n * dim);

        std::mt19937 rng(42);
        std::vector<size_t> perm(n);
        std::iota(perm.begin(), perm.end(), 0);
        std::shuffle(perm.begin(), perm.end(), rng);

        centroids.resize(nlist * dim);
        for (size_t c = 0; c < nlist; c++)
            std::copy(points.data() + perm[c] * dim,
                      points.data() + perm[c] * dim + dim,
                      centroids.data() + c * dim);

        std::vector<int32_t> assignments(n, -1);
        for (int iter = 0; iter < niter; iter++) {
            size_t changes = assign_points(points, centroids, assignments, n, dim, nlist);
            update_centroids(points, centroids, assignments, n, dim, nlist);
            if (changes == 0) break;
        }

        lists.assign(nlist, {});
        for (size_t i = 0; i < n; i++)
            lists[assignments[i]].push_back(static_cast<int32_t>(i));
    }

    nb::ndarray<nb::numpy, int32_t, nb::ndim<1>> query(float1d q, int k, int nprobe) {
        const float* qdata = q.data();

        std::vector<std::pair<float, int32_t>> cdists(nlist);
        for (size_t c = 0; c < nlist; c++)
            cdists[c] = {sq_dist(qdata, centroids.data() + c * dim, dim),
                         static_cast<int32_t>(c)};
        std::partial_sort(cdists.begin(), cdists.begin() + nprobe, cdists.end());
        g_distances_count += static_cast<int64_t>(nlist);

        std::vector<std::pair<float, int32_t>> candidates;
        for (int p = 0; p < nprobe; p++) {
            for (int32_t idx : lists[cdists[p].second]) {
                float d = sq_dist(qdata, points.data() + idx * dim, dim);
                candidates.push_back({d, idx});
            }
        }
        g_distances_count += static_cast<int64_t>(candidates.size());

        int actual_k = std::min(k, (int)candidates.size());
        std::partial_sort(candidates.begin(), candidates.begin() + actual_k, candidates.end());

        int32_t* result = new int32_t[k]();
        for (int i = 0; i < actual_k; i++) result[i] = candidates[i].second;
        nb::capsule owner(result, [](void* p) noexcept { delete[] static_cast<int32_t*>(p); });
        size_t shape[1] = {static_cast<size_t>(k)};
        return nb::ndarray<nb::numpy, int32_t, nb::ndim<1>>(result, 1, shape, owner);
    }
};

// ===========================================================================
// HNSW
// ===========================================================================
struct HNSWIndex {
    int     M               = 16;
    int     ef_construction = 100;
    float   mL              = 0.0f;
    int     max_level       = -1;
    int32_t entry_point     = -1;
    size_t  n = 0, dim = 0;
    std::vector<float> data;
    // adj[node][layer] = neighbor list; node has layers 0..level[node]
    std::vector<std::vector<std::vector<int32_t>>> adj;
    std::mt19937 rng{42};
    int64_t dist_count = 0;

    using pfi = std::pair<float, int32_t>;

    void reset(int M_, int ef_construction_) {
        M = M_; ef_construction = ef_construction_;
        mL = 1.0f / std::log(static_cast<float>(M));
        max_level = -1; entry_point = -1; dist_count = 0;
        data.clear(); adj.clear();
    }

    int random_level() {
        std::uniform_real_distribution<double> u(0.0, 1.0);
        return static_cast<int>(-std::log(u(rng)) * mL);
    }

    inline float qdist(const float* q, int32_t b) {
        dist_count++;
        return sq_dist(q, data.data() + b * dim, dim);
    }

    // Greedy descent: move to the closest neighbor until no improvement.
    // Used on upper layers where we only need one good entry point.
    int32_t greedy_search(const float* q, int32_t ep, int layer) {
        float best_d = qdist(q, ep);
        bool changed = true;
        while (changed) {
            changed = false;
            for (int32_t nb : adj[ep][layer]) {
                float d = qdist(q, nb);
                if (d < best_d) { best_d = d; ep = nb; changed = true; }
            }
        }
        return ep;
    }

    // Beam search: find `ef` nearest nodes to q from entry point ep at `layer`.
    // Returns results sorted ascending (closest first).
    std::vector<pfi> beam_search(const float* q, int32_t ep, int ef, int layer) {
        std::unordered_set<int32_t> visited;
        visited.reserve(ef * 4);
        visited.insert(ep);

        float ep_d = qdist(q, ep);

        // min-heap: closest unvisited node expanded first
        std::priority_queue<pfi, std::vector<pfi>, std::greater<pfi>> cands;
        cands.push({ep_d, ep});

        // max-heap: evict the farthest when over capacity
        std::priority_queue<pfi> results;
        results.push({ep_d, ep});

        while (!cands.empty()) {
            auto [cd, c] = cands.top(); cands.pop();
            // If the closest remaining candidate is farther than our worst result,
            // no future expansion can improve the result set.
            if (cd > results.top().first) break;

            for (int32_t nb : adj[c][layer]) {
                if (!visited.insert(nb).second) continue;
                float nd = qdist(q, nb);
                if ((int)results.size() < ef || nd < results.top().first) {
                    cands.push({nd, nb});
                    results.push({nd, nb});
                    if ((int)results.size() > ef) results.pop();
                }
            }
        }

        std::vector<pfi> out;
        out.reserve(results.size());
        while (!results.empty()) { out.push_back(results.top()); results.pop(); }
        std::sort(out.begin(), out.end());
        return out;
    }

    void insert(int32_t idx) {
        int level = random_level();
        adj[idx].resize(level + 1);

        if (entry_point < 0) { entry_point = idx; max_level = level; return; }

        int32_t ep      = entry_point;
        const float* pt = data.data() + idx * dim;

        // Phase 1: greedy descent through layers above this node's level
        for (int l = max_level; l > level; l--)
            ep = greedy_search(pt, ep, l);

        // Phase 2: beam search + wire edges from level down to 0
        for (int l = std::min(level, max_level); l >= 0; l--) {
            int Ml = (l == 0) ? 2 * M : M;

            auto found = beam_search(pt, ep, ef_construction, l);
            if (!found.empty()) ep = found[0].second;

            int take = std::min(Ml, (int)found.size());
            adj[idx][l].reserve(take);
            for (int i = 0; i < take; i++) adj[idx][l].push_back(found[i].second);

            // Reverse edges with pruning
            for (int i = 0; i < take; i++) {
                int32_t nb        = found[i].second;
                const float* nbpt = data.data() + nb * dim;
                adj[nb][l].push_back(idx);

                if ((int)adj[nb][l].size() > Ml) {
                    std::vector<pfi> nbcands;
                    nbcands.reserve(adj[nb][l].size());
                    for (int32_t x : adj[nb][l])
                        nbcands.push_back({sq_dist(nbpt, data.data() + x * dim, dim), x});
                    std::sort(nbcands.begin(), nbcands.end());
                    adj[nb][l].clear();
                    for (int j = 0; j < Ml; j++) adj[nb][l].push_back(nbcands[j].second);
                }
            }
        }

        if (level > max_level) { entry_point = idx; max_level = level; }
    }

    void fit(float2d train, int M_, int ef_construction_) {
        reset(M_, ef_construction_);
        n   = train.shape(0);
        dim = train.shape(1);
        data.assign(train.data(), train.data() + n * dim);
        adj.resize(n);
        for (size_t i = 0; i < n; i++) insert(static_cast<int32_t>(i));
        dist_count = 0;  // build distances don't count toward the prize
    }

    nb::ndarray<nb::numpy, int32_t, nb::ndim<1>> query(const float* qdata, int k, int ef_search) {
        ef_search = std::max(ef_search, k);
        dist_count = 0;

        int32_t ep = entry_point;
        for (int l = max_level; l > 0; l--)
            ep = greedy_search(qdata, ep, l);

        auto found = beam_search(qdata, ep, ef_search, 0);
        g_distances_count += dist_count;

        int actual_k = std::min(k, (int)found.size());
        int32_t* result = new int32_t[k]();
        for (int i = 0; i < actual_k; i++) result[i] = found[i].second;
        nb::capsule owner(result, [](void* p) noexcept { delete[] static_cast<int32_t*>(p); });
        size_t shape[1] = {static_cast<size_t>(k)};
        return nb::ndarray<nb::numpy, int32_t, nb::ndim<1>>(result, 1, shape, owner);
    }
};

// ===========================================================================
// Global instances
// ===========================================================================
static IVFIndex  g_ivf;
static HNSWIndex g_hnsw;

// ---------------------------------------------------------------------------
// IVF API
// ---------------------------------------------------------------------------
void ivf_fit(float2d train, int nlist = 256, int niter = 5) {
    g_distances_count = 0;
    g_ivf.fit(train, nlist, niter);
}

nb::ndarray<nb::numpy, int32_t, nb::ndim<1>>
ivf_query(float1d q, int k, int nprobe = 32) {
    return g_ivf.query(q, k, nprobe);
}

// ---------------------------------------------------------------------------
// HNSW API
// ---------------------------------------------------------------------------
void hnsw_fit(float2d train, int M = 16, int ef_construction = 100) {
    g_distances_count = 0;
    g_hnsw.fit(train, M, ef_construction);
}

nb::ndarray<nb::numpy, int32_t, nb::ndim<1>>
hnsw_query(float1d q, int k, int ef_search = 50) {
    return g_hnsw.query(q.data(), k, ef_search);
}

// ---------------------------------------------------------------------------
// Shared
// ---------------------------------------------------------------------------
int64_t total_distances_count() { return g_distances_count; }

NB_MODULE(ANNarchy, m) {
    m.doc() = "ANNarchy — IVF and HNSW nearest neighbor search";

    m.def("ivf_fit",   &ivf_fit,   nb::arg("train"), nb::arg("nlist") = 256, nb::arg("niter") = 5);
    m.def("ivf_query", &ivf_query, nb::arg("q"),     nb::arg("k"),           nb::arg("nprobe") = 32);

    m.def("hnsw_fit",   &hnsw_fit,   nb::arg("train"), nb::arg("M") = 16, nb::arg("ef_construction") = 100);
    m.def("hnsw_query", &hnsw_query, nb::arg("q"),     nb::arg("k"),      nb::arg("ef_search") = 50);

    m.def("total_distances_count", &total_distances_count);
}
