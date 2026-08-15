#include <immintrin.h>
#include <omp.h>
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

struct Config {
    std::string dir = "benchmarks/cpu_data";
    int V = 151936;
    int D = 896;
    int N = 16;
    int reps = 5;
    int pilot_k = 4;
};

static std::vector<uint8_t> read_u8(const std::string& path, size_t n) {
    std::vector<uint8_t> x(n);
    std::ifstream f(path, std::ios::binary);
    if (!f || !f.read(reinterpret_cast<char*>(x.data()), static_cast<std::streamsize>(n)))
        throw std::runtime_error("failed to read " + path);
    return x;
}

static std::vector<float> read_f32(const std::string& path, size_t n) {
    std::vector<float> x(n);
    std::ifstream f(path, std::ios::binary);
    const size_t bytes = n * sizeof(float);
    if (!f || !f.read(reinterpret_cast<char*>(x.data()), static_cast<std::streamsize>(bytes)))
        throw std::runtime_error("failed to read " + path);
    return x;
}

static inline float hsum256(__m256 x) {
    __m128 hi = _mm256_extractf128_ps(x, 1);
    __m128 lo = _mm256_castps256_ps128(x);
    __m128 s = _mm_add_ps(lo, hi);
    s = _mm_hadd_ps(s, s);
    s = _mm_hadd_ps(s, s);
    return _mm_cvtss_f32(s);
}

static inline float hmax256(__m256 x) {
    __m128 hi = _mm256_extractf128_ps(x, 1);
    __m128 lo = _mm256_castps256_ps128(x);
    __m128 m = _mm_max_ps(lo, hi);
    __m128 shuf = _mm_movehdup_ps(m);
    m = _mm_max_ps(m, shuf);
    shuf = _mm_movehl_ps(shuf, m);
    m = _mm_max_ss(m, shuf);
    return _mm_cvtss_f32(m);
}

static inline float exact_dot_row(
    const uint8_t* high, const uint8_t* low, const float* h, int D) {
    __m256 acc = _mm256_setzero_ps();
    int j = 0;
    for (; j + 8 <= D; j += 8) {
        __m128i hb8 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(high + j));
        __m128i lb8 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(low + j));
        __m128i hb16 = _mm_cvtepu8_epi16(hb8);
        __m128i lb16 = _mm_cvtepu8_epi16(lb8);
        __m128i raw = _mm_or_si128(_mm_slli_epi16(hb16, 8), lb16);
        __m256 w = _mm256_cvtph_ps(raw);
        __m256 hv = _mm256_loadu_ps(h + j);
        acc = _mm256_fmadd_ps(w, hv, acc);
    }
    float sum = hsum256(acc);
    for (; j < D; ++j) {
        const uint16_t raw = (static_cast<uint16_t>(high[j]) << 8) | low[j];
        sum = std::fma(_cvtsh_ss(raw), h[j], sum);
    }
    return sum;
}

struct UpperResult { float upper; float rowmax; };

static inline UpperResult upper_dot_row(
    const uint8_t* high, const float* h, int D,
    const float* lo_lut, const float* hi_lut) {
    __m256 acc = _mm256_setzero_ps();
    __m256 vmax = _mm256_setzero_ps();
    const __m256 signmask = _mm256_set1_ps(-0.0f);
    const __m256 zero = _mm256_setzero_ps();
    int j = 0;
    for (; j + 8 <= D; j += 8) {
        __m128i hb8 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(high + j));
        __m256i idx = _mm256_cvtepu8_epi32(hb8);
        __m256 lo = _mm256_i32gather_ps(lo_lut, idx, 4);
        __m256 hi = _mm256_i32gather_ps(hi_lut, idx, 4);
        __m256 hv = _mm256_loadu_ps(h + j);
        __m256 negmask = _mm256_cmp_ps(hv, zero, _CMP_LT_OQ);
        __m256 endpoint = _mm256_blendv_ps(hi, lo, negmask);
        acc = _mm256_fmadd_ps(endpoint, hv, acc);
        __m256 alo = _mm256_andnot_ps(signmask, lo);
        __m256 ahi = _mm256_andnot_ps(signmask, hi);
        vmax = _mm256_max_ps(vmax, _mm256_max_ps(alo, ahi));
    }
    float sum = hsum256(acc);
    float rowmax = hmax256(vmax);
    for (; j < D; ++j) {
        const int hb = high[j];
        const float endpoint = h[j] >= 0.0f ? hi_lut[hb] : lo_lut[hb];
        sum = std::fma(endpoint, h[j], sum);
        rowmax = std::max(rowmax, std::max(std::abs(lo_lut[hb]), std::abs(hi_lut[hb])));
    }
    return {sum, rowmax};
}

static inline double gamma_n(int n, double u = std::ldexp(1.0, -24)) {
    const double x = static_cast<double>(n) * u;
    if (x >= 1.0) throw std::runtime_error("gamma invalid");
    return x / (1.0 - x);
}

static void make_endpoint_lut(float* lo, float* hi) {
    for (int hb = 0; hb < 256; ++hb) {
        const int exp = (hb >> 2) & 0x1f;
        if (exp == 0x1f) {
            lo[hb] = hi[hb] = 0.0f;
            continue;
        }
        const uint16_t a = static_cast<uint16_t>(hb << 8);
        const uint16_t b = static_cast<uint16_t>((hb << 8) | 0xff);
        const float fa = _cvtsh_ss(a), fb = _cvtsh_ss(b);
        lo[hb] = std::min(fa, fb);
        hi[hb] = std::max(fa, fb);
    }
}

static int dense_argmax(
    const std::vector<uint8_t>& high, const std::vector<uint8_t>& low,
    const float* h, int V, int D, std::vector<float>& scores) {
#pragma omp parallel for schedule(static)
    for (int i = 0; i < V; ++i) {
        scores[i] = exact_dot_row(high.data() + static_cast<size_t>(i) * D,
                                  low.data() + static_cast<size_t>(i) * D,
                                  h, D);
    }
    return static_cast<int>(std::max_element(scores.begin(), scores.end()) - scores.begin());
}

struct ProofResult {
    int winner;
    int survivors;
    int low_rows;
};

static ProofResult proofbits_argmax(
    const std::vector<uint8_t>& high, const std::vector<uint8_t>& low,
    const float* h, int V, int D, int pilot_k,
    const float* lo_lut, const float* hi_lut,
    std::vector<float>& upper, std::vector<int>& survivors,
    std::vector<int>& pilots) {
    // Conservative h-L1 upper bound. Double summation makes its error negligible;
    // gamma_D is still applied to avoid relying on that empirical fact.
    double h1 = 0.0;
    for (int j = 0; j < D; ++j) h1 += std::abs(static_cast<double>(h[j]));
    h1 /= (1.0 - gamma_n(D, std::ldexp(1.0, -53)));
    const double coeff = 2.0 * gamma_n(4 * D);

#pragma omp parallel for schedule(static)
    for (int i = 0; i < V; ++i) {
        auto r = upper_dot_row(high.data() + static_cast<size_t>(i) * D,
                               h, D, lo_lut, hi_lut);
        const double safe = static_cast<double>(r.upper) + coeff * h1 * static_cast<double>(r.rowmax);
        upper[i] = static_cast<float>(safe);
    }

    pilots.resize(pilot_k);
    std::vector<int> ids(V);
    std::iota(ids.begin(), ids.end(), 0);
    std::partial_sort(ids.begin(), ids.begin() + pilot_k, ids.end(),
                      [&](int a, int b) { return upper[a] > upper[b]; });
    for (int k = 0; k < pilot_k; ++k) pilots[k] = ids[k];

    float B = -std::numeric_limits<float>::infinity();
    for (int p : pilots) {
        B = std::max(B, exact_dot_row(high.data() + static_cast<size_t>(p) * D,
                                      low.data() + static_cast<size_t>(p) * D,
                                      h, D));
    }

    survivors.clear();
    for (int i = 0; i < V; ++i) if (upper[i] >= B) survivors.push_back(i);

    int winner = -1;
    float best = -std::numeric_limits<float>::infinity();
#pragma omp parallel
    {
        int local_winner = -1;
        float local_best = -std::numeric_limits<float>::infinity();
#pragma omp for nowait schedule(static)
        for (size_t k = 0; k < survivors.size(); ++k) {
            const int i = survivors[k];
            const float z = exact_dot_row(high.data() + static_cast<size_t>(i) * D,
                                          low.data() + static_cast<size_t>(i) * D,
                                          h, D);
            if (z > local_best || (z == local_best && (local_winner < 0 || i < local_winner))) {
                local_best = z;
                local_winner = i;
            }
        }
#pragma omp critical
        {
            if (local_best > best || (local_best == best && local_winner >= 0 && (winner < 0 || local_winner < winner))) {
                best = local_best;
                winner = local_winner;
            }
        }
    }

    int low_rows = static_cast<int>(survivors.size());
    for (int p : pilots) {
        if (!std::binary_search(survivors.begin(), survivors.end(), p)) ++low_rows;
    }
    return {winner, static_cast<int>(survivors.size()), low_rows};
}

struct Stats { double median_ms, p10_ms, p90_ms; };

static Stats summarize(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    auto q = [&](double p) {
        size_t i = static_cast<size_t>(p * (v.size() - 1));
        return v[i];
    };
    return {q(0.5), q(0.1), q(0.9)};
}

int main(int argc, char** argv) {
    Config c;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto val = [&]() -> std::string { if (++i >= argc) throw std::runtime_error("missing arg"); return argv[i]; };
        if (a == "--dir") c.dir = val();
        else if (a == "--vocab") c.V = std::stoi(val());
        else if (a == "--dim") c.D = std::stoi(val());
        else if (a == "--states") c.N = std::stoi(val());
        else if (a == "--reps") c.reps = std::stoi(val());
        else if (a == "--pilot-k") c.pilot_k = std::stoi(val());
    }

#if defined(__GNUC__)
    if (!__builtin_cpu_supports("avx2") || !__builtin_cpu_supports("f16c") || !__builtin_cpu_supports("fma")) {
        std::cerr << "CPU lacks AVX2/F16C/FMA; benchmark requires these instructions.\n";
        return 3;
    }
#endif

    const size_t WN = static_cast<size_t>(c.V) * c.D;
    const size_t HN = static_cast<size_t>(c.N) * c.D;
    auto high = read_u8(c.dir + "/high.bin", WN);
    auto low = read_u8(c.dir + "/low.bin", WN);
    auto hidden = read_f32(c.dir + "/hidden.bin", HN);

    alignas(32) float lo_lut[256], hi_lut[256];
    make_endpoint_lut(lo_lut, hi_lut);
    for (uint8_t hb : high) {
        if (((hb >> 2) & 0x1f) == 0x1f) throw std::runtime_error("non-finite high-byte pattern");
    }

    std::vector<float> dense_scores(c.V), upper(c.V);
    std::vector<int> survivors; survivors.reserve(c.V);
    std::vector<int> pilots; pilots.reserve(c.pilot_k);

    std::vector<int> thread_counts = {1};
    int max_threads = omp_get_max_threads();
    if (max_threads >= 2) thread_counts.push_back(std::min(4, max_threads));
    std::sort(thread_counts.begin(), thread_counts.end());
    thread_counts.erase(std::unique(thread_counts.begin(), thread_counts.end()), thread_counts.end());

    struct Row { int threads; Stats dense, proof; double speedup; double mean_low_rows; double mean_survivors; bool exact; };
    std::vector<Row> rows;

    for (int threads : thread_counts) {
        omp_set_num_threads(threads);
        bool exact_ok = true;
        double low_sum = 0.0, surv_sum = 0.0;
        for (int n = 0; n < c.N; ++n) {
            const float* h = hidden.data() + static_cast<size_t>(n) * c.D;
            const int d = dense_argmax(high, low, h, c.V, c.D, dense_scores);
            auto p = proofbits_argmax(high, low, h, c.V, c.D, c.pilot_k, lo_lut, hi_lut, upper, survivors, pilots);
            exact_ok = exact_ok && (d == p.winner);
            low_sum += p.low_rows;
            surv_sum += p.survivors;
        }
        if (!exact_ok) throw std::runtime_error("ProofBits argmax mismatch in correctness pass");

        // Warm-up both paths over all states.
        for (int n = 0; n < c.N; ++n) {
            const float* h = hidden.data() + static_cast<size_t>(n) * c.D;
            dense_argmax(high, low, h, c.V, c.D, dense_scores);
            proofbits_argmax(high, low, h, c.V, c.D, c.pilot_k, lo_lut, hi_lut, upper, survivors, pilots);
        }

        std::vector<double> td, tp;
        for (int r = 0; r < c.reps; ++r) {
            auto t0 = std::chrono::steady_clock::now();
            for (int n = 0; n < c.N; ++n) {
                const float* h = hidden.data() + static_cast<size_t>(n) * c.D;
                dense_argmax(high, low, h, c.V, c.D, dense_scores);
            }
            auto t1 = std::chrono::steady_clock::now();
            for (int n = 0; n < c.N; ++n) {
                const float* h = hidden.data() + static_cast<size_t>(n) * c.D;
                proofbits_argmax(high, low, h, c.V, c.D, c.pilot_k, lo_lut, hi_lut, upper, survivors, pilots);
            }
            auto t2 = std::chrono::steady_clock::now();
            td.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count() / c.N);
            tp.push_back(std::chrono::duration<double, std::milli>(t2 - t1).count() / c.N);
        }
        Stats ds = summarize(td), ps = summarize(tp);
        rows.push_back({threads, ds, ps, ds.median_ms / ps.median_ms,
                        low_sum / c.N, surv_sum / c.N, exact_ok});
        std::cerr << "threads=" << threads << " dense=" << ds.median_ms
                  << " ms proof=" << ps.median_ms << " ms speedup="
                  << ds.median_ms / ps.median_ms << "x low_rows="
                  << low_sum / c.N << "\n";
    }

    std::ofstream out("experiments/artifacts/proofbits_cpu_avx2_benchmark.json");
    out << "{\n";
    out << "  \"kind\": \"matched_fp16_byteplane_avx2_hardware_benchmark\",\n";
    out << "  \"vocab\": " << c.V << ", \"hidden_dim\": " << c.D << ", \"states\": " << c.N << ",\n";
    out << "  \"pilot_k\": " << c.pilot_k << ",\n";
    out << "  \"reference\": \"same FP16 high/low byte planes; AVX2/F16C exact row reconstruction; float/FMA accumulation\",\n";
    out << "  \"safe_certificate\": \"upper-only plus conservative 2*gamma_4d*||h||_1*M_i\",\n";
    out << "  \"results\": [\n";
    for (size_t i = 0; i < rows.size(); ++i) {
        const auto& r = rows[i];
        const double f = r.mean_low_rows / c.V;
        const double ideal = 2.0 / (1.0 + f);
        out << "    {\"threads\": " << r.threads
            << ", \"exact_argmax\": " << (r.exact ? "true" : "false")
            << ", \"dense_median_ms_per_state\": " << r.dense.median_ms
            << ", \"dense_p10_ms\": " << r.dense.p10_ms
            << ", \"dense_p90_ms\": " << r.dense.p90_ms
            << ", \"proofbits_median_ms_per_state\": " << r.proof.median_ms
            << ", \"proofbits_p10_ms\": " << r.proof.p10_ms
            << ", \"proofbits_p90_ms\": " << r.proof.p90_ms
            << ", \"measured_speedup\": " << r.speedup
            << ", \"mean_survivors\": " << r.mean_survivors
            << ", \"mean_distinct_low_rows\": " << r.mean_low_rows
            << ", \"idealized_weight_byte_reduction\": " << ideal << "}";
        if (i + 1 != rows.size()) out << ",";
        out << "\n";
    }
    out << "  ],\n";
    out << "  \"caveat\": \"Real hosted-CPU wall clock, not GPU. Dense and ProofBits use the same byte-plane storage and exact-row reconstruction, but this custom AVX2/F16C kernel is not a vendor BLAS baseline. GPU timing and DRAM counters remain required.\"\n";
    out << "}\n";
    out.close();
    return 0;
}
