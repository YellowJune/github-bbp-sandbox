#include <immintrin.h>
#include <omp.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <limits>
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

static inline UpperResult upper_dot_row_nolut(
    const uint8_t* high, const float* h, int D) {
    __m256 acc = _mm256_setzero_ps();
    __m128i magmax = _mm_setzero_si128();
    const __m256 zero = _mm256_setzero_ps();
    const __m128i sign_threshold = _mm_set1_epi16(127);
    const __m128i suffix_mask = _mm_set1_epi16(0x00ff);
    const __m128i mag_mask = _mm_set1_epi16(0x007f);

    int j = 0;
    for (; j + 8 <= D; j += 8) {
        __m128i hb8 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(high + j));
        __m128i hb16 = _mm_cvtepu8_epi16(hb8);
        __m256 hv = _mm256_loadu_ps(h + j);

        // Weight-sign mask in 8x16-bit lanes.
        __m128i wneg16 = _mm_cmpgt_epi16(hb16, sign_threshold);
        // Hidden-sign mask: 8x32-bit -> 8x16-bit, preserving lane order.
        __m256 hneg_ps = _mm256_cmp_ps(hv, zero, _CMP_LT_OQ);
        __m256i hneg32 = _mm256_castps_si256(hneg_ps);
        __m128i hneg16 = _mm_packs_epi32(
            _mm256_castsi256_si128(hneg32),
            _mm256_extracti128_si256(hneg32, 1));

        // unread suffix FF exactly when signs match; otherwise 00.
        __m128i same16 = _mm_xor_si128(_mm_xor_si128(wneg16, hneg16), _mm_set1_epi16(-1));
        __m128i suffix16 = _mm_and_si128(same16, suffix_mask);
        __m128i raw_endpoint = _mm_or_si128(_mm_slli_epi16(hb16, 8), suffix16);
        __m256 endpoint = _mm256_cvtph_ps(raw_endpoint);
        acc = _mm256_fmadd_ps(endpoint, hv, acc);

        // Magnitude max needs no per-weight FP conversion: strip sign and max high code.
        __m128i mag = _mm_and_si128(hb16, mag_mask);
        magmax = _mm_max_epu16(magmax, mag);
    }

    float sum = hsum256(acc);
    alignas(16) uint16_t lanes[8];
    _mm_store_si128(reinterpret_cast<__m128i*>(lanes), magmax);
    uint16_t max_mag_hi = 0;
    for (int k = 0; k < 8; ++k) max_mag_hi = std::max(max_mag_hi, lanes[k]);

    for (; j < D; ++j) {
        const uint16_t hb = high[j];
        const int wneg = (hb >> 7) & 1;
        const int hneg = h[j] < 0.0f ? 1 : 0;
        const uint16_t suffix = (wneg == hneg) ? 0x00ff : 0x0000;
        const uint16_t raw = static_cast<uint16_t>((hb << 8) | suffix);
        sum = std::fma(_cvtsh_ss(raw), h[j], sum);
        max_mag_hi = std::max<uint16_t>(max_mag_hi, hb & 0x007f);
    }

    const uint16_t max_raw = static_cast<uint16_t>((max_mag_hi << 8) | 0x00ff);
    const float rowmax = _cvtsh_ss(max_raw);
    return {sum, rowmax};
}

static inline double gamma_n(int n, double u = std::ldexp(1.0, -24)) {
    const double x = static_cast<double>(n) * u;
    if (x >= 1.0) throw std::runtime_error("gamma invalid");
    return x / (1.0 - x);
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

struct ProofResult { int winner; int survivors; int low_rows; };

static void select_top_pilots(const std::vector<float>& upper, int k, std::vector<int>& pilots) {
    pilots.clear();
    pilots.reserve(k);
    for (int i = 0; i < static_cast<int>(upper.size()); ++i) {
        if (static_cast<int>(pilots.size()) < k) {
            pilots.push_back(i);
            for (int p = static_cast<int>(pilots.size())-1; p > 0 && upper[pilots[p]] > upper[pilots[p-1]]; --p)
                std::swap(pilots[p], pilots[p-1]);
        } else if (upper[i] > upper[pilots.back()]) {
            pilots.back() = i;
            for (int p = k-1; p > 0 && upper[pilots[p]] > upper[pilots[p-1]]; --p)
                std::swap(pilots[p], pilots[p-1]);
        }
    }
}

static ProofResult proofbits_argmax(
    const std::vector<uint8_t>& high, const std::vector<uint8_t>& low,
    const float* h, int V, int D, int pilot_k,
    std::vector<float>& upper, std::vector<int>& survivors,
    std::vector<int>& pilots) {
    // Conservative h-L1 upper bound. Inputs are FP32 exactly promoted to FP64.
    double h1 = 0.0;
    for (int j = 0; j < D; ++j) h1 += std::abs(static_cast<double>(h[j]));
    h1 /= (1.0 - gamma_n(D, std::ldexp(1.0, -53)));
    const double coeff = 2.0 * gamma_n(4 * D);

#pragma omp parallel for schedule(static)
    for (int i = 0; i < V; ++i) {
        auto r = upper_dot_row_nolut(high.data() + static_cast<size_t>(i) * D, h, D);
        const double safe64 = static_cast<double>(r.upper) + coeff * h1 * static_cast<double>(r.rowmax);
        float safe32 = static_cast<float>(safe64);
        if (static_cast<double>(safe32) < safe64)
            safe32 = std::nextafter(safe32, std::numeric_limits<float>::infinity());
        upper[i] = safe32;
    }

    select_top_pilots(upper, pilot_k, pilots);
    float B = -std::numeric_limits<float>::infinity();
    for (int p : pilots) {
        B = std::max(B, exact_dot_row(high.data() + static_cast<size_t>(p) * D,
                                      low.data() + static_cast<size_t>(p) * D, h, D));
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
                                          low.data() + static_cast<size_t>(i) * D, h, D);
            if (z > local_best || (z == local_best && (local_winner < 0 || i < local_winner))) {
                local_best = z; local_winner = i;
            }
        }
#pragma omp critical
        {
            if (local_best > best || (local_best == best && local_winner >= 0 && (winner < 0 || local_winner < winner))) {
                best = local_best; winner = local_winner;
            }
        }
    }

    int low_rows = static_cast<int>(survivors.size());
    for (int p : pilots) if (!std::binary_search(survivors.begin(), survivors.end(), p)) ++low_rows;
    return {winner, static_cast<int>(survivors.size()), low_rows};
}

struct Stats { double median_ms, p10_ms, p90_ms; };
static Stats summarize(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    auto q = [&](double p) { return v[static_cast<size_t>(p * (v.size()-1))]; };
    return {q(0.5), q(0.1), q(0.9)};
}

int main(int argc, char** argv) {
    Config c;
    for (int i=1;i<argc;++i) {
        std::string a=argv[i];
        auto val=[&](){ if(++i>=argc) throw std::runtime_error("missing arg"); return std::string(argv[i]); };
        if(a=="--dir") c.dir=val(); else if(a=="--vocab") c.V=std::stoi(val());
        else if(a=="--dim") c.D=std::stoi(val()); else if(a=="--states") c.N=std::stoi(val());
        else if(a=="--reps") c.reps=std::stoi(val()); else if(a=="--pilot-k") c.pilot_k=std::stoi(val());
    }
#if defined(__GNUC__)
    if (!__builtin_cpu_supports("avx2") || !__builtin_cpu_supports("f16c") || !__builtin_cpu_supports("fma")) {
        std::cerr << "CPU lacks AVX2/F16C/FMA\n"; return 3;
    }
#endif
    const size_t WN=static_cast<size_t>(c.V)*c.D, HN=static_cast<size_t>(c.N)*c.D;
    auto high=read_u8(c.dir+"/high.bin",WN); auto low=read_u8(c.dir+"/low.bin",WN); auto hidden=read_f32(c.dir+"/hidden.bin",HN);
    for(uint8_t hb:high) if(((hb>>2)&0x1f)==0x1f) throw std::runtime_error("non-finite high-byte pattern");

    std::vector<float> dense_scores(c.V), upper(c.V);
    std::vector<int> survivors; survivors.reserve(c.V); std::vector<int> pilots; pilots.reserve(c.pilot_k);
    std::vector<int> thread_counts={1}; int mt=omp_get_max_threads(); if(mt>=2) thread_counts.push_back(std::min(4,mt));
    std::sort(thread_counts.begin(),thread_counts.end()); thread_counts.erase(std::unique(thread_counts.begin(),thread_counts.end()),thread_counts.end());
    struct Row{int threads;Stats dense,proof;double speedup,mean_low,mean_surv;bool exact;}; std::vector<Row> rows;

    for(int threads:thread_counts){
        omp_set_num_threads(threads); bool ok=true; double low_sum=0,surv_sum=0;
        for(int n=0;n<c.N;++n){ const float* h=hidden.data()+static_cast<size_t>(n)*c.D; int d=dense_argmax(high,low,h,c.V,c.D,dense_scores); auto p=proofbits_argmax(high,low,h,c.V,c.D,c.pilot_k,upper,survivors,pilots); ok &= (d==p.winner); low_sum+=p.low_rows; surv_sum+=p.survivors; }
        if(!ok) throw std::runtime_error("ProofBits argmax mismatch");
        for(int n=0;n<c.N;++n){ const float* h=hidden.data()+static_cast<size_t>(n)*c.D; dense_argmax(high,low,h,c.V,c.D,dense_scores); proofbits_argmax(high,low,h,c.V,c.D,c.pilot_k,upper,survivors,pilots); }
        std::vector<double> td,tp;
        for(int r=0;r<c.reps;++r){
            auto t0=std::chrono::steady_clock::now();
            for(int n=0;n<c.N;++n){ const float* h=hidden.data()+static_cast<size_t>(n)*c.D; dense_argmax(high,low,h,c.V,c.D,dense_scores); }
            auto t1=std::chrono::steady_clock::now();
            for(int n=0;n<c.N;++n){ const float* h=hidden.data()+static_cast<size_t>(n)*c.D; proofbits_argmax(high,low,h,c.V,c.D,c.pilot_k,upper,survivors,pilots); }
            auto t2=std::chrono::steady_clock::now();
            td.push_back(std::chrono::duration<double,std::milli>(t1-t0).count()/c.N); tp.push_back(std::chrono::duration<double,std::milli>(t2-t1).count()/c.N);
        }
        Stats ds=summarize(td), ps=summarize(tp); rows.push_back({threads,ds,ps,ds.median_ms/ps.median_ms,low_sum/c.N,surv_sum/c.N,ok});
        std::cerr<<"threads="<<threads<<" dense="<<ds.median_ms<<"ms proof="<<ps.median_ms<<"ms speedup="<<ds.median_ms/ps.median_ms<<"x low_rows="<<low_sum/c.N<<"\n";
    }

    std::ofstream out("experiments/artifacts/proofbits_cpu_avx2_benchmark.json");
    out<<"{\n  \"kind\": \"matched_fp16_byteplane_avx2_lutfree_hardware_benchmark\",\n";
    out<<"  \"vocab\": "<<c.V<<", \"hidden_dim\": "<<c.D<<", \"states\": "<<c.N<<",\n  \"pilot_k\": "<<c.pilot_k<<",\n";
    out<<"  \"reference\": \"same FP16 high/low byte planes; AVX2/F16C exact reconstruction; float/FMA accumulation\",\n";
    out<<"  \"upper_path\": \"LUT-free: suffix FF iff hidden sign equals weight sign; row M_i from max signless high-byte magnitude code\",\n";
    out<<"  \"safe_certificate\": \"upper-only plus conservative 2*gamma_4d*||h||_1*M_i with outward FP32 rounding\",\n  \"results\": [\n";
    for(size_t i=0;i<rows.size();++i){ const auto&r=rows[i]; double f=r.mean_low/c.V, ideal=2.0/(1.0+f);
        out<<"    {\"threads\": "<<r.threads<<", \"exact_argmax\": "<<(r.exact?"true":"false")<<", \"dense_median_ms_per_state\": "<<r.dense.median_ms<<", \"dense_p10_ms\": "<<r.dense.p10_ms<<", \"dense_p90_ms\": "<<r.dense.p90_ms<<", \"proofbits_median_ms_per_state\": "<<r.proof.median_ms<<", \"proofbits_p10_ms\": "<<r.proof.p10_ms<<", \"proofbits_p90_ms\": "<<r.proof.p90_ms<<", \"measured_speedup\": "<<r.speedup<<", \"mean_survivors\": "<<r.mean_surv<<", \"mean_distinct_low_rows\": "<<r.mean_low<<", \"idealized_weight_byte_reduction\": "<<ideal<<"}"<<(i+1==rows.size()?"":",")<<"\n"; }
    out<<"  ],\n  \"caveat\": \"Real hosted-CPU wall clock, not GPU. Matched custom AVX2/F16C kernels are not a vendor BLAS baseline. GPU timing and DRAM counters remain required.\"\n}\n";
    return 0;
}
