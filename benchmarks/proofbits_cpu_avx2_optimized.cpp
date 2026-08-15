#include <immintrin.h>
#include <omp.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

struct Config { std::string dir="benchmarks/cpu_data"; int V=151936,D=896,N=16,reps=7,pilot_k=4; };

template<class T> static std::vector<T> read_bin(const std::string& p,size_t n){
    std::vector<T>x(n); std::ifstream f(p,std::ios::binary); size_t b=n*sizeof(T);
    if(!f||!f.read(reinterpret_cast<char*>(x.data()),static_cast<std::streamsize>(b))) throw std::runtime_error("read "+p);
    return x;
}
static inline float hsum256(__m256 x){__m128 hi=_mm256_extractf128_ps(x,1),lo=_mm256_castps256_ps128(x);__m128 s=_mm_add_ps(lo,hi);s=_mm_hadd_ps(s,s);s=_mm_hadd_ps(s,s);return _mm_cvtss_f32(s);}
static inline double gamma_n(int n,double u=std::ldexp(1.0,-24)){double x=(double)n*u;if(x>=1)throw std::runtime_error("gamma");return x/(1-x);}

static inline float exact_dot(const uint8_t* hi,const uint8_t* lo,const float* h,int D){
    __m256 acc=_mm256_setzero_ps(); int j=0;
    for(;j+8<=D;j+=8){
        __m128i hb8=_mm_loadl_epi64((const __m128i*)(hi+j));
        __m128i lb8=_mm_loadl_epi64((const __m128i*)(lo+j));
        __m128i hb16=_mm_cvtepu8_epi16(hb8),lb16=_mm_cvtepu8_epi16(lb8);
        __m128i raw=_mm_or_si128(_mm_slli_epi16(hb16,8),lb16);
        __m256 w=_mm256_cvtph_ps(raw),hv=_mm256_loadu_ps(h+j);
        acc=_mm256_fmadd_ps(w,hv,acc);
    }
    float s=hsum256(acc); for(;j<D;++j){uint16_t r=((uint16_t)hi[j]<<8)|lo[j];s=std::fma(_cvtsh_ss(r),h[j],s);} return s;
}

// hsign[j] is 0x80 iff h[j]<0, otherwise 0x00. This query invariant is hoisted.
static inline float upper_dot_hoisted(const uint8_t* hi,const uint8_t* hsign,const float* h,int D){
    __m256 acc=_mm256_setzero_ps(); int j=0;
    const __m128i bit7=_mm_set1_epi8((char)0x80),zero=_mm_setzero_si128();
    for(;j+8<=D;j+=8){
        __m128i hb8=_mm_loadl_epi64((const __m128i*)(hi+j));
        __m128i hs8=_mm_loadl_epi64((const __m128i*)(hsign+j));
        __m128i x=_mm_and_si128(_mm_xor_si128(hb8,hs8),bit7);
        __m128i same8=_mm_cmpeq_epi8(x,zero); // 0xff byte iff signs match
        __m128i hb16=_mm_cvtepu8_epi16(hb8);
        __m128i sf16=_mm_cvtepu8_epi16(same8);
        __m128i raw=_mm_or_si128(_mm_slli_epi16(hb16,8),sf16);
        __m256 w=_mm256_cvtph_ps(raw),hv=_mm256_loadu_ps(h+j);
        acc=_mm256_fmadd_ps(w,hv,acc);
    }
    float s=hsum256(acc); for(;j<D;++j){uint16_t suffix=(((hi[j]^hsign[j])&0x80)==0)?0xff:0;uint16_t r=((uint16_t)hi[j]<<8)|suffix;s=std::fma(_cvtsh_ss(r),h[j],s);} return s;
}

static void precompute_rowmax(const std::vector<uint8_t>& hi,int V,int D,std::vector<float>& rowmax){
    rowmax.resize(V);
#pragma omp parallel for schedule(static)
    for(int i=0;i<V;++i){uint8_t m=0;const uint8_t* p=hi.data()+(size_t)i*D;for(int j=0;j<D;++j)m=std::max<uint8_t>(m,p[j]&0x7f);rowmax[i]=_cvtsh_ss(((uint16_t)m<<8)|0xff);}
}
static void query_invariants(const float* h,int D,std::vector<uint8_t>& hsign,double& h1){
    hsign.resize(D);h1=0;for(int j=0;j<D;++j){hsign[j]=h[j]<0?0x80:0;h1+=std::abs((double)h[j]);}h1/=1.0-gamma_n(D,std::ldexp(1.0,-53));
}
static int dense_all(const std::vector<uint8_t>& hi,const std::vector<uint8_t>& lo,const float*h,int V,int D,std::vector<float>&z){
#pragma omp parallel for schedule(static)
    for(int i=0;i<V;++i)z[i]=exact_dot(hi.data()+(size_t)i*D,lo.data()+(size_t)i*D,h,D);
    return (int)(std::max_element(z.begin(),z.end())-z.begin());
}
static void select_pilots(const std::vector<float>&u,int k,std::vector<int>&p){
    p.clear();p.reserve(k);for(int i=0;i<(int)u.size();++i){if((int)p.size()<k){p.push_back(i);for(int q=p.size()-1;q>0&&u[p[q]]>u[p[q-1]];--q)std::swap(p[q],p[q-1]);}else if(u[i]>u[p.back()]){p.back()=i;for(int q=k-1;q>0&&u[p[q]]>u[p[q-1]];--q)std::swap(p[q],p[q-1]);}}
}
struct PB{int winner,survivors,lowrows;double upper_ms,pilot_ms,filter_ms,refine_ms;};
static PB proof(const std::vector<uint8_t>&hi,const std::vector<uint8_t>&lo,const std::vector<float>&rowmax,const float*h,int V,int D,int k,
                std::vector<uint8_t>&hsign,std::vector<float>&u,std::vector<int>&pilots,std::vector<int>&surv){
    double h1;query_invariants(h,D,hsign,h1);const double coeff=2.0*gamma_n(4*D);
    auto a=std::chrono::steady_clock::now();
#pragma omp parallel for schedule(static)
    for(int i=0;i<V;++i){float raw=upper_dot_hoisted(hi.data()+(size_t)i*D,hsign.data(),h,D);double s64=(double)raw+coeff*h1*(double)rowmax[i];float s=(float)s64;if((double)s<s64)s=std::nextafter(s,std::numeric_limits<float>::infinity());u[i]=s;}
    auto b=std::chrono::steady_clock::now();
    select_pilots(u,k,pilots);float B=-INFINITY;for(int p:pilots)B=std::max(B,exact_dot(hi.data()+(size_t)p*D,lo.data()+(size_t)p*D,h,D));
    auto c=std::chrono::steady_clock::now();
    surv.clear();for(int i=0;i<V;++i)if(u[i]>=B)surv.push_back(i);
    auto d=std::chrono::steady_clock::now();
    int win=-1;float best=-INFINITY;
#pragma omp parallel
    {int lw=-1;float lb=-INFINITY;
#pragma omp for nowait schedule(static)
    for(size_t q=0;q<surv.size();++q){int i=surv[q];float z=exact_dot(hi.data()+(size_t)i*D,lo.data()+(size_t)i*D,h,D);if(z>lb||(z==lb&&(lw<0||i<lw))){lb=z;lw=i;}}
#pragma omp critical
    {if(lb>best||(lb==best&&lw>=0&&(win<0||lw<win))){best=lb;win=lw;}}}
    auto e=std::chrono::steady_clock::now();
    int lowrows=surv.size();for(int p:pilots)if(!std::binary_search(surv.begin(),surv.end(),p))++lowrows;
    auto ms=[](auto x,auto y){return std::chrono::duration<double,std::milli>(y-x).count();};
    return{win,(int)surv.size(),lowrows,ms(a,b),ms(b,c),ms(c,d),ms(d,e)};
}
struct Stat{double med,p10,p90;};static Stat stat(std::vector<double>x){std::sort(x.begin(),x.end());auto q=[&](double p){return x[(size_t)(p*(x.size()-1))];};return{q(.5),q(.1),q(.9)};}

int main(int argc,char**argv){Config c;for(int i=1;i<argc;++i){std::string a=argv[i];auto v=[&](){if(++i>=argc)throw std::runtime_error("arg");return std::string(argv[i]);};if(a=="--dir")c.dir=v();else if(a=="--vocab")c.V=std::stoi(v());else if(a=="--dim")c.D=std::stoi(v());else if(a=="--states")c.N=std::stoi(v());else if(a=="--reps")c.reps=std::stoi(v());else if(a=="--pilot-k")c.pilot_k=std::stoi(v());}
#if defined(__GNUC__)
    if(!__builtin_cpu_supports("avx2")||!__builtin_cpu_supports("f16c")||!__builtin_cpu_supports("fma")){std::cerr<<"missing ISA\n";return 3;}
#endif
    auto hi=read_bin<uint8_t>(c.dir+"/high.bin",(size_t)c.V*c.D),lo=read_bin<uint8_t>(c.dir+"/low.bin",(size_t)c.V*c.D);auto H=read_bin<float>(c.dir+"/hidden.bin",(size_t)c.N*c.D);
    std::vector<float> rowmax;precompute_rowmax(hi,c.V,c.D,rowmax);
    std::vector<float> dz(c.V),u(c.V);std::vector<uint8_t>hsign(c.D);std::vector<int>pilots,surv;pilots.reserve(c.pilot_k);surv.reserve(c.V);
    std::vector<int>threads={1};int mt=omp_get_max_threads();if(mt>=2)threads.push_back(std::min(4,mt));
    std::ofstream out("experiments/artifacts/proofbits_cpu_avx2_optimized.json");out<<"{\n  \"kind\": \"hoisted_invariant_lutfree_avx2_proofbits\",\n  \"vocab\": "<<c.V<<", \"hidden_dim\": "<<c.D<<", \"states\": "<<c.N<<",\n  \"static_rowmax_metadata_bytes\": "<<(c.V*sizeof(float))<<",\n  \"results\": [\n";
    for(size_t ti=0;ti<threads.size();++ti){int th=threads[ti];omp_set_num_threads(th);bool ok=true;double low=0;for(int n=0;n<c.N;++n){const float*h=H.data()+(size_t)n*c.D;int d=dense_all(hi,lo,h,c.V,c.D,dz);PB p=proof(hi,lo,rowmax,h,c.V,c.D,c.pilot_k,hsign,u,pilots,surv);ok&=d==p.winner;low+=p.lowrows;}if(!ok)throw std::runtime_error("mismatch");
        std::vector<double>td,tp,tup,tpi,tfi,trf;
        for(int r=0;r<c.reps;++r){auto t0=std::chrono::steady_clock::now();for(int n=0;n<c.N;++n)dense_all(hi,lo,H.data()+(size_t)n*c.D,c.V,c.D,dz);auto t1=std::chrono::steady_clock::now();double pu=0,pp=0,pf=0,pr=0;for(int n=0;n<c.N;++n){PB p=proof(hi,lo,rowmax,H.data()+(size_t)n*c.D,c.V,c.D,c.pilot_k,hsign,u,pilots,surv);pu+=p.upper_ms;pp+=p.pilot_ms;pf+=p.filter_ms;pr+=p.refine_ms;}auto t2=std::chrono::steady_clock::now();td.push_back(std::chrono::duration<double,std::milli>(t1-t0).count()/c.N);tp.push_back(std::chrono::duration<double,std::milli>(t2-t1).count()/c.N);tup.push_back(pu/c.N);tpi.push_back(pp/c.N);tfi.push_back(pf/c.N);trf.push_back(pr/c.N);}
        Stat ds=stat(td),ps=stat(tp),us=stat(tup),pis=stat(tpi),fs=stat(tfi),rs=stat(trf);double fl=(low/c.N)/c.V,ideal=2.0/(1+fl);
        std::cerr<<"threads="<<th<<" dense="<<ds.med<<" proof="<<ps.med<<" speedup="<<ds.med/ps.med<<" upper="<<us.med<<" pilot="<<pis.med<<" filter="<<fs.med<<" refine="<<rs.med<<" low="<<low/c.N<<"\n";
        out<<"    {\"threads\": "<<th<<", \"exact\": true, \"dense_ms\": "<<ds.med<<", \"proof_ms\": "<<ps.med<<", \"speedup\": "<<ds.med/ps.med<<", \"upper_phase_ms\": "<<us.med<<", \"pilot_phase_ms\": "<<pis.med<<", \"filter_phase_ms\": "<<fs.med<<", \"refine_phase_ms\": "<<rs.med<<", \"mean_low_rows\": "<<low/c.N<<", \"idealized_weight_byte_reduction\": "<<ideal<<"}"<<(ti+1==threads.size()?"":",")<<"\n";
    }
    out<<"  ],\n  \"caveat\": \"Real hosted CPU. M_i is precomputed once as one float per vocabulary row; query hidden signs are hoisted once. Not a GPU result.\"\n}\n";}
