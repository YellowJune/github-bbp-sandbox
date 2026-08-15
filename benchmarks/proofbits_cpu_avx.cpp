#include <immintrin.h>
#include <omp.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

static std::vector<uint8_t> read_bytes(const std::string& p){
  std::ifstream f(p,std::ios::binary|std::ios::ate); if(!f) throw std::runtime_error("open "+p);
  auto n=f.tellg(); f.seekg(0); std::vector<uint8_t> b((size_t)n); f.read((char*)b.data(),n); return b;
}
static inline float hsum256(__m256 v){
  __m128 lo=_mm256_castps256_ps128(v), hi=_mm256_extractf128_ps(v,1);
  __m128 s=_mm_add_ps(lo,hi); s=_mm_hadd_ps(s,s); s=_mm_hadd_ps(s,s); return _mm_cvtss_f32(s);
}
static inline float dense_dot(const uint16_t* w,const float* h,int D){
  __m256 acc=_mm256_setzero_ps(); int j=0;
  for(;j+8<=D;j+=8){
    __m128i hb=_mm_loadu_si128((const __m128i*)(w+j));
    __m256 wf=_mm256_cvtph_ps(hb); __m256 hv=_mm256_loadu_ps(h+j);
    acc=_mm256_fmadd_ps(hv,wf,acc);
  }
  float s=hsum256(acc); for(;j<D;j++) s+=h[j]*_cvtsh_ss(w[j]); return s;
}
// LUT-free equivalent of the final GPU endpoint reconstruction.
// hsign[j] is 0x0080 when h_j<0 else 0. A high-byte weight sign is bit 7.
// Append suffix FF iff the two signs match, otherwise append 00.
static inline float upper_dot_direct(const uint8_t* high,const float* h,const uint16_t* hsign,int D){
  __m256 acc=_mm256_setzero_ps(); int j=0;
  const __m128i signmask=_mm_set1_epi16(0x0080);
  const __m128i ff=_mm_set1_epi16(0x00ff);
  for(;j+8<=D;j+=8){
    __m128i b8=_mm_loadl_epi64((const __m128i*)(high+j));
    __m128i b16=_mm_cvtepu8_epi16(b8);
    __m128i ws=_mm_and_si128(b16,signmask);
    __m128i hs=_mm_loadu_si128((const __m128i*)(hsign+j));
    __m128i eq=_mm_cmpeq_epi16(ws,hs);
    __m128i suffix=_mm_and_si128(eq,ff);
    __m128i raw=_mm_or_si128(_mm_slli_epi16(b16,8),suffix);
    __m256 ep=_mm256_cvtph_ps(raw); __m256 hv=_mm256_loadu_ps(h+j);
    acc=_mm256_fmadd_ps(hv,ep,acc);
  }
  float s=hsum256(acc);
  for(;j<D;j++){
    uint16_t ws=((uint16_t)high[j])&0x80u;
    uint16_t suffix=(ws==hsign[j])?0xffu:0u;
    uint16_t raw=((uint16_t)high[j]<<8)|suffix;
    s+=h[j]*_cvtsh_ss(raw);
  }
  return s;
}
static void flush_cache(std::vector<uint8_t>& f){
  volatile uint64_t x=0; for(size_t i=0;i<f.size();i+=64) x+=f[i]; if(x==123456789) std::cerr<<x;
}
struct R{double ms;int winner;int survivors;};
static R run_dense(const uint16_t* full,const float* h,int V,int D){
  std::vector<float> z(V);
  auto t0=std::chrono::steady_clock::now();
  #pragma omp parallel for schedule(static)
  for(int i=0;i<V;i++) z[i]=dense_dot(full+(size_t)i*D,h,D);
  int win=0; for(int i=1;i<V;i++) if(z[i]>z[win])win=i;
  auto t1=std::chrono::steady_clock::now(); return {std::chrono::duration<double,std::milli>(t1-t0).count(),win,V};
}
static R run_pb(const uint16_t* full,const uint8_t* high,const float* h,int V,int D){
  std::vector<uint16_t> hs(D); for(int j=0;j<D;j++)hs[j]=(h[j]<0?0x0080u:0u);
  std::vector<float> U(V);
  auto t0=std::chrono::steady_clock::now();
  #pragma omp parallel for schedule(static)
  for(int i=0;i<V;i++) U[i]=upper_dot_direct(high+(size_t)i*D,h,hs.data(),D);
  int p=0; for(int i=1;i<V;i++)if(U[i]>U[p])p=i;
  float B=dense_dot(full+(size_t)p*D,h,D);
  std::vector<int> S; S.reserve(2048); for(int i=0;i<V;i++)if(U[i]>=B)S.push_back(i);
  int win=S.empty()?p:S[0]; float best=-INFINITY;
  #pragma omp parallel
  {
    float lbest=-INFINITY; int lwin=p;
    #pragma omp for nowait schedule(static)
    for(size_t k=0;k<S.size();k++){int i=S[k];float z=dense_dot(full+(size_t)i*D,h,D);if(z>lbest){lbest=z;lwin=i;}}
    #pragma omp critical
    {if(lbest>best){best=lbest;win=lwin;}}
  }
  auto t1=std::chrono::steady_clock::now(); return {std::chrono::duration<double,std::milli>(t1-t0).count(),win,(int)S.size()};
}
int main(int argc,char**argv){
  if(argc<6){std::cerr<<"usage: exe dir V D N threads_csv\n";return 2;}
  std::string dir=argv[1]; int V=std::stoi(argv[2]),D=std::stoi(argv[3]),N=std::stoi(argv[4]); std::string ts=argv[5];
  auto fb=read_bytes(dir+"/full_u16.bin"), hb=read_bytes(dir+"/high_u8.bin"), hh=read_bytes(dir+"/hidden_f32.bin");
  if(fb.size()!=(size_t)V*D*2||hb.size()!=(size_t)V*D||hh.size()!=(size_t)N*D*4) throw std::runtime_error("size mismatch");
  const uint16_t* full=(const uint16_t*)fb.data(); const uint8_t* high=hb.data(); const float* H=(const float*)hh.data();
  std::vector<int> threads; size_t pos=0; while(pos<ts.size()){size_t q=ts.find(',',pos);threads.push_back(std::stoi(ts.substr(pos,q-pos)));if(q==std::string::npos)break;pos=q+1;}
  std::vector<uint8_t> flush(512ull*1024*1024,1);
  std::cout<<"{\n  \"kind\":\"proofbits_cpu_avx_direct_microbench\",\n  \"V\":"<<V<<",\n  \"D\":"<<D<<",\n  \"N\":"<<N<<",\n  \"results\":[\n";
  bool firstT=true;
  for(int nt:threads){
    omp_set_num_threads(nt); std::vector<double> dt,pt; std::vector<int> sc; int exact=0;
    run_dense(full,H,V,D);run_pb(full,high,H,V,D);
    for(int n=0;n<N;n++){
      const float* h=H+(size_t)n*D;
      flush_cache(flush); auto d=run_dense(full,h,V,D);
      flush_cache(flush); auto p=run_pb(full,high,h,V,D);
      dt.push_back(d.ms);pt.push_back(p.ms);sc.push_back(p.survivors);exact+=(d.winner==p.winner);
    }
    auto med=[](std::vector<double> x){std::sort(x.begin(),x.end());return x[x.size()/2];};
    double dm=med(dt),pm=med(pt); double meanS=std::accumulate(sc.begin(),sc.end(),0.0)/sc.size();
    if(!firstT)std::cout<<",\n";firstT=false;
    std::cout<<"    {\"threads\":"<<nt<<",\"dense_median_ms\":"<<dm<<",\"proofbits_median_ms\":"<<pm<<",\"speedup\":"<<(dm/pm)<<",\"exact_queries\":"<<exact<<",\"survivor_mean\":"<<meanS<<",\"survivor_fraction\":"<<(meanS/V)<<"}";
  }
  std::cout<<"\n  ],\n  \"caveat\":\"Shared GitHub CPU runner microbenchmark. Dense uses AVX2/F16C contiguous FP16. ProofBits reconstructs the final high-byte interval endpoint directly with AVX2 integer operations and F16C, with no LUT/gather. This is a CPU feasibility signal, not a GPU performance proxy. Cache is flushed with 512MB before each timed query.\"\n}\n";
}
