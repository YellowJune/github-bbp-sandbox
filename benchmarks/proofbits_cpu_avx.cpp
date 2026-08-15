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
static inline float upper_dot(const uint8_t* high,const float* h,const int32_t* signoff,const float* lut,int D){
  __m256 acc=_mm256_setzero_ps(); int j=0;
  for(;j+8<=D;j+=8){
    __m128i b8=_mm_loadl_epi64((const __m128i*)(high+j));
    __m256i idx=_mm256_cvtepu8_epi32(b8);
    idx=_mm256_add_epi32(idx,_mm256_loadu_si256((const __m256i*)(signoff+j)));
    __m256 ep=_mm256_i32gather_ps(lut,idx,4); __m256 hv=_mm256_loadu_ps(h+j);
    acc=_mm256_fmadd_ps(hv,ep,acc);
  }
  float s=hsum256(acc);
  for(;j<D;j++) s+=h[j]*lut[(int)high[j]+signoff[j]];
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
static R run_pb(const uint16_t* full,const uint8_t* high,const float* h,int V,int D,const float* lut){
  std::vector<int32_t> so(D); for(int j=0;j<D;j++)so[j]=(h[j]<0?256:0);
  std::vector<float> U(V);
  auto t0=std::chrono::steady_clock::now();
  #pragma omp parallel for schedule(static)
  for(int i=0;i<V;i++) U[i]=upper_dot(high+(size_t)i*D,h,so.data(),lut,D);
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
  alignas(32) float lut[512];
  for(int hs=0;hs<2;hs++)for(int x=0;x<256;x++){
    int ws=(x>>7)&1; uint16_t suffix=(ws==hs)?255:0; uint16_t raw=((uint16_t)x<<8)|suffix; lut[hs*256+x]=_cvtsh_ss(raw);
  }
  std::vector<int> threads; size_t pos=0; while(pos<ts.size()){size_t q=ts.find(',',pos);threads.push_back(std::stoi(ts.substr(pos,q-pos)));if(q==std::string::npos)break;pos=q+1;}
  std::vector<uint8_t> flush(512ull*1024*1024,1);
  std::cout<<"{\n  \"kind\":\"proofbits_cpu_avx_microbench\",\n  \"V\":"<<V<<",\n  \"D\":"<<D<<",\n  \"N\":"<<N<<",\n  \"results\":[\n";
  bool firstT=true;
  for(int nt:threads){
    omp_set_num_threads(nt); std::vector<double> dt,pt; std::vector<int> sc; int exact=0;
    // One untimed warm query per path.
    run_dense(full,H,V,D);run_pb(full,high,H,V,D,lut);
    for(int n=0;n<N;n++){
      const float* h=H+(size_t)n*D;
      flush_cache(flush); auto d=run_dense(full,h,V,D);
      flush_cache(flush); auto p=run_pb(full,high,h,V,D,lut);
      dt.push_back(d.ms);pt.push_back(p.ms);sc.push_back(p.survivors);exact+=(d.winner==p.winner);
    }
    auto med=[](std::vector<double> x){std::sort(x.begin(),x.end());return x[x.size()/2];};
    double dm=med(dt),pm=med(pt); double meanS=std::accumulate(sc.begin(),sc.end(),0.0)/sc.size();
    if(!firstT)std::cout<<",\n";firstT=false;
    std::cout<<"    {\"threads\":"<<nt<<",\"dense_median_ms\":"<<dm<<",\"proofbits_median_ms\":"<<pm<<",\"speedup\":"<<(dm/pm)<<",\"exact_queries\":"<<exact<<",\"survivor_mean\":"<<meanS<<",\"survivor_fraction\":"<<(meanS/V)<<"}";
  }
  std::cout<<"\n  ],\n  \"caveat\":\"Shared GitHub CPU runner microbenchmark. Dense uses AVX2/F16C contiguous FP16; ProofBits high pass uses AVX2 gather from a 2KB endpoint LUT plus high-byte plane. This is a CPU feasibility signal, not a GPU performance proxy. Cache is flushed with a 512MB buffer before each timed query.\"\n}\n";
}
