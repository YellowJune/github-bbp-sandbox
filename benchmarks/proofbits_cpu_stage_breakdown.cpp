#include <immintrin.h>
#include <omp.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

static std::vector<uint8_t> read_bytes(const std::string&p){std::ifstream f(p,std::ios::binary|std::ios::ate);if(!f)throw std::runtime_error("open");auto n=f.tellg();f.seekg(0);std::vector<uint8_t>b((size_t)n);f.read((char*)b.data(),n);return b;}
static inline float hsum(__m256 v){__m128 lo=_mm256_castps256_ps128(v),hi=_mm256_extractf128_ps(v,1);__m128 s=_mm_add_ps(lo,hi);s=_mm_hadd_ps(s,s);s=_mm_hadd_ps(s,s);return _mm_cvtss_f32(s);}
static inline float dense_dot(const uint16_t*w,const float*h,int D){__m256 a=_mm256_setzero_ps();int j=0;for(;j+8<=D;j+=8){__m128i r=_mm_loadu_si128((const __m128i*)(w+j));__m256 wf=_mm256_cvtph_ps(r),hv=_mm256_loadu_ps(h+j);a=_mm256_fmadd_ps(hv,wf,a);}float s=hsum(a);for(;j<D;j++)s+=h[j]*_cvtsh_ss(w[j]);return s;}
static inline float upper_dot(const uint8_t*hi,const float*h,const uint16_t*hs,int D){__m256 a=_mm256_setzero_ps();int j=0;const __m128i sm=_mm_set1_epi16(0x80),ff=_mm_set1_epi16(0xff);for(;j+8<=D;j+=8){__m128i b8=_mm_loadl_epi64((const __m128i*)(hi+j));__m128i b16=_mm_cvtepu8_epi16(b8);__m128i ws=_mm_and_si128(b16,sm),hh=_mm_loadu_si128((const __m128i*)(hs+j));__m128i suf=_mm_and_si128(_mm_cmpeq_epi16(ws,hh),ff);__m128i raw=_mm_or_si128(_mm_slli_epi16(b16,8),suf);__m256 ep=_mm256_cvtph_ps(raw),hv=_mm256_loadu_ps(h+j);a=_mm256_fmadd_ps(hv,ep,a);}float s=hsum(a);for(;j<D;j++){uint16_t suf=((((uint16_t)hi[j])&0x80)==hs[j])?0xff:0;uint16_t raw=((uint16_t)hi[j]<<8)|suf;s+=h[j]*_cvtsh_ss(raw);}return s;}
static void dense_pass(const uint16_t*full,const float*h,float*z,int V,int D){
#pragma omp parallel for schedule(static)
 for(int i=0;i<V;i++)z[i]=dense_dot(full+(size_t)i*D,h,D);
}
static void high_pass(const uint8_t*high,const float*h,const uint16_t*hs,float*U,int V,int D){
#pragma omp parallel for schedule(static)
 for(int i=0;i<V;i++)U[i]=upper_dot(high+(size_t)i*D,h,hs,D);
}
static int refine_pass(const uint16_t*full,const float*h,const std::vector<int>&S,int D,int p){
 float best=-INFINITY;int win=p;
#pragma omp parallel
 {
  float lb=-INFINITY;int lw=p;
#pragma omp for nowait schedule(static)
  for(size_t k=0;k<S.size();k++){int i=S[k];float q=dense_dot(full+(size_t)i*D,h,D);if(q>lb){lb=q;lw=i;}}
#pragma omp critical
  {if(lb>best){best=lb;win=lw;}}
 }
 return win;
}
template<class F> static double ms(F fn){auto t0=std::chrono::steady_clock::now();fn();auto t1=std::chrono::steady_clock::now();return std::chrono::duration<double,std::milli>(t1-t0).count();}
static double med(std::vector<double>x){std::sort(x.begin(),x.end());return x[x.size()/2];}
int main(int argc,char**argv){if(argc<6){std::cerr<<"dir V D N threads\n";return 2;}std::string dir=argv[1];int V=std::stoi(argv[2]),D=std::stoi(argv[3]),N=std::stoi(argv[4]),nt=std::stoi(argv[5]);omp_set_num_threads(nt);auto fb=read_bytes(dir+"/full_u16.bin"),hb=read_bytes(dir+"/high_u8.bin"),hh=read_bytes(dir+"/hidden_f32.bin");const uint16_t*full=(const uint16_t*)fb.data();const uint8_t*high=hb.data();const float*H=(const float*)hh.data();std::vector<float> z(V),U(V);std::vector<uint16_t>hs(D);std::vector<int>S;std::vector<double>td,th,ta,tt,tr,tp;int exact=0;double meanS=0;
 for(int n=0;n<N;n++){const float*h=H+(size_t)n*D;for(int j=0;j<D;j++)hs[j]=(h[j]<0?0x80:0);
  td.push_back(ms([&]{dense_pass(full,h,z.data(),V,D);}));
  int dwin=0;for(int i=1;i<V;i++)if(z[i]>z[dwin])dwin=i;
  th.push_back(ms([&]{high_pass(high,h,hs.data(),U.data(),V,D);}));
  int p=0;ta.push_back(ms([&]{for(int i=1;i<V;i++)if(U[i]>U[p])p=i;}));
  float B=0;tp.push_back(ms([&]{B=dense_dot(full+(size_t)p*D,h,D);}));
  tt.push_back(ms([&]{S.clear();S.reserve(2048);for(int i=0;i<V;i++)if(U[i]>=B)S.push_back(i);}));meanS+=S.size();
  int win=p;tr.push_back(ms([&]{win=refine_pass(full,h,S,D,p);}));exact+=(win==dwin);
 }
 double d=med(td),h=med(th),a=med(ta),p=med(tp),t=med(tt),r=med(tr);
 std::cout<<"{\"kind\":\"proofbits_cpu_stage_breakdown\",\"threads\":"<<nt<<",\"exact_queries\":"<<exact<<",\"survivor_mean\":"<<(meanS/N)<<",\"dense_pass_ms\":"<<d<<",\"high_pass_ms\":"<<h<<",\"high_vs_dense_speedup\":"<<(d/h)<<",\"argmax_U_ms\":"<<a<<",\"pilot_exact_ms\":"<<p<<",\"threshold_compact_ms\":"<<t<<",\"survivor_refine_ms\":"<<r<<",\"sum_pb_stage_medians_ms\":"<<(h+a+p+t+r)<<"}\n";
}
