import gc
import json
import statistics
import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx_lm import load
from mlx_lm.models import cache as cache_mod

sys.path.insert(0,str(Path(__file__).resolve().parent))
import proofbits_mlx_integrated_decode as base
import proofbits_mlx_dual_bound as dual

MODEL='mlx-community/gemma-3-270m-bf16'
D=640
BLOCK=32
NB=D//BLOCK
CAP=3
PROMPTS=[
 'Explain why entropy is measured with logarithms.',
 'Solve step by step: if x + 1/x = 3, compute x^6 + 1/x^6.',
 'Write a Python function for longest increasing contiguous subarray.',
 'Explain why memory bandwidth matters for autoregressive inference.',
 'Describe natural selection without using teleological language.',
]
TOKENS=32
REPS=20

CODE63_SRC=f'''
 uint row=threadgroup_position_in_grid.x;
 uint lane=thread_index_in_simdgroup;
 ulong wbase=(ulong)row*{D}ul;
 ulong block_base=(ulong)row*{NB}ul;
 float acc=0.0f;
 for(uint blk=0; blk<{NB}u; ++blk){{
   ulong gb=block_base+(ulong)blk;
   uint of=(overflow_bits[gb>>5] >> (gb & 31ul)) & 1u;
   uint j=blk*32u+lane;
   uchar hb;
   if(of){{
     hb=high[wbase+j];
   }} else {{
     ulong cb=gb*6ul;
     uint bit=lane*6u;
     uint wi=bit>>5;
     uint sh=bit & 31u;
     uint code=codes[cb+(ulong)wi] >> sh;
     if(sh>26u) code |= codes[cb+(ulong)wi+1ul] << (32u-sh);
     code &= 63u;
     bool isesc=(code==63u);
     uint rank=simd_prefix_exclusive_sum((uint)isesc);
     hb=isesc ? escapes[gb*{CAP}ul+(ulong)rank] : codebook[code];
   }}
   ushort ws=(ushort)(hb & (uchar)0x80);
   ushort hs=(hidden[j]<0.0f)?(ushort)0x80:(ushort)0x00;
   ushort suffix=(ws==hs)?(ushort)0x00FF:(ushort)0x0000;
   ushort raw=((ushort)hb<<8)|suffix;
   acc=fma(hidden[j],(float)as_type<half>(raw),acc);
 }}
 float total=simd_sum(acc);
 if(lane==0)upper[row]=total;
'''


def med(x):return float(statistics.median(x))


def pack_code63(high_np):
    V,DD=high_np.shape; assert DD==D
    freq=np.bincount(high_np.reshape(-1),minlength=256)
    common=np.argsort(freq)[::-1][:63].astype(np.uint8)
    cmap=np.full(256,63,dtype=np.uint8)
    for i,v in enumerate(common): cmap[int(v)]=i
    nblocks=V*NB
    codes_out=np.empty((nblocks,6),dtype=np.uint32)
    escapes_out=np.zeros((nblocks,CAP),dtype=np.uint8)
    flags=np.zeros(nblocks,dtype=np.uint8)
    CH=1024
    boff=0
    for s in range(0,V,CH):
        e=min(V,s+CH); R=e-s
        hb=high_np[s:e].reshape(R,NB,32)
        c=cmap[hb]
        em=(c==63)
        cnt=em.sum(axis=2)
        flags[boff:boff+R*NB]=(cnt>CAP).reshape(-1).astype(np.uint8)
        ranks=np.cumsum(em,axis=2)-1
        esc=np.zeros((R,NB,CAP),dtype=np.uint8)
        for r in range(CAP):
            mask=em & (ranks==r)
            esc[:,:,r]=np.max(np.where(mask,hb,0),axis=2)
        escapes_out[boff:boff+R*NB]=esc.reshape(-1,CAP)
        packed=np.zeros((R,NB,6),dtype=np.uint32)
        cu=c.astype(np.uint32)
        for lane in range(32):
            bit=lane*6; wi=bit>>5; sh=bit&31
            v=cu[:,:,lane]
            packed[:,:,wi] |= v << sh
            if sh>26: packed[:,:,wi+1] |= v >> (32-sh)
        codes_out[boff:boff+R*NB]=packed.reshape(-1,6)
        boff += R*NB
    # one packed overflow flag bit per 32 blocks
    flag_bytes=np.packbits(flags,bitorder='little')
    pad=(-len(flag_bytes))%4
    if pad: flag_bytes=np.pad(flag_bytes,(0,pad))
    flag_words=flag_bytes.view(np.uint32)
    stats={'coverage':float(freq[common].sum()/freq.sum()),'overflow_fraction':float(flags.mean()),
           'codes_bytes':int(codes_out.nbytes),'escapes_bytes':int(escapes_out.nbytes),'overflow_bits_bytes':int(flag_words.nbytes),
           'compressed_upper_bytes':int(codes_out.nbytes+escapes_out.nbytes+flag_words.nbytes+common.nbytes),
           'raw_high_bytes':int(high_np.nbytes)}
    return mx.array(codes_out.reshape(-1)),mx.array(escapes_out.reshape(-1)),mx.array(flag_words),mx.array(common),stats


def make_kernel():
    return mx.fast.metal_kernel(name='pb_upper_code63_cap3',input_names=['codes','escapes','overflow_bits','codebook','high','hidden'],output_names=['upper'],source=CODE63_SRC)


def code_upper(k,pack,high,h,V):
    codes,esc,flags,book=pack
    return k(inputs=[codes,esc,flags,book,high,h],grid=(V*32,1,1),threadgroup=(32,1,1),output_shapes=[(V,)],output_dtypes=[mx.float32],init_value=0.0)[0]


def code_decision(k,pack,ks,high,low,h,V):
    U=code_upper(k,pack,high,h,V)
    p=mx.reshape(mx.argmax(U).astype(mx.uint32),(1,))
    B=ks[2](inputs=[high,low,h,p],grid=(32,1,1),threadgroup=(32,1,1),output_shapes=[(1,)],output_dtypes=[mx.float32],init_value=0.0)[0]
    E=ks[3](inputs=[high,low,h,U,B],grid=(V*32,1,1),threadgroup=(32,1,1),output_shapes=[(V,)],output_dtypes=[mx.float32],init_value=-3.402823466e38)[0]
    return mx.argmax(E).astype(mx.uint32)


def ids(tok,p):
    try:x=tok.encode(p)
    except Exception:x=tok(p)['input_ids']
    return mx.array(x,dtype=mx.int32)


def one_hidden(model,tok,p):
    kv=cache_mod.make_prompt_cache(model);b=model.model(ids(tok,p)[None],cache=kv);h=b[:,-1,:].reshape((D,)).astype(mx.float32);mx.eval(h,[c.state for c in kv]);return h


def stage_time(fn):
    xs=[]
    for _ in range(REPS):
        t=time.perf_counter();z=fn();mx.eval(z);xs.append((time.perf_counter()-t)*1e3)
    return med(xs)


def run(model,tok,prompt,mode,k,pack,ks,w16,high,low,n):
    V,DD=[int(x) for x in w16.shape];assert DD==D
    kv=cache_mod.make_prompt_cache(model);b=model.model(ids(tok,prompt)[None],cache=kv);h=b[:,-1,:].reshape((D,)).astype(mx.float32);mx.eval(h,[c.state for c in kv])
    toks=[];times=[]
    for _ in range(n):
        t=time.perf_counter()
        if mode=='dense':y=base.call_dense(ks[0],w16,h,V)
        elif mode=='native':y=base.call_native(w16,h)
        elif mode=='rawpb':y=base.call_proofbits(ks[1],ks[2],ks[3],high,low,h,V,False)
        elif mode=='code63':y=code_decision(k,pack,ks,high,low,h,V)
        else:raise ValueError(mode)
        mx.eval(y);token=int(y.item());toks.append(token)
        b=model.model(mx.array([[token]],dtype=mx.int32),cache=kv);h=b[:,-1,:].reshape((D,)).astype(mx.float32);mx.eval(h,[c.state for c in kv]);times.append((time.perf_counter()-t)*1e3)
    return {'tokens':toks,'median_ms':med(times),'mean_ms':float(statistics.mean(times))}


def main():
    model,tok=load(MODEL);model.set_dtype(mx.float16);mx.eval(model.parameters());base.MODEL=MODEL
    w16,high,low=dual.prepare(model);V,DD=[int(x) for x in w16.shape];assert DD==D
    high_np=np.array(high,copy=True).astype(np.uint8,copy=False)
    t0=time.perf_counter();codes,esc,flags,book,stats=pack_code63(high_np);mx.eval(codes,esc,flags,book);pack_s=time.perf_counter()-t0
    pack=(codes,esc,flags,book);k=make_kernel();ks=base.make_kernels();h=one_hidden(model,tok,PROMPTS[0])
    # compile and verify upper numerical identity on one state
    Ur=ks[1](inputs=[high,h],grid=(V*32,1,1),threadgroup=(32,1,1),output_shapes=[(V,)],output_dtypes=[mx.float32],init_value=0.0)[0]
    Uc=code_upper(k,pack,high,h,V);mx.eval(Ur,Uc)
    upper_max_abs=float(mx.max(mx.abs(Ur-Uc)).item())
    raw_upper_ms=stage_time(lambda:ks[1](inputs=[high,h],grid=(V*32,1,1),threadgroup=(32,1,1),output_shapes=[(V,)],output_dtypes=[mx.float32],init_value=0.0)[0])
    code_upper_ms=stage_time(lambda:code_upper(k,pack,high,h,V))
    modes=['dense','rawpb','code63','native']
    for m in modes:run(model,tok,PROMPTS[0],m,k,pack,ks,w16,high,low,4)
    mx.synchronize();rows=[]
    orders=[['dense','rawpb','code63','native'],['code63','native','dense','rawpb'],['native','rawpb','code63','dense'],['rawpb','dense','native','code63']]
    for i,p in enumerate(PROMPTS):
        order=orders[i%4];res={}
        for m in order:
            gc.collect();mx.clear_cache();res[m]=run(model,tok,p,m,k,pack,ks,w16,high,low,TOKENS)
        d,r,c,n=res['dense'],res['rawpb'],res['code63'],res['native']
        rows.append({'prompt_index':i,'order':order,'raw_exact':r['tokens']==d['tokens'],'code63_exact':c['tokens']==d['tokens'],'native_equal_code63':n['tokens']==c['tokens'],
                     'dense_ms':d['median_ms'],'rawpb_ms':r['median_ms'],'code63_ms':c['median_ms'],'native_ms':n['median_ms'],
                     'raw_over_code63':r['median_ms']/c['median_ms'],'native_over_code63':n['median_ms']/c['median_ms']})
    out={'kind':'proofbits_code63_lossless_upper_metal','model':MODEL,'cap':CAP,'pack_seconds':pack_s,'storage':stats,
         'upper_max_abs_diff':upper_max_abs,'raw_upper_ms':raw_upper_ms,'code63_upper_ms':code_upper_ms,'raw_upper_over_code63':raw_upper_ms/code_upper_ms,
         'rows':rows,'all_code63_exact':all(r['code63_exact'] for r in rows),'median_raw_over_code63':med([r['raw_over_code63'] for r in rows]),
         'mean_raw_over_code63':float(statistics.mean(r['raw_over_code63'] for r in rows)),'median_native_over_code63':med([r['native_over_code63'] for r in rows]),
         'min_native_over_code63':min(r['native_over_code63'] for r in rows),
         'note':'Lossless high-byte reconstruction. Top-63 symbols use 6-bit codes; code 63 is escape. 32-weight blocks with <=3 escapes store three fixed raw bytes; overflow blocks read the original raw high plane. Packed overflow flag bitset and all decode overhead are in timed Metal upper. Original high remains resident for rare overflow and later refinement, but common upper blocks do not read it.'}
    Path('experiments/artifacts').mkdir(parents=True,exist_ok=True);Path('experiments/artifacts/proofbits_mlx_code63_upper.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))

if __name__=='__main__':main()
