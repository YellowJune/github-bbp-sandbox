import gc
import json
import statistics
import sys
import time
from pathlib import Path

import mlx.core as mx
from mlx_lm import load
from mlx_lm.models import cache as cache_mod

sys.path.insert(0,str(Path(__file__).resolve().parent))
import proofbits_mlx_integrated_decode as base
import proofbits_mlx_dual_bound as dual

MODEL='mlx-community/gemma-3-270m-bf16'
D=640
FACTORS=[1,2,4,5,10,20]
PROMPTS=[
 'Explain why entropy is measured with logarithms.',
 'Solve step by step: if x + 1/x = 3, compute x^6 + 1/x^6.',
 'Write a Python function for longest increasing contiguous subarray.',
 'Explain why memory bandwidth matters for autoregressive inference.',
 'Describe natural selection without using teleological language.',
 'Compare TCP and UDP for a latency-sensitive application.',
 'Give a concise proof that the square root of 2 is irrational.',
 'Explain the difference between correlation and causation.',
 'Write pseudocode for breadth-first search on a graph.',
 'Summarize why cache locality matters in matrix computation.',
 'Explain photosynthesis to a high-school student.',
 'Derive the quadratic formula from completing the square.',
]
STATES_PER_PROMPT=8
REPEATS=3


def med(x): return float(statistics.median(x))


def make_src(f):
    assert 20%f==0
    if f==1:
        return f'''
 uint row=threadgroup_position_in_grid.x; uint lane=thread_index_in_simdgroup;
 ulong base=(ulong)row*{D}ul; float acc=0.0f;
 for(uint j=lane;j<{D}u;j+=32u){{
   uchar hb=high[base+j]; ushort ws=(ushort)(hb & (uchar)0x80); ushort hs=(hidden[j]<0.0f)?(ushort)0x80:(ushort)0x00;
   ushort suffix=(ws==hs)?(ushort)0x00FF:(ushort)0x0000; ushort raw=((ushort)hb<<8)|suffix;
   acc=fma(hidden[j],(float)as_type<half>(raw),acc);
 }} float total=simd_sum(acc); if(lane==0)upper[row]=total;
'''
    chunks=[]
    for t in range(f):
        off=32*t
        chunks.append(f'''
   {{ uint j=basej+{off}u; uchar hb=high[base+j]; ushort ws=(ushort)(hb & (uchar)0x80); ushort hs=(hidden[j]<0.0f)?(ushort)0x80:(ushort)0x00;
      ushort suffix=(ws==hs)?(ushort)0x00FF:(ushort)0x0000; ushort raw=((ushort)hb<<8)|suffix;
      acc=fma(hidden[j],(float)as_type<half>(raw),acc); }}''')
    nouter=20//f
    if nouter==1:
        body='\n'.join(s.replace('basej', 'lane') for s in chunks)
        return f'''
 uint row=threadgroup_position_in_grid.x; uint lane=thread_index_in_simdgroup;
 ulong base=(ulong)row*{D}ul; float acc=0.0f;
 {body}
 float total=simd_sum(acc); if(lane==0)upper[row]=total;
'''
    return f'''
 uint row=threadgroup_position_in_grid.x; uint lane=thread_index_in_simdgroup;
 ulong base=(ulong)row*{D}ul; float acc=0.0f;
 for(uint basej=lane;basej<{D}u;basej+={32*f}u){{
 {''.join(chunks)}
 }} float total=simd_sum(acc); if(lane==0)upper[row]=total;
'''


def make_kernels():
    return {f:mx.fast.metal_kernel(name=f'pb_upper_u{f}',input_names=['high','hidden'],output_names=['upper'],source=make_src(f)) for f in FACTORS}


def pb_decision(upper_k,ks,high,low,h,V):
    U=upper_k(inputs=[high,h],grid=(V*32,1,1),threadgroup=(32,1,1),output_shapes=[(V,)],output_dtypes=[mx.float32],init_value=0.0)[0]
    p=mx.reshape(mx.argmax(U).astype(mx.uint32),(1,))
    B=ks[2](inputs=[high,low,h,p],grid=(32,1,1),threadgroup=(32,1,1),output_shapes=[(1,)],output_dtypes=[mx.float32],init_value=0.0)[0]
    E=ks[3](inputs=[high,low,h,U,B],grid=(V*32,1,1),threadgroup=(32,1,1),output_shapes=[(V,)],output_dtypes=[mx.float32],init_value=-3.402823466e38)[0]
    return mx.argmax(E).astype(mx.uint32)


def ids(tok,p):
    try:x=tok.encode(p)
    except Exception:x=tok(p)['input_ids']
    return mx.array(x,dtype=mx.int32)


def collect_states(model,tok):
    states=[]
    for pi,p in enumerate(PROMPTS):
        kv=cache_mod.make_prompt_cache(model)
        b=model.model(ids(tok,p)[None],cache=kv)
        h=b[:,-1,:].reshape((D,)).astype(mx.float32);mx.eval(h,[c.state for c in kv])
        for s in range(STATES_PER_PROMPT):
            states.append((pi,s,h))
            # Use native MLX only to advance a realistic fixed trajectory. All
            # head candidates later benchmark the same cached h tensors.
            y=base.call_native(model.lm_head.weight,h);mx.eval(y);token=int(y.item())
            b=model.model(mx.array([[token]],dtype=mx.int32),cache=kv)
            h=b[:,-1,:].reshape((D,)).astype(mx.float32);mx.eval(h,[c.state for c in kv])
    return states


def timed_decision(fn):
    t=time.perf_counter(); y=fn(); mx.eval(y); return int(y.item()),(time.perf_counter()-t)*1e3


def main():
    model,tok=load(MODEL);model.set_dtype(mx.float16);mx.eval(model.parameters());base.MODEL=MODEL
    w16,high,low=dual.prepare(model);V,DD=[int(x) for x in w16.shape];assert DD==D
    ks=base.make_kernels();ups=make_kernels();states=collect_states(model,tok)
    # compile every path
    h0=states[0][2]
    for f in FACTORS: y=pb_decision(ups[f],ks,high,low,h0,V);mx.eval(y)
    yd=base.call_dense(ks[0],w16,h0,V);yn=base.call_native(w16,h0);mx.eval(yd,yn);mx.synchronize()

    rows=[]; pooled={f:0.0 for f in FACTORS}; dense_total=0.0;native_total=0.0
    for idx,(pi,si,h) in enumerate(states):
        # Matched dense decision once per state for correctness.
        dy,dt=timed_decision(lambda:base.call_dense(ks[0],w16,h,V));dense_total+=dt
        ny,nt=timed_decision(lambda:base.call_native(w16,h));native_total+=nt
        times={f:[] for f in FACTORS}; decisions={f:[] for f in FACTORS}
        for r in range(REPEATS):
            order=FACTORS if (idx+r)%2==0 else list(reversed(FACTORS))
            for f in order:
                y,t=timed_decision(lambda f=f:pb_decision(ups[f],ks,high,low,h,V))
                decisions[f].append(y);times[f].append(t);pooled[f]+=t
        row={'state_index':idx,'prompt_index':pi,'decode_position':si,'dense_token':dy,'native_token':ny,'dense_ms':dt,'native_ms':nt}
        for f in FACTORS:
            row[f'u{f}_exact']=all(y==dy for y in decisions[f])
            row[f'u{f}_median_ms']=med(times[f])
            row[f'u1_over_u{f}']=med(times[1])/med(times[f])
            row[f'native_over_u{f}']=nt/med(times[f])
        rows.append(row)
    out={'kind':'proofbits_partial_unroll_headstress','model':MODEL,'D':D,'factors':FACTORS,'n_states':len(states),'repeats':REPEATS,'rows':rows}
    for f in FACTORS:
        out[f'u{f}_all_exact']=all(r[f'u{f}_exact'] for r in rows)
        out[f'u{f}_pooled_ms']=pooled[f]
        out[f'u1_over_u{f}_pooled']=pooled[1]/pooled[f]
        out[f'u1_over_u{f}_state_median']=med([r[f'u1_over_u{f}'] for r in rows])
        out[f'native_over_u{f}_state_median']=med([r[f'native_over_u{f}'] for r in rows])
    out['dense_total_ms_singlepass']=dense_total;out['native_total_ms_singlepass']=native_total
    out['note']='Head-only paired benchmark on 96 identical natural hidden states. Each partial-unroll factor preserves lane assignment and FMA order. AB/BA factor order alternates by state/repeat. Body/KV-cache timing is excluded after state collection.'
    Path('experiments/artifacts').mkdir(parents=True,exist_ok=True)
    Path('experiments/artifacts/proofbits_mlx_partial_unroll_headstress.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))

if __name__=='__main__':main()
