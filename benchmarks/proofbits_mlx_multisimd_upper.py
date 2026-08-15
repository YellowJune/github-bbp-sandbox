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

MODEL='mlx-community/gemma-3-270m-bf16'
PROMPTS=[
 'Explain why entropy is measured with logarithms.',
 'Solve step by step: if x + 1/x = 3, compute x^6 + 1/x^6.',
 'Write a Python function for longest increasing contiguous subarray.',
 'Explain why memory bandwidth matters for autoregressive inference.',
 'Describe natural selection without using teleological language.',
]
TOKENS=32
REPS=20


def upper_src(TG,NG):
 return f'''
    uint row = threadgroup_position_in_grid.x;
    uint tid = thread_index_in_threadgroup;
    uint lane = thread_index_in_simdgroup;
    uint sg = simdgroup_index_in_threadgroup;
    uint D = (uint)hidden_shape[0];
    ulong base = (ulong)row * (ulong)D;
    threadgroup float partial[{NG}];
    float acc = 0.0f;
    for (uint j = tid; j < D; j += {TG}u) {{
        uchar hb = high[base + j];
        ushort weightSign = (ushort)(hb & (uchar)0x80);
        ushort hiddenSign = (hidden[j] < 0.0f) ? (ushort)0x80 : (ushort)0x00;
        ushort suffix = (weightSign == hiddenSign) ? (ushort)0x00FF : (ushort)0x0000;
        ushort raw = ((ushort)hb << 8) | suffix;
        half endpoint = as_type<half>(raw);
        acc = fma(hidden[j], (float)endpoint, acc);
    }}
    float subtotal = simd_sum(acc);
    if (lane == 0) partial[sg] = subtotal;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {{
        float total = 0.0f;
        for (uint k=0;k<{NG}u;++k) total += partial[k];
        upper[row] = total;
    }}
 '''

def make_upper(name,TG):
 return mx.fast.metal_kernel(name=name,input_names=['high','hidden'],output_names=['upper'],source=upper_src(TG,TG//32))

def med(x):return float(statistics.median(x))

def prepare(model):
 w=model.lm_head.weight if hasattr(model,'lm_head') else model.model.embed_tokens.weight
 w16=w.astype(mx.float16);mx.eval(w16)
 a=np.array(w16,copy=True).astype(np.float16,copy=False);bits=a.view(np.uint16)
 high=mx.array((bits>>8).astype(np.uint8,copy=True));low=mx.array((bits&255).astype(np.uint8,copy=True));mx.eval(high,low)
 return w16,high,low

def call_pb(upper_k,TG,pilot_k,refine_k,high,low,h,V):
 U=upper_k(inputs=[high,h],grid=(V*TG,1,1),threadgroup=(TG,1,1),output_shapes=[(V,)],output_dtypes=[mx.float32],init_value=0.0)[0]
 p=mx.reshape(mx.argmax(U).astype(mx.uint32),(1,))
 B=pilot_k(inputs=[high,low,h,p],grid=(32,1,1),threadgroup=(32,1,1),output_shapes=[(1,)],output_dtypes=[mx.float32],init_value=0.0)[0]
 E=refine_k(inputs=[high,low,h,U,B],grid=(V*32,1,1),threadgroup=(32,1,1),output_shapes=[(V,)],output_dtypes=[mx.float32],init_value=-3.402823466e38)[0]
 return mx.argmax(E).astype(mx.uint32)

def ids(tok,p):
 try:x=tok.encode(p)
 except Exception:x=tok(p)['input_ids']
 return mx.array(x,dtype=mx.int32)

def one_hidden(model,tok,p,D):
 kv=cache_mod.make_prompt_cache(model);b=model.model(ids(tok,p)[None],cache=kv);h=b[:,-1,:].reshape((D,)).astype(mx.float32);mx.eval(h,[c.state for c in kv]);return h

def stage_time(k,TG,high,h,V):
 xs=[]
 for _ in range(REPS):
  t=time.perf_counter();U=k(inputs=[high,h],grid=(V*TG,1,1),threadgroup=(TG,1,1),output_shapes=[(V,)],output_dtypes=[mx.float32],init_value=0.0)[0];mx.eval(U);xs.append((time.perf_counter()-t)*1e3)
 return med(xs)

def run(model,tok,prompt,mode,uppers,baseks,w16,high,low,n):
 V,D=[int(x) for x in w16.shape];kv=cache_mod.make_prompt_cache(model);b=model.model(ids(tok,prompt)[None],cache=kv);h=b[:,-1,:].reshape((D,)).astype(mx.float32);mx.eval(h,[c.state for c in kv]);toks=[];times=[]
 for _ in range(n):
  t=time.perf_counter()
  if mode=='dense':y=base.call_dense(baseks[0],w16,h,V)
  elif mode=='native':y=base.call_native(w16,h)
  else:
   TG=int(mode[2:]);y=call_pb(uppers[TG],TG,baseks[2],baseks[3],high,low,h,V)
  mx.eval(y);token=int(y.item());toks.append(token);b=model.model(mx.array([[token]],dtype=mx.int32),cache=kv);h=b[:,-1,:].reshape((D,)).astype(mx.float32);mx.eval(h,[c.state for c in kv]);times.append((time.perf_counter()-t)*1e3)
 return {'tokens':toks,'median_ms':med(times),'mean_ms':float(statistics.mean(times))}

def main():
 model,tok=load(MODEL);model.set_dtype(mx.float16);mx.eval(model.parameters());base.MODEL=MODEL
 w16,high,low=prepare(model);V,D=[int(x) for x in w16.shape];baseks=base.make_kernels();uppers={32:baseks[1],64:make_upper('pb_upper_tg64',64),128:make_upper('pb_upper_tg128',128)}
 # compile
 h=one_hidden(model,tok,PROMPTS[0],D)
 for TG,k in uppers.items():U=k(inputs=[high,h],grid=(V*TG,1,1),threadgroup=(TG,1,1),output_shapes=[(V,)],output_dtypes=[mx.float32],init_value=0.0)[0];mx.eval(U)
 stage={str(TG):stage_time(k,TG,high,h,V) for TG,k in uppers.items()}
 modes=['dense','pb32','pb64','pb128','native'];rows=[]
 for m in modes:run(model,tok,PROMPTS[0],m,uppers,baseks,w16,high,low,4)
 for pi,p in enumerate(PROMPTS):
  order=modes[pi%len(modes):]+modes[:pi%len(modes)];res={}
  for m in order:
   gc.collect();mx.clear_cache();res[m]=run(model,tok,p,m,uppers,baseks,w16,high,low,TOKENS)
  d=res['dense'];n=res['native'];r={'prompt_index':pi,'order':order,'dense_ms':d['median_ms'],'native_ms':n['median_ms']}
  for TG in [32,64,128]:
   q=res[f'pb{TG}'];r[f'pb{TG}_exact']=q['tokens']==d['tokens'];r[f'pb{TG}_ms']=q['median_ms'];r[f'pb32_over_pb{TG}']=res['pb32']['median_ms']/q['median_ms'];r[f'native_over_pb{TG}']=n['median_ms']/q['median_ms']
  rows.append(r)
 out={'kind':'proofbits_multisimd_upper','model':MODEL,'stage_upper_ms_first_prompt':stage,'rows':rows}
 for TG in [32,64,128]:
  out[f'pb{TG}_all_exact']=all(r[f'pb{TG}_exact'] for r in rows);out[f'pb{TG}_median_native_over']=med([r[f'native_over_pb{TG}'] for r in rows]);out[f'pb{TG}_min_native_over']=min(r[f'native_over_pb{TG}'] for r in rows)
 if True:
  out['pb64_median_vs_pb32']=med([r['pb32_over_pb64'] for r in rows]);out['pb128_median_vs_pb32']=med([r['pb32_over_pb128'] for r in rows])
 Path('experiments/artifacts').mkdir(parents=True,exist_ok=True);Path('experiments/artifacts/proofbits_mlx_multisimd_upper.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
if __name__=='__main__':main()
