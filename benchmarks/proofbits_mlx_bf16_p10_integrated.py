import gc
import json
import statistics
import time
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx_lm import load
from mlx_lm.models import cache as cache_mod

MODEL = "mlx-community/gemma-3-270m-bf16"
PROMPT = "Explain in one paragraph why exact inference decisions can sometimes be certified from partial numerical representations."
NEW_TOKENS = 48
ROUNDS = 4
PREFIX_BITS = 10
SUFFIX_BITS = 6
PACK_CHUNK_ROWS = 1024


def med(xs):
    return float(statistics.median(xs))


def pack_groups(values, bits):
    """Pack [R,G,16] unsigned values into [R,G,16*bits/32] uint32 words."""
    R, G, K = values.shape
    assert K == 16 and (16 * bits) % 32 == 0
    W = (16 * bits) // 32
    out = np.zeros((R, G, W), dtype=np.uint32)
    vals = values.astype(np.uint32, copy=False)
    mask = (1 << bits) - 1
    for j in range(16):
        b = j * bits
        wi = b >> 5
        sh = b & 31
        v = vals[:, :, j] & mask
        out[:, :, wi] |= v << sh
        if sh + bits > 32:
            out[:, :, wi + 1] |= v >> (32 - sh)
    return out


def pack_bf16_weight(w):
    V, D = [int(x) for x in w.shape]
    assert D % 16 == 0
    pwr = D * PREFIX_BITS // 32
    swr = D * SUFFIX_BITS // 32
    prefix = np.empty((V, pwr), dtype=np.uint32)
    suffix = np.empty((V, swr), dtype=np.uint32)
    for s in range(0, V, PACK_CHUNK_ROWS):
        e = min(V, s + PACK_CHUNK_ROWS)
        # BF16 -> FP32 is exact; the original BF16 storage word is the high
        # 16 bits of the resulting IEEE-754 FP32 representation.
        vals = np.array(w[s:e].astype(mx.float32), copy=True)
        raw = (vals.view(np.uint32) >> 16).astype(np.uint16)
        groups = raw.reshape((e - s, D // 16, 16))
        pv = (groups >> SUFFIX_BITS).astype(np.uint16)
        sv = (groups & ((1 << SUFFIX_BITS) - 1)).astype(np.uint16)
        prefix[s:e] = pack_groups(pv, PREFIX_BITS).reshape((e - s, pwr))
        suffix[s:e] = pack_groups(sv, SUFFIX_BITS).reshape((e - s, swr))
    return mx.array(prefix.reshape(-1)), mx.array(suffix.reshape(-1))


def make_kernels(D):
    PWR = D * PREFIX_BITS // 32
    SWR = D * SUFFIX_BITS // 32
    dense_src = f'''
        uint row = threadgroup_position_in_grid.x;
        uint lane = thread_index_in_simdgroup;
        ulong base = (ulong)row * {D}ul;
        float acc = 0.0f;
        for (uint j = lane; j < {D}u; j += 32u) {{
            acc = fma(hidden[j], (float)weight[base + j], acc);
        }}
        float total = simd_sum(acc);
        if (lane == 0) scores[row] = total;
    '''
    upper_src = f'''
        uint row = threadgroup_position_in_grid.x;
        uint lane = thread_index_in_simdgroup;
        ulong rowBase = (ulong)row * {PWR}ul;
        float acc = 0.0f;
        for (uint j = lane; j < {D}u; j += 32u) {{
            uint bit = j * 10u;
            ulong wi = rowBase + (ulong)(bit >> 5);
            uint sh = bit & 31u;
            uint pre = prefix[wi] >> sh;
            if (sh > 22u) pre |= prefix[wi + 1] << (32u - sh);
            pre &= 0x3FFu;
            uint rawBase = pre << 6u;
            uint weightSign = rawBase & 0x8000u;
            uint hiddenSign = hidden[j] < 0.0f ? 0x8000u : 0u;
            uint raw = rawBase | ((weightSign == hiddenSign) ? 0x3Fu : 0u);
            float endpoint = as_type<float>(raw << 16u);
            acc = fma(hidden[j], endpoint, acc);
        }}
        float total = simd_sum(acc);
        if (lane == 0) upper[row] = total;
    '''
    exact_common = f'''
            uint pbit = j * 10u;
            ulong pwi = pbase + (ulong)(pbit >> 5);
            uint psh = pbit & 31u;
            uint pre = prefix[pwi] >> psh;
            if (psh > 22u) pre |= prefix[pwi + 1] << (32u - psh);
            pre &= 0x3FFu;
            uint sbit = j * 6u;
            ulong swi = sbase + (ulong)(sbit >> 5);
            uint ssh = sbit & 31u;
            uint suf = suffix[swi] >> ssh;
            if (ssh > 26u) suf |= suffix[swi + 1] << (32u - ssh);
            suf &= 0x3Fu;
            uint raw = (pre << 6u) | suf;
            float w = as_type<float>(raw << 16u);
            acc = fma(hidden[j], w, acc);
    '''
    pilot_src = f'''
        uint lane = thread_index_in_simdgroup;
        uint row = (uint)pilot[0];
        ulong pbase = (ulong)row * {PWR}ul;
        ulong sbase = (ulong)row * {SWR}ul;
        float acc = 0.0f;
        for (uint j = lane; j < {D}u; j += 32u) {{
            {exact_common}
        }}
        float total = simd_sum(acc);
        if (lane == 0) bound[0] = total;
    '''
    refine_src = f'''
        uint row = threadgroup_position_in_grid.x;
        uint lane = thread_index_in_simdgroup;
        if (upper[row] < bound[0]) {{
            if (lane == 0) exact[row] = -3.402823466e+38f;
            return;
        }}
        ulong pbase = (ulong)row * {PWR}ul;
        ulong sbase = (ulong)row * {SWR}ul;
        float acc = 0.0f;
        for (uint j = lane; j < {D}u; j += 32u) {{
            {exact_common}
        }}
        float total = simd_sum(acc);
        if (lane == 0) exact[row] = total;
    '''
    dense = mx.fast.metal_kernel(name='pb_bf16_dense', input_names=['weight','hidden'], output_names=['scores'], source=dense_src)
    upper = mx.fast.metal_kernel(name='pb_bf16_p10_upper', input_names=['prefix','hidden'], output_names=['upper'], source=upper_src)
    pilot = mx.fast.metal_kernel(name='pb_bf16_p10_pilot', input_names=['prefix','suffix','hidden','pilot'], output_names=['bound'], source=pilot_src)
    refine = mx.fast.metal_kernel(name='pb_bf16_p10_refine', input_names=['prefix','suffix','hidden','upper','bound'], output_names=['exact'], source=refine_src)
    return dense, upper, pilot, refine


def dense_decision(k, weight, h, V):
    s = k(inputs=[weight,h], grid=(V*32,1,1), threadgroup=(32,1,1), output_shapes=[(V,)], output_dtypes=[mx.float32], init_value=0.0)[0]
    return mx.argmax(s).astype(mx.uint32)


def proofbits_decision(ks, prefix, suffix, h, V, diagnostic=False):
    _, ku, kp, kr = ks
    U = ku(inputs=[prefix,h], grid=(V*32,1,1), threadgroup=(32,1,1), output_shapes=[(V,)], output_dtypes=[mx.float32], init_value=0.0)[0]
    p = mx.reshape(mx.argmax(U).astype(mx.uint32),(1,))
    B = kp(inputs=[prefix,suffix,h,p], grid=(32,1,1), threadgroup=(32,1,1), output_shapes=[(1,)], output_dtypes=[mx.float32], init_value=0.0)[0]
    E = kr(inputs=[prefix,suffix,h,U,B], grid=(V*32,1,1), threadgroup=(32,1,1), output_shapes=[(V,)], output_dtypes=[mx.float32], init_value=-3.402823466e38)[0]
    y = mx.argmax(E).astype(mx.uint32)
    if diagnostic:
        return y, mx.sum(U >= B)
    return y


def tokenize(tok):
    try: ids=tok.encode(PROMPT)
    except Exception: ids=tok(PROMPT)['input_ids']
    return mx.array(ids,dtype=mx.int32)


def decode_once(model,tok,mode,ks,weight,prefix,suffix,max_new,diagnostic=False):
    V,D=[int(x) for x in weight.shape]
    cache=cache_mod.make_prompt_cache(model)
    body=model.model(tokenize(tok)[None],cache=cache)
    h=body[:,-1,:].reshape((D,)).astype(mx.float32)
    mx.eval(h,[c.state for c in cache])
    tokens=[]; total=[]; heads=[]; bodies=[]; surv=[]
    for _ in range(max_new):
        ts=time.perf_counter(); th=time.perf_counter()
        if mode=='dense_bf16':
            y=dense_decision(ks[0],weight,h,V)
        else:
            if diagnostic: y,s=proofbits_decision(ks,prefix,suffix,h,V,True)
            else: y=proofbits_decision(ks,prefix,suffix,h,V,False)
        mx.eval(y)
        if diagnostic and mode!='dense_bf16': mx.eval(s); surv.append(int(s.item()))
        token=int(y.item()); tokens.append(token)
        heads.append((time.perf_counter()-th)*1e3)
        tb=time.perf_counter()
        body=model.model(mx.array([[token]],dtype=mx.int32),cache=cache)
        h=body[:,-1,:].reshape((D,)).astype(mx.float32)
        mx.eval(h,[c.state for c in cache])
        bodies.append((time.perf_counter()-tb)*1e3)
        total.append((time.perf_counter()-ts)*1e3)
    out={'tokens':tokens,'median_total_ms':med(total),'mean_total_ms':float(statistics.mean(total)),'median_head_ms':med(heads),'median_body_ms':med(bodies)}
    if surv:
        out['survivor_mean']=float(statistics.mean(surv));out['survivor_fraction_mean']=float(statistics.mean(surv)/V)
    return out


def main():
    model,tok=load(MODEL); mx.eval(model.parameters())
    weight=model.lm_head.weight if hasattr(model,'lm_head') else model.model.embed_tokens.weight
    assert weight.dtype==mx.bfloat16
    V,D=[int(x) for x in weight.shape]
    tpack=time.perf_counter(); prefix,suffix=pack_bf16_weight(weight); mx.eval(prefix,suffix); pack_s=time.perf_counter()-tpack
    ks=make_kernels(D)
    # compile paths
    for m in ['dense_bf16','proofbits_bf16_p10']: decode_once(model,tok,m,ks,weight,prefix,suffix,4,False)
    mx.synchronize()
    rounds=[]
    for r in range(ROUNDS):
        order=['dense_bf16','proofbits_bf16_p10'] if r%2==0 else ['proofbits_bf16_p10','dense_bf16']
        res={}
        for m in order:
            gc.collect();mx.clear_cache();res[m]=decode_once(model,tok,m,ks,weight,prefix,suffix,NEW_TOKENS,False)
        d=res['dense_bf16'];p=res['proofbits_bf16_p10']
        rounds.append({'round':r+1,'order':order,'sequence_exact':d['tokens']==p['tokens'],'dense_total_ms':d['median_total_ms'],'proofbits_total_ms':p['median_total_ms'],'total_speedup':d['median_total_ms']/p['median_total_ms'],'dense_head_ms':d['median_head_ms'],'proofbits_head_ms':p['median_head_ms'],'head_speedup':d['median_head_ms']/p['median_head_ms'],'dense_body_ms':d['median_body_ms'],'proofbits_body_ms':p['median_body_ms']})
    diag=decode_once(model,tok,'proofbits_bf16_p10',ks,weight,prefix,suffix,NEW_TOKENS,True)
    sp=[x['total_speedup'] for x in rounds]; hp=[x['head_speedup'] for x in rounds]
    out={'kind':'proofbits_native_bf16_p10_packed_integrated','model':MODEL,'V':V,'D':D,'prefix_bits':10,'suffix_bits':6,'prefix_words':int(prefix.size),'suffix_words':int(suffix.size),'prefix_bytes':int(prefix.nbytes),'suffix_bytes':int(suffix.nbytes),'original_bf16_bytes':int(weight.nbytes),'packing_seconds':pack_s,'rounds':rounds,'all_sequences_exact':all(x['sequence_exact'] for x in rounds),'median_total_speedup':med(sp),'mean_total_speedup':float(statistics.mean(sp)),'median_head_speedup':med(hp),'diagnostic_survivor_mean':diag.get('survivor_mean'),'diagnostic_survivor_fraction_mean':diag.get('survivor_fraction_mean'),'note':'Native checkpoint BF16 semantics. Prefix and suffix are densely bit-packed 10+6; setup packing is outside timed generation. Matched dense reference reads original BF16 weight storage and accumulates in FP32 with the same row/SIMD reduction structure.'}
    Path('experiments/artifacts').mkdir(parents=True,exist_ok=True);Path('experiments/artifacts/proofbits_mlx_bf16_p10_integrated.json').write_text(json.dumps(out,indent=2,default=str));print(json.dumps(out,indent=2,default=str))

if __name__=='__main__': main()
