import gc
import json
import statistics
import sys
from pathlib import Path

import mlx.core as mx
from mlx_lm import load

sys.path.insert(0, str(Path(__file__).resolve().parent))
import proofbits_mlx_integrated_decode as bench

MODEL='mlx-community/gemma-3-270m-bf16'
TOKENS=32
PROMPTS=[
    'Explain why logarithms appear naturally in entropy and information theory.',
    'Solve carefully: if x + 1/x = 3, derive x^5 + 1/x^5.',
    'Write a short Python function that returns the longest increasing contiguous run in a list and explain its complexity.',
    'Summarize the key tradeoff between memory bandwidth and arithmetic intensity in autoregressive inference.',
    'Write a compact science-fiction scene about a computer that can prove which memory bytes it never needs to read.',
]


def med(x): return float(statistics.median(x))


def main():
    bench.MODEL=MODEL
    model,tok=load(MODEL)
    # Actual runtime conversion: body and output head are both FP16 before any
    # warmup/timing. The source repository is BF16; this experiment tests a
    # fully FP16 deployment state rather than the earlier BF16-body/FP16-head mix.
    model.set_dtype(mx.float16)
    mx.eval(model.parameters())
    head=model.lm_head.weight if hasattr(model,'lm_head') else model.model.embed_tokens.weight
    head_dtype=str(head.dtype)

    kernels=bench.make_kernels()
    w16,high,low=bench.prepare_weights(model)

    old_prompt=bench.PROMPT
    bench.PROMPT=PROMPTS[0]
    for mode in ['dense_custom_fp16','proofbits_fp16','native_mlx_fp16']:
        bench.decode_once(model,tok,mode,kernels,w16,high,low,6,False)
    mx.synchronize()

    matched=[]
    native=[]
    body_dtype=None
    for i,prompt in enumerate(PROMPTS):
        bench.PROMPT=prompt
        # Matched exact-reference AB/BA.
        order=['dense_custom_fp16','proofbits_fp16'] if i%2==0 else ['proofbits_fp16','dense_custom_fp16']
        res={}
        for mode in order:
            gc.collect();mx.clear_cache()
            res[mode]=bench.decode_once(model,tok,mode,kernels,w16,high,low,TOKENS,False)
        d,p=res['dense_custom_fp16'],res['proofbits_fp16']
        matched.append({
            'prompt_index':i,'order':order,'sequence_exact':d['tokens']==p['tokens'],
            'dense_ms':d['median_ms_per_token'],'proofbits_ms':p['median_ms_per_token'],
            'speedup':d['median_ms_per_token']/p['median_ms_per_token']})

        # Optimized MLX FP16 reference vs ProofBits, counterbalanced independently.
        order2=['native_mlx_fp16','proofbits_fp16'] if i%2==0 else ['proofbits_fp16','native_mlx_fp16']
        res2={}
        for mode in order2:
            gc.collect();mx.clear_cache()
            res2[mode]=bench.decode_once(model,tok,mode,kernels,w16,high,low,TOKENS,False)
        n,p2=res2['native_mlx_fp16'],res2['proofbits_fp16']
        native.append({
            'prompt_index':i,'order':order2,'sequence_equal':n['tokens']==p2['tokens'],
            'native_ms':n['median_ms_per_token'],'proofbits_ms':p2['median_ms_per_token'],
            'native_over_pb_speedup':n['median_ms_per_token']/p2['median_ms_per_token']})

    # Probe body output dtype explicitly after conversion.
    bench.PROMPT=PROMPTS[0]
    ids=bench.tokenize(tok)
    b=model.model(ids[None])
    mx.eval(b);body_dtype=str(b.dtype)
    bench.PROMPT=old_prompt

    ms=[x['speedup'] for x in matched];ns=[x['native_over_pb_speedup'] for x in native]
    out={
        'kind':'proofbits_full_model_fp16_mlx_runtime',
        'source_model':MODEL,
        'runtime_head_dtype':head_dtype,
        'runtime_body_output_dtype':body_dtype,
        'n_prompts':len(PROMPTS),'tokens_per_prompt':TOKENS,
        'matched_tokens':len(PROMPTS)*TOKENS,
        'all_matched_sequences_exact':all(x['sequence_exact'] for x in matched),
        'matched_median_speedup':med(ms),'matched_mean_speedup':float(statistics.mean(ms)),
        'matched_min_speedup':float(min(ms)),'matched_max_speedup':float(max(ms)),
        'native_all_sequences_equal':all(x['sequence_equal'] for x in native),
        'native_median_over_pb_speedup':med(ns),'native_mean_over_pb_speedup':float(statistics.mean(ns)),
        'native_min_over_pb_speedup':float(min(ns)),'native_max_over_pb_speedup':float(max(ns)),
        'matched_rows':matched,'native_rows':native,
        'note':'The downloadable source checkpoint is BF16, but model.set_dtype(float16) is applied to the full MLX model before warmup/timing. Thus both transformer body and head operate in an FP16 runtime state. ProofBits high/low planes are made from that actual FP16 head state; conversion/plane setup is outside timed generation.'
    }
    Path('experiments/artifacts').mkdir(parents=True,exist_ok=True)
    Path('experiments/artifacts/proofbits_mlx_full_fp16_runtime.json').write_text(json.dumps(out,indent=2))
    print(json.dumps(out,indent=2))

if __name__=='__main__': main()
