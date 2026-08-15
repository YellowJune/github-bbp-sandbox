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
TOKENS=48
PROMPTS=[
    'Explain the intuition behind entropy and why logarithms appear in information theory.',
    'Derive the closed form of the geometric series and state the convergence condition.',
    'Solve step by step: if x + 1/x = 3, compute x^6 + 1/x^6.',
    'Write a Python function for longest increasing contiguous subarray and discuss time and space complexity.',
    'Explain the difference between optimistic and pessimistic concurrency control in databases.',
    'Summarize why memory bandwidth can dominate autoregressive neural-network inference at batch size one.',
    'Compare TCP congestion control with application-level rate limiting in a concise technical explanation.',
    'Describe how natural selection can produce complex adaptation without foresight.',
    'Give a rigorous but intuitive explanation of why the central limit theorem is useful.',
    'Write a short science-fiction scene in which a machine proves which memory bytes need not be read.',
    'Explain the ethical tradeoffs of deploying highly capable AI systems in education without using slogans.',
    'Design a small experiment that distinguishes correlation from causation and explain possible confounders.',
]


def med(xs): return float(statistics.median(xs))


def main():
    bench.MODEL=MODEL
    model,tok=load(MODEL)
    model.set_dtype(mx.float16)
    mx.eval(model.parameters())
    head=model.lm_head.weight if hasattr(model,'lm_head') else model.model.embed_tokens.weight
    kernels=bench.make_kernels()
    w16,high,low=bench.prepare_weights(model)

    # Compile all three paths before measurements.
    old_prompt=bench.PROMPT
    bench.PROMPT=PROMPTS[0]
    for mode in ['dense_custom_fp16','proofbits_fp16','native_mlx_fp16']:
        bench.decode_once(model,tok,mode,kernels,w16,high,low,6,False)
    mx.synchronize()

    # Three-method balanced order across prompts. Each method gets each ordinal
    # position four times, limiting thermal/order bias without duplicating PB.
    orders=[
        ['dense_custom_fp16','proofbits_fp16','native_mlx_fp16'],
        ['proofbits_fp16','native_mlx_fp16','dense_custom_fp16'],
        ['native_mlx_fp16','dense_custom_fp16','proofbits_fp16'],
    ]
    rows=[]
    sum_dense=sum_pb=sum_native=0.0
    for i,prompt in enumerate(PROMPTS):
        bench.PROMPT=prompt
        order=orders[i%3]
        res={}
        for mode in order:
            gc.collect();mx.clear_cache()
            res[mode]=bench.decode_once(model,tok,mode,kernels,w16,high,low,TOKENS,False)
        d=res['dense_custom_fp16'];p=res['proofbits_fp16'];n=res['native_mlx_fp16']
        sum_dense += d['mean_ms_per_token']*TOKENS
        sum_pb += p['mean_ms_per_token']*TOKENS
        sum_native += n['mean_ms_per_token']*TOKENS
        rows.append({
            'prompt_index':i,'order':order,
            'matched_exact':d['tokens']==p['tokens'],
            'native_equal':n['tokens']==p['tokens'],
            'dense_median_ms':d['median_ms_per_token'],
            'pb_median_ms':p['median_ms_per_token'],
            'native_median_ms':n['median_ms_per_token'],
            'matched_speedup':d['median_ms_per_token']/p['median_ms_per_token'],
            'native_over_pb_speedup':n['median_ms_per_token']/p['median_ms_per_token'],
            'dense_mean_ms':d['mean_ms_per_token'],
            'pb_mean_ms':p['mean_ms_per_token'],
            'native_mean_ms':n['mean_ms_per_token'],
        })

    # Explicitly verify runtime body output dtype after conversion.
    bench.PROMPT=PROMPTS[0]
    b=model.model(bench.tokenize(tok)[None]);mx.eval(b)
    body_dtype=str(b.dtype)
    bench.PROMPT=old_prompt

    msp=[x['matched_speedup'] for x in rows]
    nsp=[x['native_over_pb_speedup'] for x in rows]
    out={
        'kind':'proofbits_full_model_fp16_replicated_stress',
        'source_model':MODEL,
        'runtime_head_dtype':str(head.dtype),
        'runtime_body_output_dtype':body_dtype,
        'n_prompts':len(PROMPTS),'tokens_per_prompt':TOKENS,
        'paired_tokens':len(PROMPTS)*TOKENS,
        'all_matched_exact':all(x['matched_exact'] for x in rows),
        'all_native_equal':all(x['native_equal'] for x in rows),
        'matched_prompt_median_speedup':med(msp),
        'matched_prompt_mean_speedup':float(statistics.mean(msp)),
        'matched_prompt_min_speedup':float(min(msp)),
        'matched_prompt_max_speedup':float(max(msp)),
        'matched_pooled_speedup':sum_dense/sum_pb,
        'native_prompt_median_over_pb':med(nsp),
        'native_prompt_mean_over_pb':float(statistics.mean(nsp)),
        'native_prompt_min_over_pb':float(min(nsp)),
        'native_prompt_max_over_pb':float(max(nsp)),
        'native_pooled_over_pb':sum_native/sum_pb,
        'rows':rows,
        'note':'The downloaded source is BF16, but the entire MLX model is converted to FP16 before warmup/timing; body output and output head are both FP16. Dense custom, ProofBits, and native MLX use the same FP16 runtime weights/body. Three-method execution order is rotated across 12 prompts so each method appears first/second/third four times. ProofBits setup is outside timing.'
    }
    Path('experiments/artifacts').mkdir(parents=True,exist_ok=True)
    Path('experiments/artifacts/proofbits_mlx_full_fp16_stress.json').write_text(json.dumps(out,indent=2))
    print(json.dumps(out,indent=2))

if __name__=='__main__': main()
