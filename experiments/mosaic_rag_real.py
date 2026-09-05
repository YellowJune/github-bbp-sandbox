#!/usr/bin/env python3
"""MOSAIC-RAG real-model pilot on HotpotQA-derived RAG contexts.

This script is deliberately evidence-bounded. It evaluates three separable claims:
1) independently cached document KV states are not generally equivalent to full-context KV;
2) full-context cross-document attention mass predicts the resulting KV error;
3) selectively recomputing a small subset of context tokens can repair a warm document cache,
   and the resulting quality/latency trade-off can be measured on a real pretrained LM.

The attention-mass selector is an oracle diagnostic because exact full-context attention is used
only to score the selector. A learned proxy is trained on disjoint examples from features available
at independent document-cache compilation time; conformal calibration is then evaluated on a
held-out split. The token-repair path itself is real execution: selected tokens are re-run through
all model layers against the current mixed KV prefix and replace their cached K/V states.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import itertools
import json
import math
import os
import random
import re
import statistics
import string
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from transformers import AutoModelForCausalLM, AutoTokenizer
import transformers


Tensor = torch.Tensor
LegacyCache = Tuple[Tuple[Tensor, Tensor], ...]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--train-examples", type=int, default=4)
    p.add_argument("--cal-examples", type=int, default=4)
    p.add_argument("--test-examples", type=int, default=4)
    p.add_argument("--trace-examples", type=int, default=500)
    p.add_argument("--docs", type=int, default=4)
    p.add_argument("--max-chunk-tokens", type=int, default=40)
    p.add_argument("--max-answer-tokens", type=int, default=10)
    p.add_argument("--generation-tokens", type=int, default=8)
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--output", default="experiments/artifacts/mosaic_rag_real.json")
    return p.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def timed_call(fn, warmup: int = 0, reps: int = 1):
    for _ in range(warmup):
        _ = fn()
    sync()
    vals = []
    out = None
    for _ in range(reps):
        sync()
        t0 = time.perf_counter()
        out = fn()
        sync()
        vals.append((time.perf_counter() - t0) * 1000.0)
    return out, float(statistics.median(vals)), vals


def cache_to_legacy(cache: Any) -> LegacyCache:
    if isinstance(cache, (tuple, list)):
        return tuple((x[0], x[1]) for x in cache)
    if hasattr(cache, "to_legacy_cache"):
        return tuple((x[0], x[1]) for x in cache.to_legacy_cache())
    if hasattr(cache, "layers"):
        pairs = []
        for layer in cache.layers:
            k = getattr(layer, "keys", None)
            v = getattr(layer, "values", None)
            if k is None or v is None:
                raise RuntimeError("DynamicCache layer does not expose keys/values")
            pairs.append((k, v))
        return tuple(pairs)
    if hasattr(cache, "key_cache") and hasattr(cache, "value_cache"):
        return tuple(zip(cache.key_cache, cache.value_cache))
    raise TypeError(f"Unsupported cache type: {type(cache)}")


def legacy_to_cache(legacy: LegacyCache, config: Any) -> Any:
    """Best-effort conversion for current Transformers cache API.

    Passing a legacy tuple directly still works in several releases, but DynamicCache is preferred.
    """
    try:
        from transformers.cache_utils import DynamicCache
        if hasattr(DynamicCache, "from_legacy_cache"):
            return DynamicCache.from_legacy_cache(legacy)
        try:
            c = DynamicCache(config=config)
        except TypeError:
            c = DynamicCache()
        for i, (k, v) in enumerate(legacy):
            c.update(k, v, i)
        return c
    except Exception:
        return legacy


def clone_legacy(cache: LegacyCache) -> LegacyCache:
    return tuple((k.clone(), v.clone()) for k, v in cache)


def concat_legacy(caches: Sequence[LegacyCache]) -> LegacyCache:
    if not caches:
        raise ValueError("No caches to concatenate")
    n_layers = len(caches[0])
    out = []
    for l in range(n_layers):
        out.append((torch.cat([c[l][0] for c in caches], dim=-2),
                    torch.cat([c[l][1] for c in caches], dim=-2)))
    return tuple(out)


def cache_seq_len(cache: LegacyCache) -> int:
    return int(cache[0][0].shape[-2])


def model_forward_with_past(model, input_ids: Tensor, past: LegacyCache, total_past: int,
                            use_cache: bool, output_attentions: bool = False,
                            output_hidden_states: bool = False):
    device = input_ids.device
    qlen = input_ids.shape[1]
    pos = torch.arange(total_past, total_past + qlen, device=device, dtype=torch.long).unsqueeze(0)
    attn = torch.ones((1, total_past + qlen), device=device, dtype=torch.long)
    kwargs = dict(
        input_ids=input_ids,
        past_key_values=legacy_to_cache(past, model.config),
        attention_mask=attn,
        position_ids=pos,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=True,
    )
    # Newer decoder models accept cache_position; older ones may not.
    try:
        kwargs["cache_position"] = torch.arange(total_past, total_past + qlen, device=device, dtype=torch.long)
        return model(**kwargs)
    except TypeError as e:
        if "cache_position" not in str(e):
            raise
        kwargs.pop("cache_position", None)
        return model(**kwargs)


def model_forward_fresh(model, input_ids: Tensor, position_start: int = 0,
                        use_cache: bool = True, output_attentions: bool = False,
                        output_hidden_states: bool = False):
    device = input_ids.device
    qlen = input_ids.shape[1]
    pos = torch.arange(position_start, position_start + qlen, device=device, dtype=torch.long).unsqueeze(0)
    return model(
        input_ids=input_ids,
        position_ids=pos,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=True,
    )


def normalize_answer(s: str) -> str:
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)
    def white_space_fix(text):
        return " ".join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)
    return white_space_fix(remove_articles(remove_punc(s.lower())))


def answer_f1(pred: str, gold: str) -> float:
    p = normalize_answer(pred).split()
    g = normalize_answer(gold).split()
    if not p and not g:
        return 1.0
    if not p or not g:
        return 0.0
    common = Counter(p) & Counter(g)
    n = sum(common.values())
    if n == 0:
        return 0.0
    prec = n / len(p)
    rec = n / len(g)
    return 2 * prec * rec / (prec + rec)


def answer_em(pred: str, gold: str) -> float:
    return float(normalize_answer(pred) == normalize_answer(gold))


def select_rag_chunks(ex: Dict[str, Any], n_docs: int) -> List[Tuple[str, List[str], bool]]:
    titles = list(ex["context"]["title"])
    sents = list(ex["context"]["sentences"])
    sf_titles = list(ex["supporting_facts"]["title"])
    sf_ids = list(ex["supporting_facts"]["sent_id"])
    support_map: Dict[str, List[int]] = {}
    for t, i in zip(sf_titles, sf_ids):
        support_map.setdefault(t, []).append(int(i))

    support_indices = [i for i, t in enumerate(titles) if t in support_map]
    distractor_indices = [i for i, t in enumerate(titles) if t not in support_map]
    chosen = support_indices[:]
    for i in distractor_indices:
        if len(chosen) >= n_docs:
            break
        chosen.append(i)
    chosen = sorted(chosen[:n_docs])

    result = []
    for i in chosen:
        title = titles[i]
        para = list(sents[i])
        if title in support_map:
            # Put the gold supporting sentence(s) first, then one neighbor. This keeps the
            # RAG slice compact while preserving the evidence needed by the question.
            ids = []
            for sid in support_map[title]:
                if 0 <= sid < len(para):
                    ids.append(sid)
                    if sid + 1 < len(para):
                        ids.append(sid + 1)
            ids = list(dict.fromkeys(ids))
            chosen_sents = [para[j] for j in ids] or para[:2]
            is_support = True
        else:
            chosen_sents = para[:2]
            is_support = False
        result.append((title, chosen_sents, is_support))
    return result


def build_tokenized_context(ex: Dict[str, Any], tokenizer, n_docs: int, max_chunk_tokens: int) -> Dict[str, Any]:
    chunks = select_rag_chunks(ex, n_docs)
    chunk_ids: List[List[int]] = []
    chunk_texts: List[str] = []
    doc_ids: List[int] = []
    starts: List[int] = []
    support_flags: List[bool] = []
    cursor = 0
    for i, (title, sents, is_support) in enumerate(chunks):
        text = f"\n\n[Document {i+1}] {title}\n" + " ".join(sents)
        ids = tokenizer.encode(text, add_special_tokens=False)
        if len(ids) > max_chunk_tokens:
            ids = ids[:max_chunk_tokens]
        if not ids:
            continue
        starts.append(cursor)
        chunk_ids.append(ids)
        chunk_texts.append(text)
        support_flags.append(bool(is_support))
        doc_ids.extend([len(chunk_ids) - 1] * len(ids))
        cursor += len(ids)

    ctx_ids = list(itertools.chain.from_iterable(chunk_ids))
    query_text = f"\n\nUse the documents above to answer with a short phrase.\nQuestion: {ex['question']}\nAnswer:"
    q_ids = tokenizer.encode(query_text, add_special_tokens=False)
    ans_ids = tokenizer.encode(" " + str(ex["answer"]), add_special_tokens=False)
    return {
        "id": ex["id"],
        "question": ex["question"],
        "answer": str(ex["answer"]),
        "chunk_ids": chunk_ids,
        "chunk_texts": chunk_texts,
        "ctx_ids": ctx_ids,
        "doc_ids": doc_ids,
        "starts": starts,
        "support_flags": support_flags,
        "q_ids": q_ids,
        "ans_ids": ans_ids,
    }


def independent_cache(model, tokenizer, data: Dict[str, Any], device: torch.device,
                      collect_features: bool = True) -> Tuple[LegacyCache, np.ndarray, float]:
    caches = []
    features_parts = []
    cursor = 0
    emb_layer = model.get_input_embeddings()

    sync()
    t0 = time.perf_counter()
    with torch.inference_mode():
        for cidx, ids in enumerate(data["chunk_ids"]):
            x = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
            out = model_forward_fresh(
                model, x, position_start=cursor, use_cache=True,
                output_attentions=False, output_hidden_states=collect_features,
            )
            caches.append(cache_to_legacy(out.past_key_values))
            if collect_features:
                with torch.no_grad():
                    emb = emb_layer(x).float().norm(dim=-1)[0].detach().cpu().numpy()
                hid = out.hidden_states[-1].float().norm(dim=-1)[0].detach().cpu().numpy()
                clen = len(ids)
                for j, tok in enumerate(ids):
                    abs_pos = (cursor + j) / max(1, len(data["ctx_ids"]) - 1)
                    rel_pos = j / max(1, clen - 1)
                    feat = [
                        cidx / max(1, len(data["chunk_ids"]) - 1),
                        rel_pos,
                        abs_pos,
                        clen / max(1, len(data["ctx_ids"])),
                        cursor / max(1, len(data["ctx_ids"])),
                        min(j, 16) / 16.0,
                        float(j < 4),
                        float(j < 8),
                        float(data["support_flags"][cidx]),
                        float(emb[j]),
                        float(hid[j]),
                        (int(tok) % 997) / 996.0,
                    ]
                    features_parts.append(feat)
            cursor += len(ids)
    sync()
    compile_ms = (time.perf_counter() - t0) * 1000.0
    feats = np.asarray(features_parts, dtype=np.float32) if collect_features else np.zeros((len(data["ctx_ids"]), 12), dtype=np.float32)
    return concat_legacy(caches), feats, compile_ms


def full_cache_and_attention(model, data: Dict[str, Any], device: torch.device) -> Tuple[LegacyCache, np.ndarray]:
    x = torch.tensor(data["ctx_ids"], dtype=torch.long, device=device).unsqueeze(0)
    with torch.inference_mode():
        out = model_forward_fresh(model, x, 0, use_cache=True, output_attentions=True, output_hidden_states=False)
    legacy = cache_to_legacy(out.past_key_values)
    L = x.shape[1]
    doc = torch.tensor(data["doc_ids"], device=device)
    diff_doc = doc[:, None] != doc[None, :]
    idx = torch.arange(L, device=device)
    causal = idx[None, :] <= idx[:, None]
    cross_mask = (diff_doc & causal).to(dtype=torch.float32)
    beta_layers = []
    for att in out.attentions:
        # [1, H, L, L] -> head-mean [L, L]
        a = att[0].float().mean(dim=0)
        beta_layers.append((a * cross_mask).sum(dim=-1).detach().cpu().numpy())
    beta = np.stack(beta_layers, axis=0).mean(axis=0).astype(np.float32)
    del out
    return legacy, beta


def kv_error_per_token(exact: LegacyCache, approx: LegacyCache) -> np.ndarray:
    rows = []
    for (ke, ve), (ka, va) in zip(exact, approx):
        ke = ke.float(); ve = ve.float(); ka = ka.float(); va = va.float()
        dk = ((ke - ka) ** 2).mean(dim=(0, 1, 3))
        dv = ((ve - va) ** 2).mean(dim=(0, 1, 3))
        bk = (ke ** 2).mean(dim=(0, 1, 3))
        bv = (ve ** 2).mean(dim=(0, 1, 3))
        rel = torch.sqrt((dk + dv) / (bk + bv + 1e-8))
        rows.append(rel.detach().cpu().numpy())
    return np.stack(rows, axis=0).mean(axis=0).astype(np.float32)


def candidate_positions(data: Dict[str, Any]) -> np.ndarray:
    d = np.asarray(data["doc_ids"], dtype=np.int64)
    return np.flatnonzero(d > 0)


def boundary_scores(data: Dict[str, Any]) -> np.ndarray:
    L = len(data["ctx_ids"])
    score = np.zeros(L, dtype=np.float32)
    starts = data["starts"]
    doc_ids = data["doc_ids"]
    for t in range(L):
        c = doc_ids[t]
        if c <= 0:
            score[t] = -1e9
        else:
            dist = t - starts[c]
            score[t] = 1.0 / (1.0 + dist)
    return score


def select_top(score: np.ndarray, data: Dict[str, Any], frac: float, seed: int | None = None,
               random_mode: bool = False) -> List[int]:
    cand = candidate_positions(data)
    k = min(len(cand), max(1, int(round(frac * len(data["ctx_ids"]))))) if frac > 0 else 0
    if k <= 0:
        return []
    if random_mode:
        rng = np.random.default_rng(seed)
        return sorted(rng.choice(cand, size=k, replace=False).tolist())
    order = cand[np.argsort(score[cand])[::-1]]
    return sorted(order[:k].tolist())


def select_risk_budget(upper: np.ndarray, data: Dict[str, Any], remaining_fraction: float = 0.25) -> List[int]:
    cand = candidate_positions(data)
    vals = np.clip(upper[cand], 0.0, None)
    total = float(vals.sum())
    if total <= 1e-12:
        return []
    order_local = np.argsort(vals)[::-1]
    remaining = total
    selected = []
    target = remaining_fraction * total
    for oi in order_local:
        if remaining <= target:
            break
        selected.append(int(cand[oi]))
        remaining -= float(vals[oi])
    return sorted(selected)


def recompute_selected_tokens(model, data: Dict[str, Any], approx: LegacyCache,
                              selected: Sequence[int], device: torch.device) -> Tuple[LegacyCache, float]:
    current = [[k.clone(), v.clone()] for k, v in approx]
    ctx = torch.tensor(data["ctx_ids"], dtype=torch.long, device=device).unsqueeze(0)
    sync(); t0 = time.perf_counter()
    with torch.inference_mode():
        for t in sorted(selected):
            token = ctx[:, t:t+1]
            if t == 0:
                out = model_forward_fresh(model, token, 0, use_cache=True)
                new = cache_to_legacy(out.past_key_values)
            else:
                prefix: LegacyCache = tuple((kv[0][:, :, :t, :], kv[1][:, :, :t, :]) for kv in current)
                out = model_forward_with_past(model, token, prefix, t, use_cache=True)
                new = cache_to_legacy(out.past_key_values)
            for l in range(len(current)):
                current[l][0][:, :, t:t+1, :] = new[l][0][:, :, -1:, :]
                current[l][1][:, :, t:t+1, :] = new[l][1][:, :, -1:, :]
            del out, new
    sync(); ms = (time.perf_counter() - t0) * 1000.0
    return tuple((x[0], x[1]) for x in current), ms


def eval_tail(model, tokenizer, data: Dict[str, Any], cache: LegacyCache, device: torch.device,
              exact_first_logits: Tensor | None = None) -> Dict[str, float]:
    q = list(data["q_ids"])
    ans = list(data["ans_ids"])[:10]
    tail_ids = q + ans
    x = torch.tensor(tail_ids, dtype=torch.long, device=device).unsqueeze(0)
    L = len(data["ctx_ids"])
    with torch.inference_mode():
        out = model_forward_with_past(model, x, cache, L, use_cache=False)
    logits = out.logits[0].float()
    qlast = max(0, len(q) - 1)
    first = logits[qlast].detach()
    if ans:
        logp = F.log_softmax(logits, dim=-1)
        nlls = []
        for j, tok in enumerate(ans):
            idx = len(q) - 1 + j
            if 0 <= idx < logp.shape[0]:
                nlls.append(float(-logp[idx, tok].item()))
        gold_nll = float(np.mean(nlls)) if nlls else float("nan")
    else:
        gold_nll = float("nan")
    result = {"gold_nll": gold_nll}
    if exact_first_logits is not None:
        p = F.log_softmax(exact_first_logits.float(), dim=-1)
        qlog = F.log_softmax(first.float(), dim=-1)
        prob = p.exp()
        kl = float((prob * (p - qlog)).sum().item())
        cos = float(F.cosine_similarity(exact_first_logits.float().unsqueeze(0), first.float().unsqueeze(0)).item())
        top = float(int(exact_first_logits.argmax().item() == first.argmax().item()))
        result.update({"kl": kl, "cosine": cos, "top1_agree": top})
    result["_first_logits"] = first
    return result


def query_ttft_ms(model, data: Dict[str, Any], cache: LegacyCache, device: torch.device, reps: int = 3) -> float:
    q = torch.tensor(data["q_ids"], dtype=torch.long, device=device).unsqueeze(0)
    L = len(data["ctx_ids"])
    def fn():
        with torch.inference_mode():
            return model_forward_with_past(model, q, cache, L, use_cache=False)
    _, ms, _ = timed_call(fn, warmup=1, reps=reps)
    return ms


def full_ttft_ms(model, data: Dict[str, Any], device: torch.device, reps: int = 3) -> float:
    ids = data["ctx_ids"] + data["q_ids"]
    x = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
    def fn():
        with torch.inference_mode():
            return model_forward_fresh(model, x, 0, use_cache=False, output_attentions=False)
    _, ms, _ = timed_call(fn, warmup=1, reps=reps)
    return ms


def greedy_generate(model, tokenizer, data: Dict[str, Any], cache: LegacyCache, device: torch.device,
                    max_new_tokens: int) -> str:
    q = torch.tensor(data["q_ids"], dtype=torch.long, device=device).unsqueeze(0)
    L = len(data["ctx_ids"])
    with torch.inference_mode():
        out = model_forward_with_past(model, q, cache, L, use_cache=True)
    cur_cache = cache_to_legacy(out.past_key_values)
    next_id = int(out.logits[0, -1].argmax().item())
    generated = [next_id]
    total = L + q.shape[1]
    for _ in range(max_new_tokens - 1):
        x = torch.tensor([[generated[-1]]], dtype=torch.long, device=device)
        with torch.inference_mode():
            out = model_forward_with_past(model, x, cur_cache, total, use_cache=True)
        cur_cache = cache_to_legacy(out.past_key_values)
        next_id = int(out.logits[0, -1].argmax().item())
        generated.append(next_id)
        total += 1
        if tokenizer.eos_token_id is not None and next_id == tokenizer.eos_token_id:
            break
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def trace_reuse_stats(examples: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    title_counter = Counter()
    text_counter = Counter()
    total = 0
    consecutive_overlaps = []
    prev_titles = None
    for ex in examples:
        titles = list(ex["context"]["title"])
        sents = list(ex["context"]["sentences"])
        cur_set = set(titles)
        if prev_titles is not None:
            inter = len(cur_set & prev_titles)
            union = len(cur_set | prev_titles)
            consecutive_overlaps.append(inter / union if union else 0.0)
        prev_titles = cur_set
        for t, ss in zip(titles, sents):
            total += 1
            title_counter[t] += 1
            txt = t + "\n" + " ".join(ss)
            text_counter[hashlib.sha256(txt.encode("utf-8")).hexdigest()] += 1
    unique_titles = len(title_counter)
    unique_text = len(text_counter)
    repeated_title_occ = sum(c - 1 for c in title_counter.values() if c > 1)
    repeated_text_occ = sum(c - 1 for c in text_counter.values() if c > 1)
    return {
        "queries": len(examples),
        "document_occurrences": total,
        "unique_titles": unique_titles,
        "unique_exact_documents": unique_text,
        "repeated_title_occurrence_fraction": repeated_title_occ / max(1, total),
        "repeated_exact_document_occurrence_fraction": repeated_text_occ / max(1, total),
        "mean_consecutive_title_jaccard": float(np.mean(consecutive_overlaps)) if consecutive_overlaps else 0.0,
        "top_repeated_titles": title_counter.most_common(10),
    }


def safe_spearman(a: Sequence[float], b: Sequence[float]) -> float:
    r = spearmanr(np.asarray(a), np.asarray(b)).statistic
    return float(r) if np.isfinite(r) else 0.0


def aggregate_rows(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    by_method: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        by_method.setdefault(r["method"], []).append(r)
    out = {}
    metrics = ["kl", "cosine", "top1_agree", "gold_nll_delta", "recompute_fraction", "repair_ms", "query_ms", "warm_ttft_ms", "speedup", "amortized_speedup_16"]
    for m, rr in by_method.items():
        d = {"n": len(rr)}
        for key in metrics:
            vals = [float(x[key]) for x in rr if key in x and np.isfinite(float(x[key]))]
            if vals:
                d[key + "_mean"] = float(np.mean(vals))
                d[key + "_median"] = float(np.median(vals))
                d[key + "_std"] = float(np.std(vals))
        out[m] = d
    return out


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    if not torch.cuda.is_available():
        raise RuntimeError("This pilot is intended for the CUDA self-hosted runner")
    device = torch.device("cuda")
    gpu_name = torch.cuda.get_device_name(0)
    print(f"GPU={gpu_name}")
    print(f"torch={torch.__version__} transformers={transformers.__version__}")

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    dtype = torch.float16
    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=dtype,
            attn_implementation="eager",
            low_cpu_mem_usage=True,
        ).to(device)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        ).to(device)
        if hasattr(model, "set_attn_implementation"):
            model.set_attn_implementation("eager")
    model.eval()

    n_total = args.train_examples + args.cal_examples + args.test_examples
    print("Loading HotpotQA streaming validation trace...")
    stream = load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation", streaming=True)
    trace_raw = list(itertools.islice(stream, max(args.trace_examples, n_total)))
    trace = trace_reuse_stats(trace_raw[:args.trace_examples])
    exp_raw = trace_raw[:n_total]
    prepared = [build_tokenized_context(ex, tokenizer, args.docs, args.max_chunk_tokens) for ex in exp_raw]
    prepared = [x for x in prepared if len(x["ctx_ids"]) > 16 and len(x["q_ids"]) > 1]
    if len(prepared) < n_total:
        raise RuntimeError(f"Only {len(prepared)} usable examples; need {n_total}")
    prepared = prepared[:n_total]

    # Phase A: oracle diagnostics for predictor training/calibration/test splits.
    diagnostics = []
    print(f"Phase A diagnostics: {n_total} examples")
    for i, data in enumerate(prepared):
        print(f"  diag {i+1}/{n_total} id={data['id']} ctx={len(data['ctx_ids'])}")
        approx, feats, compile_ms = independent_cache(model, tokenizer, data, device, collect_features=True)
        exact, beta = full_cache_and_attention(model, data, device)
        err = kv_error_per_token(exact, approx)
        cand = candidate_positions(data)
        diagnostics.append({
            "features": feats,
            "beta": beta,
            "error": err,
            "compile_ms": compile_ms,
            "beta_error_spearman": safe_spearman(beta[cand], err[cand]) if len(cand) else 0.0,
        })
        del approx, exact
        gc.collect(); torch.cuda.empty_cache()

    tr0 = 0
    tr1 = args.train_examples
    ca1 = tr1 + args.cal_examples
    te1 = ca1 + args.test_examples
    def stack_part(start: int, end: int, key: str):
        arr = []
        for j in range(start, end):
            cand = candidate_positions(prepared[j])
            arr.append(diagnostics[j][key][cand])
        return np.concatenate(arr, axis=0)
    X_train = np.concatenate([diagnostics[j]["features"][candidate_positions(prepared[j])] for j in range(tr0, tr1)], axis=0)
    y_train = stack_part(tr0, tr1, "beta")
    X_cal = np.concatenate([diagnostics[j]["features"][candidate_positions(prepared[j])] for j in range(tr1, ca1)], axis=0)
    y_cal = stack_part(tr1, ca1, "beta")
    X_test = np.concatenate([diagnostics[j]["features"][candidate_positions(prepared[j])] for j in range(ca1, te1)], axis=0)
    y_test = stack_part(ca1, te1, "beta")
    e_test = stack_part(ca1, te1, "error")

    predictor = HistGradientBoostingRegressor(
        max_iter=160, learning_rate=0.06, max_depth=4, l2_regularization=1e-3,
        random_state=args.seed,
    )
    predictor.fit(X_train, y_train)
    cal_pred = np.clip(predictor.predict(X_cal), 0.0, 1.0)
    residual = y_cal - cal_pred
    try:
        q95 = float(np.quantile(residual, 0.95, method="higher"))
    except TypeError:
        q95 = float(np.quantile(residual, 0.95, interpolation="higher"))
    q95 = max(0.0, q95)
    test_pred = np.clip(predictor.predict(X_test), 0.0, 1.0)
    test_upper = np.clip(test_pred + q95, 0.0, 1.0)
    coverage = float(np.mean(y_test <= test_upper + 1e-12))

    predictor_summary = {
        "train_tokens": int(len(y_train)),
        "cal_tokens": int(len(y_cal)),
        "test_tokens": int(len(y_test)),
        "test_pred_beta_spearman": safe_spearman(test_pred, y_test),
        "test_pred_kv_error_spearman": safe_spearman(test_pred, e_test),
        "test_oracle_beta_kv_error_spearman": safe_spearman(y_test, e_test),
        "one_sided_q95": q95,
        "empirical_upper_coverage": coverage,
        "mean_per_example_beta_error_spearman": float(np.mean([diagnostics[j]["beta_error_spearman"] for j in range(ca1, te1)])),
    }
    print("Predictor summary:", json.dumps(predictor_summary, sort_keys=True))

    # Attach predictions per example.
    for j in range(ca1, te1):
        cand = candidate_positions(prepared[j])
        pred = np.zeros(len(prepared[j]["ctx_ids"]), dtype=np.float32)
        upper = np.zeros_like(pred)
        pp = np.clip(predictor.predict(diagnostics[j]["features"][cand]), 0.0, 1.0)
        pred[cand] = pp
        upper[cand] = np.clip(pp + q95, 0.0, 1.0)
        diagnostics[j]["pred"] = pred
        diagnostics[j]["upper"] = upper

    # Phase B: real warm-cache repair + real HotpotQA answer-tail evaluation.
    rows = []
    generation_rows = []
    print(f"Phase B real execution: {args.test_examples} held-out examples")
    for local_i, j in enumerate(range(ca1, te1)):
        data = prepared[j]
        diag = diagnostics[j]
        print(f"  test {local_i+1}/{args.test_examples} id={data['id']} ctx={len(data['ctx_ids'])}")
        approx, _, compile_ms = independent_cache(model, tokenizer, data, device, collect_features=False)
        # Full cache without attentions is enough in Phase B.
        ctx_t = torch.tensor(data["ctx_ids"], dtype=torch.long, device=device).unsqueeze(0)
        with torch.inference_mode():
            exact_out = model_forward_fresh(model, ctx_t, 0, use_cache=True, output_attentions=False)
        exact = cache_to_legacy(exact_out.past_key_values)
        del exact_out

        exact_eval = eval_tail(model, tokenizer, data, exact, device, exact_first_logits=None)
        exact_first = exact_eval.pop("_first_logits")
        exact_nll = exact_eval["gold_nll"]
        full_ms = full_ttft_ms(model, data, device, reps=3)

        # Warm no-repair baseline.
        reuse_eval = eval_tail(model, tokenizer, data, approx, device, exact_first_logits=exact_first)
        reuse_eval.pop("_first_logits", None)
        reuse_q_ms = query_ttft_ms(model, data, approx, device, reps=3)
        rows.append({
            "id": data["id"], "method": "reuse_0pct", "selected_tokens": 0,
            "recompute_fraction": 0.0, "repair_ms": 0.0, "query_ms": reuse_q_ms,
            "warm_ttft_ms": reuse_q_ms, "full_ttft_ms": full_ms,
            "speedup": full_ms / max(1e-9, reuse_q_ms),
            "amortized_speedup_16": 16 * full_ms / max(1e-9, compile_ms + 16 * reuse_q_ms),
            "kl": reuse_eval["kl"], "cosine": reuse_eval["cosine"], "top1_agree": reuse_eval["top1_agree"],
            "gold_nll": reuse_eval["gold_nll"], "gold_nll_delta": reuse_eval["gold_nll"] - exact_nll,
            "compile_ms": compile_ms,
        })

        # Selection methods / budgets.
        method_specs = []
        for frac in (0.05, 0.10, 0.20):
            method_specs.append((f"proxy_{int(frac*100)}pct", select_top(diag["pred"], data, frac)))
            method_specs.append((f"oracle_{int(frac*100)}pct", select_top(diag["beta"], data, frac)))
        method_specs.append(("boundary_10pct", select_top(boundary_scores(data), data, 0.10)))
        method_specs.append(("random_10pct", select_top(np.zeros(len(data["ctx_ids"]), dtype=np.float32), data, 0.10, seed=args.seed + j, random_mode=True)))
        method_specs.append(("proxy_risk25", select_risk_budget(diag["upper"], data, remaining_fraction=0.25)))

        caches_for_generation: Dict[str, LegacyCache] = {}
        for name, selected in method_specs:
            repaired, repair_ms = recompute_selected_tokens(model, data, approx, selected, device)
            ev = eval_tail(model, tokenizer, data, repaired, device, exact_first_logits=exact_first)
            ev.pop("_first_logits", None)
            q_ms = query_ttft_ms(model, data, repaired, device, reps=2)
            total_ms = repair_ms + q_ms
            rows.append({
                "id": data["id"], "method": name, "selected_tokens": len(selected),
                "recompute_fraction": len(selected) / max(1, len(data["ctx_ids"])),
                "repair_ms": repair_ms, "query_ms": q_ms, "warm_ttft_ms": total_ms,
                "full_ttft_ms": full_ms, "speedup": full_ms / max(1e-9, total_ms),
                "amortized_speedup_16": 16 * full_ms / max(1e-9, compile_ms + 16 * total_ms),
                "kl": ev["kl"], "cosine": ev["cosine"], "top1_agree": ev["top1_agree"],
                "gold_nll": ev["gold_nll"], "gold_nll_delta": ev["gold_nll"] - exact_nll,
                "compile_ms": compile_ms,
            })
            if name in {"proxy_10pct", "oracle_10pct"}:
                caches_for_generation[name] = repaired
            else:
                del repaired

        # End-task greedy generation on a small, fixed set of caches.
        gen_methods = {"exact_full": exact, "reuse_0pct": approx}
        gen_methods.update(caches_for_generation)
        exact_text = None
        for name, cache in gen_methods.items():
            text = greedy_generate(model, tokenizer, data, cache, device, args.generation_tokens)
            if name == "exact_full":
                exact_text = text
            generation_rows.append({
                "id": data["id"], "method": name, "prediction": text, "gold": data["answer"],
                "gold_em": answer_em(text, data["answer"]), "gold_f1": answer_f1(text, data["answer"]),
                "exact_output_match": None,
            })
        if exact_text is not None:
            for r in generation_rows[-len(gen_methods):]:
                r["exact_output_match"] = answer_em(r["prediction"], exact_text)
                r["exact_output_f1"] = answer_f1(r["prediction"], exact_text)

        del approx, exact, caches_for_generation, ctx_t, exact_first
        gc.collect(); torch.cuda.empty_cache()

    generation_agg = {}
    for m in sorted({r["method"] for r in generation_rows}):
        rr = [r for r in generation_rows if r["method"] == m]
        generation_agg[m] = {
            "n": len(rr),
            "gold_em": float(np.mean([r["gold_em"] for r in rr])),
            "gold_f1": float(np.mean([r["gold_f1"] for r in rr])),
            "exact_output_match": float(np.mean([r["exact_output_match"] for r in rr])),
            "exact_output_f1": float(np.mean([r["exact_output_f1"] for r in rr])),
        }

    result = {
        "metadata": {
            "model": args.model,
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "gpu": gpu_name,
            "dataset": "hotpotqa/hotpot_qa:distractor:validation",
            "context_construction": "2 gold-support documents when available + distractors; compact support sentences; no claim of official HotpotQA EM/F1 protocol",
            "train_examples": args.train_examples,
            "cal_examples": args.cal_examples,
            "test_examples": args.test_examples,
            "docs_per_context": args.docs,
            "max_chunk_tokens": args.max_chunk_tokens,
            "seed": args.seed,
            "selector_boundary": "oracle beta uses exact full-context attention only as an upper-bound diagnostic; proxy selector is trained on disjoint examples from independent-cache features",
            "timing_boundary": "warm document-cache TTFT; repair path includes real token recomputation and Python orchestration; isolated document compilation reported separately",
        },
        "trace": trace,
        "predictor": predictor_summary,
        "aggregate": aggregate_rows(rows),
        "generation": generation_agg,
        "rows": rows,
        "generation_rows": generation_rows,
        "example_ids": [prepared[j]["id"] for j in range(ca1, te1)],
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    summary = {
        "metadata": result["metadata"],
        "trace": trace,
        "predictor": predictor_summary,
        "aggregate": result["aggregate"],
        "generation": generation_agg,
    }
    print("MOSAIC_RAG_RESULT_JSON=" + json.dumps(summary, ensure_ascii=False, separators=(",", ":")))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
