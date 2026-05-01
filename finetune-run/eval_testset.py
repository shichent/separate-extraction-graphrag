#!/usr/bin/env python
"""
Evaluate a fine-tuned adapter on its dataset's test split (or any dataset).
Computes Precision / Recall / F1 / label-precision using the notebook's formulas.

Usage:
    python eval_testset.py --adapter adapters/XXX_fiNERweb --dataset fiNERweb \
        --n 100 --out logs/eval_testset_fiNERweb.json
"""
import argparse
import json
import os
from datetime import datetime


def log(msg):
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


SYSTEM_PROMPT = (
    "You are an expert in named entity extraction. Extract named entities from the user's "
    "Chinese text, label them with types, and return results as a JSON dict mapping entity "
    "text to label. Entities must appear verbatim in the text. "
    'Example: {"北京": "location", "李明": "person"}'
)


def calculate_metrics(trues, preds):
    """Notebook's formulas, copy-pasted: P / R / F1 on entity strings, plus label precision."""
    tp = fp = fn = tl = 0
    for true_dict, pred_dict in zip(trues, preds):
        ts, ps = set(true_dict.keys()), set(pred_dict.keys())
        tp += len(ts & ps)
        fp += len(ps - ts)
        fn += len(ts - ps)
        for k in true_dict:
            if k in pred_dict and true_dict[k] == pred_dict[k]:
                tl += 1
    P = tp / (tp + fp) if (tp + fp) else 0.0
    R = tp / (tp + fn) if (tp + fn) else 0.0
    F = 2 * P * R / (P + R) if (P + R) else 0.0
    LP = tl / (tp + fp) if (tp + fp) else 0.0
    return {
        "precision": round(P, 4),
        "recall": round(R, 4),
        "f1": round(F, 4),
        "label_precision": round(LP, 4),
        "tp": tp, "fp": fp, "fn": fn, "tl": tl,
    }


def load_test_data(dataset, n):
    from datasets import load_dataset
    if dataset == "fiNERweb":
        # fiNERweb's train split is all there is on HF; use a held-out slice.
        ds = load_dataset("whoisjones/fiNERweb", "cmn",
                          split=f"train[10000:{10000 + n}]")
        out = []
        for ex in ds:
            text = ex["text"]
            target = {text[e["start"]:e["end"]]: e["label"] for e in ex["char_spans"]}
            out.append({"text": text, "entities": target})
        return out
    if dataset == "msra":
        ds = load_dataset("msra_ner", split=f"test[:{n}]", trust_remote_code=True)
        label_names = ds.features["ner_tags"].feature.names
        type_map = {"PER": "person", "ORG": "organization", "LOC": "location"}
        out = []
        for ex in ds:
            tokens = ex["tokens"]
            tag_names = [label_names[i] for i in ex["ner_tags"]]
            text = "".join(tokens)
            target, cur_type, cur_chars = {}, None, []
            for tok, name in zip(tokens, tag_names):
                if name == "O":
                    if cur_type:
                        target["".join(cur_chars)] = type_map[cur_type]
                        cur_type, cur_chars = None, []
                elif name.startswith("B-"):
                    if cur_type:
                        target["".join(cur_chars)] = type_map[cur_type]
                    cur_type, cur_chars = name[2:], [tok]
                elif name.startswith("I-") and cur_type == name[2:]:
                    cur_chars.append(tok)
                else:
                    if cur_type:
                        target["".join(cur_chars)] = type_map[cur_type]
                    cur_type, cur_chars = name[2:], [tok]
            if cur_type:
                target["".join(cur_chars)] = type_map[cur_type]
            out.append({"text": text, "entities": target})
        return out
    if dataset == "cluener":
        ds = load_dataset("nlhappy/CLUE-NER", split=f"train[10000:{10000 + n}]",
                          trust_remote_code=True)
        out = []
        for ex in ds:
            target = {e["text"]: e["label"] for e in ex["ents"] if e["is_continuous"]}
            out.append({"text": ex["text"], "entities": target})
        return out
    raise ValueError(f"unknown dataset: {dataset}")


def run_inference(adapter, examples, max_new_tokens):
    import torch
    from unsloth import FastLanguageModel
    torch._dynamo.config.cache_size_limit = 1_000_000
    torch._dynamo.config.accumulated_cache_size_limit = 1_000_000
    torch._dynamo.config.suppress_errors = True
    import json_repair

    log(f"loading {adapter}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=adapter,
        max_seq_length=2048,
        load_in_4bit=False,
        dtype=torch.bfloat16,
    )
    FastLanguageModel.for_inference(model)

    preds = []
    for i, ex in enumerate(examples):
        if i > 0 and i % 15 == 0:
            torch._dynamo.reset()
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"text: {ex['text']}"},
        ]
        try:
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=False)
        except TypeError:
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text=prompt, return_tensors="pt").to("cuda")
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.1,
            pad_token_id=tokenizer.eos_token_id,
        )
        gen = tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        parsed = json_repair.loads(gen)
        if not isinstance(parsed, dict):
            parsed = {}
        parsed = {k: v for k, v in parsed.items()
                  if isinstance(k, str) and isinstance(v, str)}
        preds.append(parsed)
        if (i + 1) % 25 == 0:
            log(f"  {i + 1}/{len(examples)} done")
    return preds


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--adapter", required=True)
    p.add_argument("--dataset", required=True, choices=["fiNERweb", "cluener", "msra"])
    p.add_argument("--n", type=int, default=100)
    p.add_argument("--out", required=True)
    p.add_argument("--max_new_tokens", type=int, default=512)
    args = p.parse_args()

    examples = load_test_data(args.dataset, args.n)
    log(f"test set: {len(examples)} examples from {args.dataset}")

    preds = run_inference(args.adapter, examples, args.max_new_tokens)

    trues = [ex["entities"] for ex in examples]
    metrics = calculate_metrics(trues, preds)
    log(f"P={metrics['precision']:.4f} R={metrics['recall']:.4f} "
        f"F1={metrics['f1']:.4f} LP={metrics['label_precision']:.4f}")

    report = {
        "adapter": args.adapter,
        "dataset": args.dataset,
        "n": args.n,
        "metrics": metrics,
        "predictions": [{"text": e["text"], "true": e["entities"], "pred": p}
                        for e, p in zip(examples, preds)],
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    log(f"wrote {args.out}")


if __name__ == "__main__":
    main()
