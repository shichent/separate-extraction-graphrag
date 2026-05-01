#!/usr/bin/env python
"""
Evaluation skeleton: run NER (zero-shot or LoRA-adapted) on novel passages.

Usage:
    # zero-shot baseline
    python eval_novel.py --eval_file eval_set.jsonl --out zeroshot.json

    # fine-tuned
    python eval_novel.py --eval_file eval_set.jsonl --out ft.json \
                         --adapter adapters/20260419_fiNERweb

eval_set.jsonl format (one object per line):
    {"id": "...", "text": "...", "entities": {"韩立": "person", "初圣宗": "organization"}}

Only PER and LOC are scored by default (matches AnonyRAG schema).
"""
import argparse
import json
import os
import sys
from datetime import datetime


def log(msg):
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


SYSTEM_PROMPT = (
    "You are an expert in named entity extraction. Extract named entities from the user's "
    "Chinese text, label them with types, and return results as a JSON dict mapping entity "
    "text to label. Entities must appear verbatim in the text. "
    'Example: {"北京": "location", "李明": "person"}'
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--eval_file", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--model_name", default="unsloth/Qwen2.5-7B")
    p.add_argument("--adapter", default=None, help="path to LoRA adapter; omit for zero-shot")
    p.add_argument("--max_new_tokens", type=int, default=512)
    p.add_argument("--score_labels", nargs="+", default=["person", "location"])
    return p.parse_args()


def main():
    args = parse_args()
    import torch
    from unsloth import FastLanguageModel
    import json_repair

    log(f"loading model (adapter={args.adapter})")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.adapter if args.adapter else args.model_name,
        max_seq_length=2048,
        load_in_4bit=False,
        dtype=torch.bfloat16,
    )
    FastLanguageModel.for_inference(model)

    examples = []
    with open(args.eval_file) as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    log(f"eval set: {len(examples)} passages")

    predictions = []
    for ex in examples:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"text: {ex['text']}"},
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        out = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            temperature=0.1,
            pad_token_id=tokenizer.eos_token_id,
        )
        gen = tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        parsed = json_repair.loads(gen)
        if not isinstance(parsed, dict):
            parsed = {}
        predictions.append({"id": ex.get("id"), "pred": parsed, "raw": gen})

    # score (span-only and label-conditional, restricted to score_labels)
    def norm(d):
        return {k: v.lower() for k, v in d.items() if v.lower() in args.score_labels}

    tp = fp = fn = tl = 0
    for ex, p in zip(examples, predictions):
        gt = norm(ex["entities"])
        pr = norm(p["pred"])
        gt_keys, pr_keys = set(gt), set(pr)
        tp += len(gt_keys & pr_keys)
        fp += len(pr_keys - gt_keys)
        fn += len(gt_keys - pr_keys)
        for k in gt_keys & pr_keys:
            if gt[k] == pr[k]:
                tl += 1

    P = tp / (tp + fp) if (tp + fp) else 0.0
    R = tp / (tp + fn) if (tp + fn) else 0.0
    F = 2 * P * R / (P + R) if (P + R) else 0.0
    LP = tl / (tp + fp) if (tp + fp) else 0.0

    report = {
        "adapter": args.adapter,
        "base_model": args.model_name,
        "n_examples": len(examples),
        "score_labels": args.score_labels,
        "metrics": {"precision": round(P, 4), "recall": round(R, 4),
                    "f1": round(F, 4), "label_precision": round(LP, 4)},
        "predictions": predictions,
    }
    with open(args.out, "w") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    log(f"wrote {args.out}")
    log(f"P={P:.4f} R={R:.4f} F1={F:.4f} LP={LP:.4f}")


if __name__ == "__main__":
    main()
