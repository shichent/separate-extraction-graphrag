#!/usr/bin/env python
"""
Extract named entities from a full Chinese novel using one fine-tuned adapter.
One process per adapter — keeps torch._dynamo cache fresh.

Chunking strategy: split at chapter boundaries ("第N章"/"第N回"), then sub-split
any chunk longer than MAX_CHARS into overlapping passages.

Output: JSONL, one record per passage:
    {"chunk_id": "1_c003", "text": "...", "entities": {"韩立": "person", ...}}
"""
import argparse
import json
import os
import re
from datetime import datetime


def log(msg):
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


SYSTEM_PROMPT = (
    "You are an expert in named entity extraction. Extract named entities from the user's "
    "Chinese text, label them with types, and return results as a JSON dict mapping entity "
    "text to label. Entities must appear verbatim in the text. "
    'Example: {"北京": "location", "李明": "person"}'
)

MAX_CHARS = 1200  # fits comfortably in 2048 seq_len with system prompt
CHAPTER_RE = re.compile(r"(第[一二三四五六七八九十百千零〇\d]+[章回])")


def chunk_novel(path):
    with open(path, encoding="utf-8") as f:
        text = f.read().lstrip("\ufeff")
    text = re.sub(r"[\u3000\t ]+", " ", text)  # normalise whitespace
    parts = CHAPTER_RE.split(text)
    # parts = [preamble, "第一章", chapter1_text, "第二回", chapter2_text, ...]
    chunks = []
    # preamble
    if parts and parts[0].strip():
        chunks.append(parts[0].strip())
    i = 1
    while i < len(parts):
        title = parts[i]
        body = parts[i + 1] if i + 1 < len(parts) else ""
        chapter = (title + " " + body).strip()
        chunks.append(chapter)
        i += 2

    # split any chapter longer than MAX_CHARS
    final = []
    for c in chunks:
        if len(c) <= MAX_CHARS:
            final.append(c)
        else:
            for start in range(0, len(c), MAX_CHARS):
                final.append(c[start:start + MAX_CHARS])
    # drop tiny fragments
    final = [c for c in final if len(c) >= 50]
    return final


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--adapter", required=True, help="path to LoRA adapter dir")
    p.add_argument("--novel", required=True, help="path to novel text file")
    p.add_argument("--out", required=True, help="output JSONL path")
    p.add_argument("--prefix", default="n",
                   help="chunk id prefix (e.g. '1' for 1.txt)")
    p.add_argument("--max_new_tokens", type=int, default=384)
    p.add_argument("--max_seq_length", type=int, default=2048)
    p.add_argument("--max_chunks", type=int, default=None,
                   help="if set, process only every Nth chunk so total <= max_chunks")
    return p.parse_args()


def main():
    args = parse_args()
    log(f"adapter={args.adapter} novel={args.novel} out={args.out}")

    import torch
    from unsloth import FastLanguageModel
    # Set AFTER unsloth import (unsloth overrides dynamo config on import).
    # Also disable compilation errors from being fatal.
    torch._dynamo.config.cache_size_limit = 1_000_000
    torch._dynamo.config.accumulated_cache_size_limit = 1_000_000
    torch._dynamo.config.suppress_errors = True
    import json_repair

    log(f"loading adapter {args.adapter}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.adapter,
        max_seq_length=args.max_seq_length,
        load_in_4bit=False,
        dtype=torch.bfloat16,
    )
    FastLanguageModel.for_inference(model)

    log(f"chunking {args.novel}")
    chunks = chunk_novel(args.novel)
    raw_count = len(chunks)
    if args.max_chunks and raw_count > args.max_chunks:
        stride = (raw_count + args.max_chunks - 1) // args.max_chunks
        chunks = chunks[::stride]
        log(f"subsampling: stride={stride}, {raw_count} -> {len(chunks)} chunks")
    log(f"{len(chunks)} chunks to process (max chars: {max(len(c) for c in chunks)}, "
        f"median: {sorted(len(c) for c in chunks)[len(chunks)//2]})")

    # stream results to disk so a crash mid-run still leaves partial data
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    written = 0
    with open(args.out, "w", encoding="utf-8") as fout:
        for i, text in enumerate(chunks):
            # Periodically clear dynamo compile cache — the per-chunk shape
            # variance otherwise fills the cache within ~30 chunks.
            if i > 0 and i % 15 == 0:
                torch._dynamo.reset()
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"text: {text}"},
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
                max_new_tokens=args.max_new_tokens,
                temperature=0.1,
                pad_token_id=tokenizer.eos_token_id,
            )
            gen = tokenizer.decode(
                out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
            parsed = json_repair.loads(gen)
            if not isinstance(parsed, dict):
                parsed = {}
            # filter: only keep entities that actually appear in source
            parsed = {k: v for k, v in parsed.items()
                      if isinstance(k, str) and isinstance(v, str) and k in text}

            record = {
                "chunk_id": f"{args.prefix}_c{i:04d}",
                "text": text,
                "entities": parsed,
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            fout.flush()
            written += 1
            if written % 25 == 0:
                log(f"  {written}/{len(chunks)} chunks done")

    log(f"wrote {written} records to {args.out}")


if __name__ == "__main__":
    main()
