#!/usr/bin/env python
"""
Build a novel eval set by sampling passages from 1.txt and 2.txt.
No gold labels — just (id, text) pairs for comparative inference.
"""
import argparse
import json
import os
import random


def slice_passages(path, passages_per_file, approx_len, seed):
    with open(path, encoding="utf-8") as f:
        text = f.read()
    # remove gutenberg-like headers / BOM
    text = text.lstrip("\ufeff")
    # find chapter markers (第N章 / 第N回) to anchor samples
    marks = []
    for pat in ["第一章", "第二章", "第三章", "第四章", "第五章", "第六章", "第七章",
                "第八章", "第九章", "第十章", "第二十章", "第三十章",
                "第一回", "第二回", "第三回", "第五回", "第十回"]:
        idx = text.find(pat)
        if idx >= 0:
            marks.append(idx)
    # fallback: regular intervals
    if len(marks) < passages_per_file:
        step = max(1, (len(text) - approx_len) // passages_per_file)
        marks.extend(range(1000, len(text) - approx_len, step))
    marks = sorted(set(marks))[:passages_per_file]

    passages = []
    base = os.path.basename(path).replace(".txt", "")
    for i, start in enumerate(marks):
        chunk = text[start:start + approx_len].strip()
        # drop excessive whitespace/blank lines
        chunk = " ".join(chunk.split())[:approx_len]
        passages.append({"id": f"{base}_p{i:02d}", "text": chunk})
    return passages


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sources", nargs="+", required=True)
    p.add_argument("--per_file", type=int, default=25)
    p.add_argument("--length", type=int, default=800)
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    random.seed(args.seed)

    all_p = []
    for src in args.sources:
        all_p.extend(slice_passages(src, args.per_file, args.length, args.seed))
    with open(args.out, "w", encoding="utf-8") as f:
        for ex in all_p:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    print(f"wrote {len(all_p)} passages to {args.out}")


if __name__ == "__main__":
    main()
