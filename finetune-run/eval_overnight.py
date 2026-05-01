#!/usr/bin/env python
"""
Overnight eval: compare zero-shot base vs N fine-tuned adapters on Chinese novel
passages. No gold labels (intentional). Reports counts, label distributions, and
pairwise agreement — enough to show whether general-NER fine-tuning changes model
behavior on novels.

Usage:
    python eval_overnight.py \
        --base_model unsloth/Qwen3.5-9B \
        --adapters fiNERweb:adapters/XXX_fiNERweb cluener:adapters/XXX_cluener msra:adapters/XXX_msra \
        --passages passages.jsonl \
        --out eval_report.json
"""
import argparse
import json
import os
from collections import Counter
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
    p.add_argument("--base_model", default="unsloth/Qwen3.5-9B")
    p.add_argument("--adapters", nargs="*", default=[],
                   help="name:path items; omit to eval only zero-shot")
    p.add_argument("--passages", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--max_new_tokens", type=int, default=512)
    return p.parse_args()


def load_passages(path):
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def run_one_model(model_path, passages, max_new_tokens):
    import torch
    # torch._dynamo's cache is per-process and accumulates across models.
    # With 4 models × varying sequence shapes, the default limit is trivial.
    torch._dynamo.config.cache_size_limit = 1_000_000
    torch._dynamo.config.accumulated_cache_size_limit = 1_000_000
    torch._dynamo.reset()
    from unsloth import FastLanguageModel
    import json_repair

    log(f"loading {model_path}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=2048,
        load_in_4bit=False,
        dtype=torch.bfloat16,
    )
    FastLanguageModel.for_inference(model)

    results = []
    for i, ex in enumerate(passages):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"text: {ex['text']}"},
        ]
        # Qwen3.5 is a VLM: must pass `text=` kwarg or the processor tries to
        # interpret the prompt as an image. Also disable "thinking" mode so the
        # model emits the JSON directly without a <think>...</think> preamble.
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
        gen = tokenizer.decode(
            out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        parsed = json_repair.loads(gen)
        if not isinstance(parsed, dict):
            parsed = {}
        # keep only entities that actually appear in the source text
        parsed = {k: v for k, v in parsed.items()
                  if isinstance(k, str) and isinstance(v, str) and k in ex["text"]}
        results.append({"id": ex["id"], "entities": parsed})
        if (i + 1) % 10 == 0:
            log(f"  {i+1}/{len(passages)} passages done")

    # free GPU
    del model, tokenizer
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    return results


def summarize(model_name, results):
    n_passages = len(results)
    n_entities = sum(len(r["entities"]) for r in results)
    unique_entities = set()
    label_counts = Counter()
    per_passage_counts = []
    for r in results:
        per_passage_counts.append(len(r["entities"]))
        for k, v in r["entities"].items():
            unique_entities.add(k)
            label_counts[v.lower()] += 1
    return {
        "model": model_name,
        "n_passages": n_passages,
        "total_entities": n_entities,
        "unique_entities": len(unique_entities),
        "mean_entities_per_passage": round(n_entities / max(n_passages, 1), 2),
        "top_labels": label_counts.most_common(10),
    }


def pairwise_jaccard(all_results):
    """Jaccard of entity-string sets between each pair of models, averaged over passages."""
    names = list(all_results.keys())
    matrix = {}
    for a in names:
        matrix[a] = {}
        for b in names:
            scores = []
            for ra, rb in zip(all_results[a], all_results[b]):
                sa = set(ra["entities"].keys())
                sb = set(rb["entities"].keys())
                if not sa and not sb:
                    continue
                scores.append(len(sa & sb) / max(len(sa | sb), 1))
            matrix[a][b] = round(sum(scores) / max(len(scores), 1), 3)
    return matrix


def main():
    args = parse_args()
    passages = load_passages(args.passages)
    log(f"eval set: {len(passages)} passages")

    # Parse adapter specs
    adapter_specs = {}
    for item in args.adapters:
        if ":" in item:
            name, path = item.split(":", 1)
            if os.path.isdir(path):
                adapter_specs[name] = path
            else:
                log(f"WARNING: adapter path missing, skipping: {path}")

    all_results = {}

    # Zero-shot baseline
    log("=== zero-shot baseline ===")
    try:
        all_results["zeroshot"] = run_one_model(args.base_model, passages, args.max_new_tokens)
    except Exception as e:
        log(f"ERROR zero-shot: {type(e).__name__}: {e}")

    # Each adapter
    for name, path in adapter_specs.items():
        log(f"=== adapter: {name} ({path}) ===")
        try:
            all_results[name] = run_one_model(path, passages, args.max_new_tokens)
        except Exception as e:
            log(f"ERROR {name}: {type(e).__name__}: {e}")

    if not all_results:
        log("no model produced results; aborting")
        return

    summaries = {k: summarize(k, v) for k, v in all_results.items()}
    agreement = pairwise_jaccard(all_results)

    # A few sample passages shown across all models
    sample_idx = list(range(0, len(passages), max(1, len(passages) // 5)))[:5]
    samples = []
    for i in sample_idx:
        samples.append({
            "id": passages[i]["id"],
            "text_preview": passages[i]["text"][:200],
            "outputs": {k: v[i]["entities"] for k, v in all_results.items()},
        })

    report = {
        "eval_time": datetime.now().isoformat(),
        "base_model": args.base_model,
        "n_passages": len(passages),
        "models": list(all_results.keys()),
        "summaries": summaries,
        "pairwise_jaccard": agreement,
        "samples": samples,
    }
    with open(args.out, "w") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    log(f"wrote {args.out}")

    # Human-readable bottom line
    bl = args.out.replace(".json", ".md")
    with open(bl, "w") as f:
        f.write(f"# Eval report ({report['eval_time']})\n\n")
        f.write(f"Passages: {report['n_passages']}\n\n")
        f.write("## Per-model summary\n\n")
        f.write("| model | total ents | unique ents | mean/passage | top labels |\n")
        f.write("|---|---|---|---|---|\n")
        for k, s in summaries.items():
            top = ", ".join(f"{lbl}={cnt}" for lbl, cnt in s["top_labels"][:5])
            f.write(f"| {k} | {s['total_entities']} | {s['unique_entities']} | "
                    f"{s['mean_entities_per_passage']} | {top} |\n")
        f.write("\n## Pairwise Jaccard (entity strings)\n\n")
        f.write("| | " + " | ".join(agreement.keys()) + " |\n")
        f.write("|" + "---|" * (len(agreement) + 1) + "\n")
        for a, row in agreement.items():
            f.write(f"| {a} | " + " | ".join(str(row[b]) for b in agreement.keys()) + " |\n")
        f.write("\n## Sample passages\n\n")
        for s in samples:
            f.write(f"### {s['id']}\n\n`{s['text_preview']}...`\n\n")
            for k, ents in s["outputs"].items():
                ent_str = ", ".join(f"{n}:{l}" for n, l in list(ents.items())[:8])
                f.write(f"- **{k}**: {ent_str}\n")
            f.write("\n")
    log(f"wrote {bl}")


if __name__ == "__main__":
    main()
