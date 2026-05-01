#!/usr/bin/env python
"""
General-domain Chinese NER fine-tune for the novel-transfer experiment.

One invocation = one GPU, one dataset. Launcher pins GPU via CUDA_VISIBLE_DEVICES.

Datasets (HF paths verified 2026-04-19):
    fiNERweb : whoisjones/fiNERweb (cmn)  char-spans, free-form labels
    msra     : msra_ner                    token-BIO, {PER, ORG, LOC}
    cluener  : nlhappy/CLUE-NER            spans, 10 CLUENER labels
"""
import argparse
import json
import os
from datetime import datetime


def log(msg):
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True,
                   choices=["fiNERweb", "cluener", "msra", "combined"])
    p.add_argument("--output_dir", required=True)
    p.add_argument("--model_name", default="unsloth/Qwen3.5-9B")
    p.add_argument("--max_seq_length", type=int, default=2048)
    p.add_argument("--train_samples", type=int, default=10000)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--grad_accum", type=int, default=4)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--lora_r", type=int, default=32)
    p.add_argument("--seed", type=int, default=3407)
    return p.parse_args()


SYSTEM_PROMPT = (
    "You are an expert in named entity extraction. Extract named entities from the user's "
    "Chinese text, label them with types, and return results as a JSON dict mapping entity "
    "text to label. Entities must appear verbatim in the text. "
    'Example: {"北京": "location", "李明": "person"}'
)


def build_chat(tokenizer, text, target_dict):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"text: {text}"},
        {"role": "assistant", "content": json.dumps(target_dict, ensure_ascii=False)},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)


def load_fiNERweb(tokenizer, n):
    from datasets import load_dataset
    ds = load_dataset("whoisjones/fiNERweb", "cmn", split=f"train[:{n}]")

    def map_batch(ex):
        rows = []
        for text, ents in zip(ex["text"], ex["char_spans"]):
            target = {text[e["start"]:e["end"]]: e["label"] for e in ents}
            rows.append(build_chat(tokenizer, text, target))
        return {"text": rows}

    return ds.map(map_batch, batched=True, remove_columns=ds.column_names)


_MSRA_TYPE_MAP = {"PER": "person", "ORG": "organization", "LOC": "location"}


def _bio_to_entities(tokens, tag_names):
    """Reconstruct {entity_string: label} from char-level BIO tags."""
    out = {}
    cur_type, cur_chars = None, []
    for tok, name in zip(tokens, tag_names):
        if name == "O":
            if cur_type is not None:
                out["".join(cur_chars)] = _MSRA_TYPE_MAP[cur_type]
                cur_type, cur_chars = None, []
        elif name.startswith("B-"):
            if cur_type is not None:
                out["".join(cur_chars)] = _MSRA_TYPE_MAP[cur_type]
            cur_type, cur_chars = name[2:], [tok]
        elif name.startswith("I-") and cur_type == name[2:]:
            cur_chars.append(tok)
        else:
            if cur_type is not None:
                out["".join(cur_chars)] = _MSRA_TYPE_MAP[cur_type]
            cur_type, cur_chars = name[2:], [tok]
    if cur_type is not None:
        out["".join(cur_chars)] = _MSRA_TYPE_MAP[cur_type]
    return out


def load_msra(tokenizer, n):
    from datasets import load_dataset
    ds = load_dataset("msra_ner", split=f"train[:{n}]", trust_remote_code=True)
    label_names = ds.features["ner_tags"].feature.names

    def map_batch(ex):
        rows = []
        for tokens, tag_ids in zip(ex["tokens"], ex["ner_tags"]):
            text = "".join(tokens)
            target = _bio_to_entities(tokens, [label_names[i] for i in tag_ids])
            rows.append(build_chat(tokenizer, text, target))
        return {"text": rows}

    return ds.map(map_batch, batched=True, remove_columns=ds.column_names)


def load_cluener(tokenizer, n):
    from datasets import load_dataset
    ds = load_dataset("nlhappy/CLUE-NER", split=f"train[:{n}]", trust_remote_code=True)

    def map_batch(ex):
        rows = []
        for text, ents in zip(ex["text"], ex["ents"]):
            target = {e["text"]: e["label"] for e in ents if e["is_continuous"]}
            rows.append(build_chat(tokenizer, text, target))
        return {"text": rows}

    return ds.map(map_batch, batched=True, remove_columns=ds.column_names)


def load_combined(tokenizer, n):
    """Equal-parts mix of fiNERweb + cluener + msra, shuffled."""
    from datasets import concatenate_datasets
    per = n // 3
    parts = [
        load_fiNERweb(tokenizer, per),
        load_cluener(tokenizer, per),
        load_msra(tokenizer, per),
    ]
    ds = concatenate_datasets(parts).shuffle(seed=3407)
    return ds


LOADERS = {
    "fiNERweb": load_fiNERweb,
    "cluener": load_cluener,
    "msra": load_msra,
    "combined": load_combined,
}


def main():
    args = parse_args()
    log(f"dataset={args.dataset} gpu={os.environ.get('CUDA_VISIBLE_DEVICES', 'unset')} "
        f"model={args.model_name}")
    os.makedirs(args.output_dir, exist_ok=True)

    import torch
    from unsloth import FastLanguageModel
    from trl import SFTTrainer, SFTConfig

    log(f"loading base model: {args.model_name}")
    try:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=args.model_name,
            max_seq_length=args.max_seq_length,
            load_in_4bit=False,
            dtype=torch.bfloat16,
        )
    except Exception as e:
        fallback = "unsloth/Qwen2.5-7B"
        log(f"PRIMARY MODEL LOAD FAILED ({type(e).__name__}: {e}). "
            f"Falling back to {fallback}")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=fallback,
            max_seq_length=args.max_seq_length,
            load_in_4bit=False,
            dtype=torch.bfloat16,
        )
        args.model_name = fallback
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha=args.lora_r,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=args.seed,
    )

    log("loading training data")
    dataset = LOADERS[args.dataset](tokenizer, args.train_samples)
    log(f"dataset size: {len(dataset)}")

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        args=SFTConfig(
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            num_train_epochs=args.epochs,
            learning_rate=args.lr,
            bf16=True,
            optim="adamw_8bit",
            output_dir=args.output_dir,
            save_strategy="no",
            logging_steps=50,
            report_to="none",
            seed=args.seed,
        ),
    )

    log("starting training")
    trainer.train()

    log(f"saving adapter to {args.output_dir}")
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    log("done")


if __name__ == "__main__":
    main()
