#!/usr/bin/env bash
# Overnight orchestrator: env setup -> smoke -> 3 parallel trainings -> eval -> report.
#
# Designed to be launched ONCE with nohup in the background and run unattended.
# Writes a running status report to logs/overnight_<STAMP>.md — the user can
# tail that file at any time to see progress.
#
# Safety:
#   - aborts before training if any GPU is already in use
#   - `timeout 8h` on every training job
#   - nice/ionice to yield to other users
#   - save_strategy="no" in train_ner.py (only final adapter kept)
#   - no model checkpoints, only LoRA adapters (~100-500MB each)

set -uo pipefail  # no -e: we want to capture failures and still produce a report

PROJ="/home/hh68/Desktop/Homework/COMP584_project/run"
cd "$PROJ"
STAMP=$(date +%Y%m%d_%H%M%S)
REPORT="${PROJ}/logs/overnight_${STAMP}.md"
mkdir -p logs adapters

say() {
    echo "[$(date +%F\ %T)] $*" | tee -a "$REPORT"
}

say "# Overnight run ${STAMP}"
say ""
say "Host: $(hostname)   PWD: ${PROJ}"
nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv | tee -a "$REPORT"
say ""

# ===== Step 1: env setup (idempotent) =====
say "## Step 1: env setup"
if bash -lc "conda env list | awk '{print \$1}' | grep -qx 'ner_ft'" 2>/dev/null; then
    say "   env 'ner_ft' already exists — skipping creation"
else
    say "   creating env 'ner_ft' (takes ~30 min)"
    bash setup_env.sh >> "logs/overnight_${STAMP}_setup.log" 2>&1
    rc=$?
    if [ $rc -ne 0 ]; then
        say "   ERROR: setup_env.sh failed (rc=$rc). Aborting."
        say "   See logs/overnight_${STAMP}_setup.log"
        exit 1
    fi
    say "   env ready"
fi

# ===== Step 2: smoke test (100 samples, 1 GPU, ~5 min) =====
say ""
say "## Step 2: smoke test"
SMOKE_LOG="logs/overnight_${STAMP}_smoke.log"
CUDA_VISIBLE_DEVICES=0 HF_HUB_DISABLE_PROGRESS_BARS=1 \
    nice -n 10 ionice -c 2 -n 7 \
    bash -lc "conda activate ner_ft && python train_ner.py \
        --dataset fiNERweb --output_dir adapters/${STAMP}_smoke \
        --train_samples 100 --epochs 1 --batch_size 4 --grad_accum 2" \
    > "$SMOKE_LOG" 2>&1
rc=$?
if [ $rc -ne 0 ]; then
    say "   ERROR: smoke test failed (rc=$rc). Aborting before launching 3 runs."
    tail -30 "$SMOKE_LOG" | sed 's/^/    /' | tee -a "$REPORT"
    exit 1
fi
say "   smoke test passed"
# free smoke adapter to save disk (keep only the final, full trainings)
rm -rf "adapters/${STAMP}_smoke"
say "   cleaned up smoke adapter"

# ===== Step 3: launch 3 parallel trainings =====
say ""
say "## Step 3: launch 3 parallel trainings (fiNERweb, cluener, msra)"

# Preflight: confirm all 3 GPUs are idle before pinning jobs
for gpu in 0 1 2; do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "${gpu}")
    if [ "${used}" -gt 2048 ]; then
        say "   ERROR: GPU ${gpu} has ${used} MiB in use by another process. Aborting 3-way."
        say "   You can manually launch fewer jobs with: JOBS='0:fiNERweb' ./launch.sh"
        exit 1
    fi
done

declare -A PIDS
for gpu_ds in "0:fiNERweb" "1:cluener" "2:msra"; do
    gpu="${gpu_ds%%:*}"
    ds="${gpu_ds##*:}"
    out="adapters/${STAMP}_${ds}"
    log="logs/overnight_${STAMP}_gpu${gpu}_${ds}.log"
    say "   launching ${ds} on GPU ${gpu} -> ${log}"
    CUDA_VISIBLE_DEVICES="${gpu}" \
        HF_HUB_DISABLE_PROGRESS_BARS=1 \
        TRANSFORMERS_VERBOSITY=warning \
        nohup nice -n 10 ionice -c 2 -n 7 \
            timeout 8h \
            bash -lc "conda activate ner_ft && python train_ner.py \
                --dataset ${ds} --output_dir ${out}" \
            > "${log}" 2>&1 &
    PIDS[$ds]=$!
    sleep 30   # stagger HF model loads
done

say "   all 3 launched. PIDs: fiNERweb=${PIDS[fiNERweb]} cluener=${PIDS[cluener]} msra=${PIDS[msra]}"

# ===== Step 4: wait for all trainings =====
say ""
say "## Step 4: waiting for training to finish (up to 8h each)"
START=$(date +%s)
declare -A RC
for ds in fiNERweb cluener msra; do
    wait "${PIDS[$ds]}"
    RC[$ds]=$?
    say "   ${ds} finished (rc=${RC[$ds]}) at $(date +%F\ %T)"
done
END=$(date +%s)
say "   total training wall-clock: $(( (END - START) / 60 )) min"

# ===== Step 5: eval =====
say ""
say "## Step 5: building novel eval set + running all 4 models"

# Build 50-passage eval set from 1.txt and 2.txt
NOVELS="${PROJ}/../NLP project-20260419T053035Z-3-001/NLP project/datasets"
EVAL_SET="${PROJ}/eval_passages_${STAMP}.jsonl"
bash -lc "conda activate ner_ft && python build_eval_set.py \
    --sources '${NOVELS}/1.txt' '${NOVELS}/2.txt' \
    --per_file 25 --length 800 --out '${EVAL_SET}'" >> "$REPORT" 2>&1

# Assemble adapter list from what actually succeeded
ADAPTERS=""
for ds in fiNERweb cluener msra; do
    path="adapters/${STAMP}_${ds}"
    if [ -d "$path" ] && [ -f "$path/adapter_config.json" ]; then
        ADAPTERS="${ADAPTERS} ${ds}:${path}"
    else
        say "   skipping ${ds} (no adapter produced)"
    fi
done

EVAL_OUT="${PROJ}/logs/overnight_${STAMP}_eval.json"
CUDA_VISIBLE_DEVICES=0 HF_HUB_DISABLE_PROGRESS_BARS=1 \
    nice -n 10 ionice -c 2 -n 7 \
    bash -lc "conda activate ner_ft && python eval_overnight.py \
        --base_model unsloth/Qwen3.5-9B \
        --adapters ${ADAPTERS} \
        --passages '${EVAL_SET}' \
        --out '${EVAL_OUT}'" \
    >> "logs/overnight_${STAMP}_eval.log" 2>&1
rc=$?
if [ $rc -ne 0 ]; then
    say "   WARNING: eval failed (rc=$rc). See logs/overnight_${STAMP}_eval.log"
else
    say "   eval done -> ${EVAL_OUT}"
fi

# ===== Step 6: bottom-line summary =====
say ""
say "## Bottom line"
say ""
for ds in fiNERweb cluener msra; do
    rc=${RC[$ds]:-?}
    path="adapters/${STAMP}_${ds}"
    if [ -d "$path" ] && [ -f "$path/adapter_config.json" ]; then
        say "   ✓ ${ds}: adapter saved (rc=${rc})"
    else
        say "   ✗ ${ds}: no adapter (rc=${rc})"
    fi
done

# append human-readable eval markdown
if [ -f "${EVAL_OUT%.json}.md" ]; then
    say ""
    cat "${EVAL_OUT%.json}.md" >> "$REPORT"
fi

say ""
say "## DONE at $(date +%F\ %T)"
