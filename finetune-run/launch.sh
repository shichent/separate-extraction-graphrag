#!/usr/bin/env bash
# Server-polite launcher for NER fine-tuning runs.
#
# Safety features:
#   - refuses to start if any target GPU already has >2GB used
#   - nice/ionice to yield to other users
#   - per-job `timeout` cap so a runaway can't burn indefinitely
#   - staggered model loads (avoid 3x simultaneous HF downloads)
#   - PID file so kill_all.sh can clean up
#
# Usage:
#   ./launch.sh                     # default: fiNERweb on GPU 0 only (safe first run)
#   JOBS="0:fiNERweb"   ./launch.sh # explicit single run
#   JOBS="0:fiNERweb 1:cluener 2:msra" ./launch.sh   # full 3-way
#
# After launch:
#   tail -f logs/<stamp>_gpu0_fiNERweb.log
#   ./kill_all.sh  (see bottom of this file for how it's built)

set -euo pipefail

PROJ_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "${PROJ_DIR}"
mkdir -p logs adapters

ENV_NAME="${ENV_NAME:-ner_ft}"
TIMEOUT_HOURS="${TIMEOUT_HOURS:-12}"
STAGGER_SECS="${STAGGER_SECS:-30}"
JOBS="${JOBS:-0:fiNERweb}"   # default = single run for tonight's smoke test

STAMP=$(date +%Y%m%d_%H%M%S)
PID_FILE="logs/${STAMP}.pids"
: > "${PID_FILE}"

# --- preflight: check all target GPUs are really idle -----------------
echo "[launch] preflight GPU check"
for spec in ${JOBS}; do
    gpu="${spec%%:*}"
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "${gpu}")
    if [ "${used}" -gt 2048 ]; then
        echo "[launch] ABORT: GPU ${gpu} has ${used} MiB in use. Someone else may be training." >&2
        nvidia-smi
        exit 1
    fi
    echo "[launch]   GPU ${gpu}: ${used} MiB used -> ok"
done

# --- preflight: confirm conda env exists --------------------------------
if ! bash -lc "conda env list | awk '{print \$1}' | grep -qx '${ENV_NAME}'"; then
    echo "[launch] ABORT: conda env '${ENV_NAME}' not found. Run setup_env.sh first." >&2
    exit 1
fi

# --- launch -----------------------------------------------------------
for spec in ${JOBS}; do
    gpu="${spec%%:*}"
    ds="${spec##*:}"
    log="logs/${STAMP}_gpu${gpu}_${ds}.log"
    out="adapters/${STAMP}_${ds}"
    mkdir -p "${out}"

    echo "[launch] GPU ${gpu} -> ${ds}  (log: ${log})"

    CUDA_VISIBLE_DEVICES="${gpu}" \
    HF_HUB_DISABLE_PROGRESS_BARS=1 \
    TRANSFORMERS_VERBOSITY=warning \
    nohup nice -n 10 ionice -c 2 -n 7 \
        timeout "${TIMEOUT_HOURS}h" \
        bash -lc "conda activate ${ENV_NAME} && python '${PROJ_DIR}/train_ner.py' \
            --dataset '${ds}' \
            --output_dir '${PROJ_DIR}/${out}'" \
        > "${log}" 2>&1 &

    pid=$!
    echo "${pid}" >> "${PID_FILE}"
    echo "[launch]   pid ${pid}"

    # stagger so model loads don't collide on HF cache / NFS
    sleep "${STAGGER_SECS}"
done

echo ""
echo "[launch] all jobs launched. stamp=${STAMP}"
echo "[launch] monitor:  tail -f ${PROJ_DIR}/logs/${STAMP}_*.log"
echo "[launch] gpus:     watch -n 5 nvidia-smi"
echo "[launch] kill all: xargs -a ${PID_FILE} kill -TERM"
