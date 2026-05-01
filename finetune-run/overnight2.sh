#!/usr/bin/env bash
# Tonight's pipeline (2026-04-20):
#   Phase 1: 3 parallel novel extractions (one adapter per GPU, max 500 chunks/novel) ~1.8 h
#   Phase 2: 3 parallel test-set P/R/F1 evals                                         ~0.5 h
#   Phase 3: combined-dataset fine-tune on GPU 0                                      ~4 h
#   Phase 4: combined-adapter test-set eval + novel extraction                        ~1.3 h
#   Phase 5: compile presentation summary                                             ~1 min
#
# NOTE: launches are inlined (no function + $() wrapper) — a subshell around & kills
#       the nohup'd process on subshell exit. Ugly but correct.

set -uo pipefail

PROJ="/home/hh68/Desktop/Homework/COMP584_project/run"
cd "$PROJ"
STAMP=$(date +%Y%m%d_%H%M%S)
REPORT="${PROJ}/logs/overnight2_${STAMP}.md"
mkdir -p logs adapters predictions

A_FIN="adapters/20260419_130003_fiNERweb"
A_CLU="adapters/20260419_130003_cluener"
A_MSR="adapters/20260419_130003_msra"

NOVELS="${PROJ}/../NLP project-20260419T053035Z-3-001/NLP project/datasets"
N1="${NOVELS}/1.txt"
N2="${NOVELS}/2.txt"

say() { echo "[$(date +%F\ %T)] $*" | tee -a "$REPORT"; }

say "# Tonight's pipeline  ${STAMP}"
say "Host: $(hostname)   Adapters: ${A_FIN}, ${A_CLU}, ${A_MSR}"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv | tee -a "$REPORT"

for gpu in 0 1 2; do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "${gpu}")
    if [ "${used}" -gt 2048 ]; then
        say "ABORT: GPU ${gpu} has ${used} MiB used." ; exit 1
    fi
done
for d in "$A_FIN" "$A_CLU" "$A_MSR"; do
    [ -d "$d" ] || { say "ABORT: missing adapter $d"; exit 1; }
done
[ -f "$N1" ] || { say "ABORT: missing $N1"; exit 1; }
[ -f "$N2" ] || { say "ABORT: missing $N2"; exit 1; }
say "preflight OK"

# ============ Phase 1: full-novel extraction (3 parallel) ============
say ""
say "## Phase 1: novel extraction (each adapter -> 1.txt then 2.txt, max 500 chunks/novel)"

CUDA_VISIBLE_DEVICES=0 HF_HUB_DISABLE_PROGRESS_BARS=1 \
    nohup nice -n 10 ionice -c 2 -n 7 timeout 4h \
    bash -lc "conda activate ner_ft && \
        python extract_novels.py --max_chunks 500 --adapter '${A_FIN}' --novel '${N1}' --prefix 1 --out 'predictions/${STAMP}_fiNERweb_1.jsonl' && \
        python extract_novels.py --max_chunks 500 --adapter '${A_FIN}' --novel '${N2}' --prefix 2 --out 'predictions/${STAMP}_fiNERweb_2.jsonl'" \
    > "logs/overnight2_${STAMP}_p1_gpu0_fiNERweb.log" 2>&1 &
P1_FIN=$!
say "  GPU 0 -> fiNERweb (pid ${P1_FIN})"
sleep 30

CUDA_VISIBLE_DEVICES=1 HF_HUB_DISABLE_PROGRESS_BARS=1 \
    nohup nice -n 10 ionice -c 2 -n 7 timeout 4h \
    bash -lc "conda activate ner_ft && \
        python extract_novels.py --max_chunks 500 --adapter '${A_CLU}' --novel '${N1}' --prefix 1 --out 'predictions/${STAMP}_cluener_1.jsonl' && \
        python extract_novels.py --max_chunks 500 --adapter '${A_CLU}' --novel '${N2}' --prefix 2 --out 'predictions/${STAMP}_cluener_2.jsonl'" \
    > "logs/overnight2_${STAMP}_p1_gpu1_cluener.log" 2>&1 &
P1_CLU=$!
say "  GPU 1 -> cluener (pid ${P1_CLU})"
sleep 30

CUDA_VISIBLE_DEVICES=2 HF_HUB_DISABLE_PROGRESS_BARS=1 \
    nohup nice -n 10 ionice -c 2 -n 7 timeout 4h \
    bash -lc "conda activate ner_ft && \
        python extract_novels.py --max_chunks 500 --adapter '${A_MSR}' --novel '${N1}' --prefix 1 --out 'predictions/${STAMP}_msra_1.jsonl' && \
        python extract_novels.py --max_chunks 500 --adapter '${A_MSR}' --novel '${N2}' --prefix 2 --out 'predictions/${STAMP}_msra_2.jsonl'" \
    > "logs/overnight2_${STAMP}_p1_gpu2_msra.log" 2>&1 &
P1_MSR=$!
say "  GPU 2 -> msra (pid ${P1_MSR})"

wait "$P1_FIN"; say "  fiNERweb finished (rc=$?)"
wait "$P1_CLU"; say "  cluener finished (rc=$?)"
wait "$P1_MSR"; say "  msra finished (rc=$?)"

# ============ Phase 2: test-set P/R/F1 ============
say ""
say "## Phase 2: test-set P/R/F1 (n=100 each)"

CUDA_VISIBLE_DEVICES=0 HF_HUB_DISABLE_PROGRESS_BARS=1 \
    nohup nice -n 10 ionice -c 2 -n 7 timeout 1h \
    bash -lc "conda activate ner_ft && python eval_testset.py --adapter '${A_FIN}' --dataset fiNERweb --n 100 --out 'logs/overnight2_${STAMP}_testset_fiNERweb.json'" \
    > "logs/overnight2_${STAMP}_p2_fiNERweb.log" 2>&1 &
P2_FIN=$!
sleep 15
CUDA_VISIBLE_DEVICES=1 HF_HUB_DISABLE_PROGRESS_BARS=1 \
    nohup nice -n 10 ionice -c 2 -n 7 timeout 1h \
    bash -lc "conda activate ner_ft && python eval_testset.py --adapter '${A_CLU}' --dataset cluener --n 100 --out 'logs/overnight2_${STAMP}_testset_cluener.json'" \
    > "logs/overnight2_${STAMP}_p2_cluener.log" 2>&1 &
P2_CLU=$!
sleep 15
CUDA_VISIBLE_DEVICES=2 HF_HUB_DISABLE_PROGRESS_BARS=1 \
    nohup nice -n 10 ionice -c 2 -n 7 timeout 1h \
    bash -lc "conda activate ner_ft && python eval_testset.py --adapter '${A_MSR}' --dataset msra --n 100 --out 'logs/overnight2_${STAMP}_testset_msra.json'" \
    > "logs/overnight2_${STAMP}_p2_msra.log" 2>&1 &
P2_MSR=$!

say "  launched: p2_fiNERweb=${P2_FIN}, p2_cluener=${P2_CLU}, p2_msra=${P2_MSR}"
wait "$P2_FIN"; say "  fiNERweb test (rc=$?)"
wait "$P2_CLU"; say "  cluener test (rc=$?)"
wait "$P2_MSR"; say "  msra test (rc=$?)"

# ============ Phase 3: combined fine-tune (GPU 0) ============
say ""
say "## Phase 3: combined-dataset fine-tune (9999 samples = 3333 x 3)"
P3_ADAPTER="adapters/${STAMP}_combined"

CUDA_VISIBLE_DEVICES=0 HF_HUB_DISABLE_PROGRESS_BARS=1 \
    nohup nice -n 10 ionice -c 2 -n 7 timeout 8h \
    bash -lc "conda activate ner_ft && python train_ner.py --dataset combined --output_dir '${P3_ADAPTER}' --train_samples 9999" \
    > "logs/overnight2_${STAMP}_p3_combined_train.log" 2>&1 &
P3=$!
say "  GPU 0 -> combined training (pid ${P3})"

wait "$P3"; say "  combined training finished (rc=$?)"

# ============ Phase 4: combined adapter evaluations ============
say ""
say "## Phase 4: combined adapter test-set evals + novel extraction"

CUDA_VISIBLE_DEVICES=0 HF_HUB_DISABLE_PROGRESS_BARS=1 \
    nohup nice -n 10 ionice -c 2 -n 7 timeout 1h \
    bash -lc "conda activate ner_ft && python eval_testset.py --adapter '${P3_ADAPTER}' --dataset fiNERweb --n 100 --out 'logs/overnight2_${STAMP}_testset_combined_fiNERweb.json'" \
    > "logs/overnight2_${STAMP}_p4_combined_fin.log" 2>&1 &
P4_FIN=$!
sleep 15
CUDA_VISIBLE_DEVICES=1 HF_HUB_DISABLE_PROGRESS_BARS=1 \
    nohup nice -n 10 ionice -c 2 -n 7 timeout 1h \
    bash -lc "conda activate ner_ft && python eval_testset.py --adapter '${P3_ADAPTER}' --dataset cluener --n 100 --out 'logs/overnight2_${STAMP}_testset_combined_cluener.json'" \
    > "logs/overnight2_${STAMP}_p4_combined_clu.log" 2>&1 &
P4_CLU=$!
sleep 15
CUDA_VISIBLE_DEVICES=2 HF_HUB_DISABLE_PROGRESS_BARS=1 \
    nohup nice -n 10 ionice -c 2 -n 7 timeout 1h \
    bash -lc "conda activate ner_ft && python eval_testset.py --adapter '${P3_ADAPTER}' --dataset msra --n 100 --out 'logs/overnight2_${STAMP}_testset_combined_msra.json'" \
    > "logs/overnight2_${STAMP}_p4_combined_msr.log" 2>&1 &
P4_MSR=$!

wait "$P4_FIN"; say "  combined on fiNERweb test (rc=$?)"
wait "$P4_CLU"; say "  combined on cluener test (rc=$?)"
wait "$P4_MSR"; say "  combined on msra test (rc=$?)"

# novel extraction with combined adapter, one novel per GPU
CUDA_VISIBLE_DEVICES=0 HF_HUB_DISABLE_PROGRESS_BARS=1 \
    nohup nice -n 10 ionice -c 2 -n 7 timeout 3h \
    bash -lc "conda activate ner_ft && python extract_novels.py --max_chunks 500 --adapter '${P3_ADAPTER}' --novel '${N1}' --prefix 1 --out 'predictions/${STAMP}_combined_1.jsonl'" \
    > "logs/overnight2_${STAMP}_p4_combined_novel1.log" 2>&1 &
P4_N1=$!
sleep 30
CUDA_VISIBLE_DEVICES=1 HF_HUB_DISABLE_PROGRESS_BARS=1 \
    nohup nice -n 10 ionice -c 2 -n 7 timeout 3h \
    bash -lc "conda activate ner_ft && python extract_novels.py --max_chunks 500 --adapter '${P3_ADAPTER}' --novel '${N2}' --prefix 2 --out 'predictions/${STAMP}_combined_2.jsonl'" \
    > "logs/overnight2_${STAMP}_p4_combined_novel2.log" 2>&1 &
P4_N2=$!

wait "$P4_N1"; say "  combined on 1.txt (rc=$?)"
wait "$P4_N2"; say "  combined on 2.txt (rc=$?)"

# ============ Phase 5: presentation summary ============
say ""
say "## Phase 5: presentation summary"
bash -lc "conda activate ner_ft && python make_presentation.py --stamp '${STAMP}' --out 'PRESENTATION_${STAMP}.md'" >> "$REPORT" 2>&1
say "  summary written to PRESENTATION_${STAMP}.md"

say ""
say "## DONE at $(date +%F\ %T)"
for f in predictions/${STAMP}_*.jsonl; do
    [ -f "$f" ] && say "   $(wc -l < "$f") records  $f"
done
