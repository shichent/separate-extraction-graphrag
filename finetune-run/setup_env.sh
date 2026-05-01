#!/usr/bin/env bash
# One-time conda env creation for the NER transfer experiment.
# Run INTERACTIVELY (not under launcher). Takes ~20-40 min on first run.
# Idempotent: re-running after a partial install is safe.

set -eo pipefail   # NOT -u: conda's activate.d scripts reference unbound vars

ENV_NAME="${ENV_NAME:-ner_ft}"
PY_VER="${PY_VER:-3.11}"

echo "[setup] creating conda env '${ENV_NAME}' (python ${PY_VER}) ..."

# conda activate only works under a login shell; also avoid -u inside
bash -lc "
set -eo pipefail
if conda env list | awk '{print \$1}' | grep -qx '${ENV_NAME}'; then
    echo '[setup] env already exists, reusing'
else
    conda create -n '${ENV_NAME}' python=${PY_VER} -y
fi
conda activate '${ENV_NAME}'

python -m pip install --upgrade pip wheel setuptools

# Match CUDA 12.1. Using torch 2.5.1: unsloth_zoo needs torch._inductor.config
# which is absent in 2.4.1, and torch 2.6+ pulls torchao>=0.13 (needs torch.int1).
pip install --index-url https://download.pytorch.org/whl/cu121 \
    'torch==2.5.1' 'torchvision==0.20.1'

# Unsloth stack — pin torchao to last version compatible with torch 2.5
pip install 'unsloth @ git+https://github.com/unslothai/unsloth.git'
pip install unsloth_zoo
pip install 'torchao==0.6.1'
pip install 'xformers==0.0.28.post1' trl peft accelerate bitsandbytes
pip install json_repair pyarrow tqdm

python - <<'PY'
import torch, transformers, peft, trl, unsloth
print('torch        :', torch.__version__, 'cuda:', torch.cuda.is_available(), 'gpus:', torch.cuda.device_count())
print('transformers :', transformers.__version__)
print('peft         :', peft.__version__)
print('trl          :', trl.__version__)
print('unsloth      :', unsloth.__version__)
PY
"

echo "[setup] done. Activate with:   conda activate ${ENV_NAME}"
