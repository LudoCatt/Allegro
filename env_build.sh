#!/usr/bin/env bash

###############################################################################
# You may tweak these names/versions if you like
###############################################################################
RAW_REQ="allegro_requirements_raw.txt"
REQ="allegro_requirements.txt"
PYTHON_VERSION="3.10"        # pick 3.11 if you prefer
PYTORCH_VERSION="2.4.0"      # or 2.4.* / nightly
CUDA_TAG="cu124"             # keep in sync with your driver (≥ 550.x)
###############################################################################

echo "—— 0. Capturing package list from current env ————————————————————————"
if conda info --envs | grep -q '^allegro'; then
  # Dump every pip-installed package in the env
  conda run -n allegro --no-capture-output pip freeze > "$RAW_REQ"
  # Strip any lines we’ll overwrite later (torch, torchvision, torchaudio, deepspeed)
  grep -Ev '^(torch($|==)|torchvision|torchaudio|deepspeed)' "$RAW_REQ" > "$REQ"
  echo "   ✔  Saved package list to $REQ"
else
  echo "   (!) Environment 'allegro' not found — nothing to back up."
fi

echo "—— 1. Removing old env ———————————————————————————————————————————"
if conda info --envs | grep -q '^allegro'; then
  conda deactivate || true
  conda env remove -n allegro -y
fi
rm -rf ~/.virtualenvs/allegro 2>/dev/null || true

echo "—— 2. Creating fresh env with Python ≥ $PYTHON_VERSION —————————————"
conda create -n allegro python="$PYTHON_VERSION" -y
conda activate allegro

echo "—— 3. Re-installing previous packages ————————————————————————————"
pip install --upgrade pip
if [[ -f "$REQ" ]]; then
  pip install -r "$REQ"
else
  echo "   (!) No $REQ found — skipping refill step."
fi

echo "—— 4. Installing PyTorch $PYTORCH_VERSION ($CUDA_TAG) + DeepSpeed ——————"
pip install --pre torch=="$PYTORCH_VERSION" torchvision torchaudio \
  --index-url "https://download.pytorch.org/whl/nightly/$CUDA_TAG"
# Build DeepSpeed against the active CUDA toolchain
DS_BUILD_OPS=1 pip install deepspeed --pre

echo "—— 5. Sanity check ———————————————————————————————————————————————"
python - <<'PY'
import sys, torch, deepspeed
print(f"Python    : {sys.version.split()[0]}")
print(f"PyTorch   : {torch.__version__} (CUDA {torch.version.cuda})")
print(f"DeepSpeed : {deepspeed.__version__}")
PY

echo "✅  All done — environment 'allegro' rebuilt, refilled, and armed with DeepSpeed."
