#!/usr/bin/env bash
# Separate env for NeMo (it pins torch/numpy differently from the main env).
#   bash scripts/setup_nemo_env.sh          # creates .venv-nemo next to .venv
set -euo pipefail
cd "$(dirname "$0")/.."
export PATH="$HOME/.local/bin:$PATH"
uv venv -p 3.12 .venv-nemo
uv pip install -p .venv-nemo/bin/python --torch-backend auto "torch>=2.4" "torchaudio"
uv pip install -p .venv-nemo/bin/python "nemo_toolkit[asr]" "cython" "packaging"
uv pip install -p .venv-nemo/bin/python -e . --no-deps
uv pip install -p .venv-nemo/bin/python jiwer rapidfuzz numpy scipy soundfile pyyaml tenacity httpx tqdm python-dotenv huggingface_hub librosa
.venv-nemo/bin/python -c "import nemo, torch; print('nemo', nemo.__version__, 'torch', torch.__version__, torch.cuda.is_available())"
