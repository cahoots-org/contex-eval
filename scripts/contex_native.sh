#!/bin/sh
# Start Contex natively with scripts/contex-native.env (see that file). Run from the repo root.
# One-time: python3.12 -m venv contex/.venv-native && contex/.venv-native/bin/pip install -r contex/requirements.txt sentence-transformers torch
#           git -C contex apply ../scripts/contex-mps.patch
set -a; . ./scripts/contex-native.env; set +a
cd contex && exec .venv-native/bin/python main.py
