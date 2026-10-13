"""Concurrency check for scripts/contex-mps.patch: Contex encodes from worker threads, and MPS
aborts the process on concurrent use (exit 134) unless encodes are serialized.

Run: contex/.venv-native/bin/python scripts/check_mps_embedder.py
"""
import sys
import threading

sys.path.insert(0, "contex")
from src.core.embedder import TorchEmbedder

texts = ["Volcano erupts. " * 40] * 64
m = TorchEmbedder("thenlper/gte-base", "mps")


def work(n):
    for _ in range(n):
        m.encode(texts)
        m.encode("news about volcanoes")


threads = [threading.Thread(target=work, args=(15,)) for _ in range(6)]
for t in threads:
    t.start()
for t in threads:
    t.join()
print("STRESS OK")
