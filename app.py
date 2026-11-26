"""Gradio app for the trained CI/CD log Character-CNN + Transformer classifier.

Place these two training artifacts beside this file:
  - gha_log_transformer_model.pt
  - char_vocab.json

The preprocessing helpers below are copied from the training notebook so that
inference follows the same cleaning pipeline used before model training.
"""
from pathlib import Path
from collections import Counter
import os
import json
import re

# Keep CPU thread usage modest on Render's small Free instances.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gradio as gr

try:
    torch.set_num_threads(max(1, int(os.environ.get("TORCH_NUM_THREADS", "1"))))
    torch.set_num_interop_threads(1)
except RuntimeError:
    # PyTorch may already have initialized its thread pool in some environments.
    pass

# Render Free Web Services use CPU inference.

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "gha_log_cnn_bilstm_model.pt"
VOCAB_PATH = BASE_DIR / "char_vocab.json"
PAD_ID, UNK_ID = 0, 1

if not MODEL_PATH.is_file():
    raise FileNotFoundError(
        f"Missing {MODEL_PATH.name}. Upload the trained checkpoint from your notebook output folder."
    )
if not VOCAB_PATH.is_file():
    raise FileNotFoundError(
        f"Missing {VOCAB_PATH.name}. Upload the character vocabulary saved by the training notebook."
    )

# weights_only=True is supported by the PyTorch version used by ZeroGPU.
try:
    CHECKPOINT = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
except TypeError:  # compatibility with older local PyTorch versions
    CHECKPOINT = torch.load(MODEL_PATH, map_location="cpu")

with VOCAB_PATH.open("r", encoding="utf-8") as f:
    char_to_id = json.load(f)

CLASS_NAMES = CHECKPOINT.get(
    "class_names",
    ["Normal", "Flaky test", "Configuration drift", "Security anomaly", "Silent failure"],
)
NUM_CLASSES = int(CHECKPOINT.get("num_classes", len(CLASS_NAMES)))
MAX_CHARS = int(CHECKPOINT.get("max_chars", 256))
WINDOW_SIZE = int(CHECKPOINT.get("max_window", 30))
WINDOW_STEP = int(CHECKPOINT.get("window_step", 15))
REMOVE_LEADING_TIMESTAMPS = bool(CHECKPOINT.get("remove_leading_timestamps", True))
INFERENCE_BATCH_WINDOWS = 1
MAX_INPUT_CHARACTERS = 500_000
MAX_RETAINED_LINES = 4_000

# The model is loaded at module scope. This is required by ZeroGPU's PyTorch
# CUDA emulation; on ordinary CPU hardware it falls back to CPU automatically.
DEVICE = torch.device("cpu")

TAG_TO_ID = {
    'FLAKY': 1,
    'DRIFT': 2,
    'SECURITY': 3,
    'SILENT': 4
}

TAG_RE = re.compile(
    r'<\s*(FLAKY|DRIFT|SECURITY|SILENT)\s*>',
    re.IGNORECASE
)


# ---------------------------------------------------------
# 2. ANSI escape and terminal control characters
# ---------------------------------------------------------

ANSI_RE = re.compile(
    r'\x1b(?:\[[0-?]*[ -/]*[@-~]|\][^\x07]*(?:\x07|\x1b\\)|[@-_])'
)
