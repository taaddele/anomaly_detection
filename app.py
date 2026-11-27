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
import urllib.request
import urllib.error
from datetime import datetime, timezone

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
MODEL_PATH = BASE_DIR / "gha_log_transformer_model.pt"
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
print(f"[STARTUP] Loading checkpoint: {MODEL_PATH.name}", flush=True)
try:
    CHECKPOINT = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
except TypeError:  # compatibility with older local PyTorch versions
    CHECKPOINT = torch.load(MODEL_PATH, map_location="cpu")

print("[STARTUP] Checkpoint loaded", flush=True)

with VOCAB_PATH.open("r", encoding="utf-8") as f:
    char_to_id = json.load(f)
print(f"[STARTUP] Vocabulary loaded ({len(char_to_id)} entries)", flush=True)

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

# # Load the trained model for CPU inference on Render.
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

ESC_RE = re.compile(r'\x1b')

CONTROL_RE = re.compile(
    r'[\x00-\x08\x0b-\x1f\x7f]'
)


# ---------------------------------------------------------
# 3. Timestamp removal
# ---------------------------------------------------------

TIMESTAMP_RE = re.compile(
    r'^\s*(?:'
    r'\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}'
    r'(?:[.,]\d+)?(?:Z|[+-]\d{2}:?\d{2})?'
    r'|\[\d{2}:\d{2}:\d{2}\]'
    r')\s*'
)


# ---------------------------------------------------------
# 4. Detect standalone unwanted terminal artifacts
# ---------------------------------------------------------

NOISE_PATTERNS = [

    # Lines containing only braces or square brackets
    re.compile(r'^[{}\[\]]+$'),

    # Repeated separators and empty ASCII table borders
    re.compile(r'^[+\-|:=]+$'),

    # Unicode box-drawing characters and table borders
    re.compile(
        r'^[┌┐└┘├┤┬┴┼╔╗╚╝╠╣╦╩╬─━═│┃]+$'
    ),

    # Terminal progress indicators and progress bars
    re.compile(r'^[━─—═█▉▊▋▌▍▎▏▓▒░#=]+$'),

    # Repeated dots
    re.compile(r'^[.]{2,}$'),

    # Repeated terminal time/progress placeholders
    re.compile(r'^(?:--:--:--\s*){2,}$'),

    # Standalone GitHub Actions group artifacts
    re.compile(r'^(?:endgroup|group\])$', re.IGNORECASE),

    # Standalone decorative characters
    re.compile(r'^[\s_~*.,/\\]+$'),

]


def is_noise_line(line):
    """
    Identifies lines consisting entirely of unwanted
    formatting, terminal artifacts, or structural characters.

    Text-containing lines are not removed by this function.
    """

    cleaned = line.strip()

    if not cleaned:
        return True

    for pattern in NOISE_PATTERNS:
        if pattern.fullmatch(cleaned):
            return True

    return False


# ---------------------------------------------------------
# 5. Remove unnecessary formatting at text boundaries
# ---------------------------------------------------------

# Repeated decorative characters at the beginning.
# Preserve actual command-line options such as --help.

EDGE_PREFIX_RE = re.compile(
    r'^[ \t]*(?:'
    r'[=+|:;.,_~*#━─—═]{2,}'
    r'|-{2,}[ \t]+'
    r'|\|[ \t]+'
    r')'
)

# Repeated decorative characters at the end.
# Dots are included to remove .... after text.

EDGE_SUFFIX_RE = re.compile(
    r'(?:'
    r'[=+|:;.,_~*#━─—═]{2,}'
    r'|\.{2,}'
    r'|[ \t]+-{2,}'
    r'|[ \t]+\|'
    r')[ \t]*$'
)


def clean_line_edges(line):
    """
    Removes unnecessary repeated formatting characters
    from the beginning and end of a line.

    Examples:
        ======Testing PUT Methods======
        -> Testing PUT Methods

        ...... ERROR: Connection failed ......
        -> ERROR: Connection failed

        ------ Build Started ------
        -> Build Started

        --help
        -> --help
    """

    line = line.strip()

    while True:

        previous_line = line

        # Remove decorative prefixes
        line = EDGE_PREFIX_RE.sub('', line, count=1).strip()

        # Remove decorative suffixes
        line = EDGE_SUFFIX_RE.sub('', line, count=1).strip()

        # Stop when no further changes occur
        if line == previous_line:
            break

    return line.strip()


# ---------------------------------------------------------
# 6. Normalize unwanted information inside valid log lines
# ---------------------------------------------------------

def normalize_dynamic_log_fields(line):
    """
    Normalizes dynamic, dataset-specific, and unnecessary
    information inside otherwise valid log lines.

    Preserves meaningful CI/CD and anomaly indicators.

    Normalizes:
        GitHub Actions group annotations
        Progress-bar artifacts
        API endpoints and URLs
        Absolute and relative file paths
        UUIDs and long hexadecimal identifiers
        IP addresses
        Python test method identifiers
        Assertion method identifiers
        Dynamic ID fields
        Secret values
        Execution durations

    Important:
        Terms such as error, warn, failed, secret, debug,
        dependency, package, changed, signing_alg and hs256
        are preserved.
    """

    # -----------------------------------------------------
    # 6.1 Remove GitHub Actions group annotations
    # -----------------------------------------------------

    # Examples:
    # ##[group]Initializing
    # ##[endgroup]

    line = re.sub(
        r'(?i)##\[\s*(?:group|endgroup)\s*\]',
        ' ',
        line
    )

    line = re.sub(
        r'(?i)::(?:group|endgroup)::',
        ' ',
        line
    )

    # Remove incomplete group annotation remnants
    line = re.sub(
        r'(?i)\bgroup\]',
        ' ',
        line
    )

    # Remove remaining endgroup words
    line = re.sub(
        r'(?i)\bendgroup\b',
        ' ',
        line
    )

    # -----------------------------------------------------
    # 6.2 Remove inline terminal progress bars
    # -----------------------------------------------------

    # Example:
    # Downloading 100%|██████████|

    line = re.sub(
        r'\s*\d{1,3}%\s*\|'
        r'[█▉▊▋▌▍▎▏▓▒░#=.\- ]{2,}\|',
        ' ',
        line
    )

    # -----------------------------------------------------
    # 6.3 Normalize API endpoint paths
    # -----------------------------------------------------

    # Example:
    # /api/subjects/123/students/456/exams
    # becomes API_ENDPOINT_TOKEN

    line = re.sub(
        r'(?i)(?<![\w.])/api/[^\s\'"<>),;]+',
        'API_ENDPOINT_TOKEN',
        line
    )

    # Preserve endpoint field names but normalize their values.
    # Example:
    # endpoint='/subjects/123/exams'
    # becomes endpoint=URL_PATH_TOKEN

    line = re.sub(
        r"""(?i)\b(url|uri|endpoint)\s*([=:])\s*
        (['"]?)/(?!/)[^\s'"]*""",
        r'\1\2URL_PATH_TOKEN',
        line,
        flags=re.VERBOSE
    )

    # -----------------------------------------------------
    # 6.4 Normalize web URLs
    # -----------------------------------------------------

    # Examples:
    # https://example.com/api/test
    # http://153.105.55.110
    # www.example.com

    line = re.sub(
        r'(?i)https?://[^\s\'"<>]+',
        'URL_TOKEN',
        line
    )

    line = re.sub(
        r'(?i)\bwww\.[^\s\'"<>]+',
        'URL_TOKEN',
        line
    )

    # -----------------------------------------------------
    # 6.5 Normalize absolute Linux and runner paths
    # -----------------------------------------------------

    # Examples:
    # /home/runner/work/dummy/dummy/test.py
    # /usr/bin/git
    # /tmp/build/output
    # /workspace/project/file.txt

    line = re.sub(
        r"(?<![\w.])/(?:home|workspace|tmp|root|opt|usr|var|"
        r"etc|build|runner|github)(?:/[^\s,;'\"()]*)*",
        'PATH_TOKEN',
        line,
        flags=re.IGNORECASE
    )

    # -----------------------------------------------------
    # 6.6 Normalize Windows paths
    # -----------------------------------------------------

    line = re.sub(
        r'\b[A-Za-z]:\\(?:[^\\\s]+\\)*[^\\\s]+',
        'PATH_TOKEN',
        line
    )

    # -----------------------------------------------------
    # 6.7 Normalize relative file paths
    # -----------------------------------------------------

    # Examples:
    # pom/test_values.json
    # src/tests/test_api.py
    # config/dependencies.yaml

    RELATIVE_FILE_PATH_RE = re.compile(
        r"""(?i)(?<![\w./])
        (?:[\w.-]+/)+[\w.-]+\.
        (?:json|yaml|yml|toml|xml|txt|py|js|ts|lock|sh)
        \b""",
        re.VERBOSE
    )

    line = RELATIVE_FILE_PATH_RE.sub(
        'FILE_PATH_TOKEN',
        line
    )

    # -----------------------------------------------------
    # 6.8 Normalize UUID values
    # -----------------------------------------------------

    line = re.sub(
        r'\b[0-9a-fA-F]{8}-(?:[0-9a-fA-F]{4}-){3}'
        r'[0-9a-fA-F]{12}\b',
        'UUID_TOKEN',
        line
    )

    # -----------------------------------------------------
    # 6.9 Normalize long hexadecimal identifiers and hashes
    # -----------------------------------------------------

    # Examples:
    # 30f793927d2ae2a6335907cf93ab4ded
    # 22ac2ceda7cbae0309f3d5c014ac8bca

    line = re.sub(
        r'\b[0-9a-fA-F]{16,}\b',
        'HASH_TOKEN',
        line
    )

    # -----------------------------------------------------
    # 6.10 Normalize Python test-method identifiers
    # -----------------------------------------------------

    # Example:
    # __main__.testpostmethods.test_created
    # becomes TEST_CASE_TOKEN

    line = re.sub(
        r'\b__main__\.[A-Za-z_]\w*'
        r'(?:\.[A-Za-z_]\w*)+\b',
        'TEST_CASE_TOKEN',
        line
    )

    # Normalize individual test method names.
    # Example:
    # test_created -> TEST_CASE_TOKEN

    line = re.sub(
        r'\btest_[A-Za-z0-9_]+\b',
        'TEST_CASE_TOKEN',
        line
    )

    # -----------------------------------------------------
    # 6.11 Normalize Python assertion method identifiers
    # -----------------------------------------------------

    # Example:
    # self.assertEqual -> ASSERTION_TOKEN

    line = re.sub(
        r'\bself\.assert[A-Za-z_]+\b',
        'ASSERTION_TOKEN',
        line,
        flags=re.IGNORECASE
    )

    # -----------------------------------------------------
    # 6.12 Normalize dynamic ID field names
    # -----------------------------------------------------

    # Examples:
    # id_subject
    # id_person
    # client_id

    line = re.sub(
        r'\b(?:id_[A-Za-z0-9_]+|client_id)\b',
        'ID_FIELD_TOKEN',
        line,
        flags=re.IGNORECASE
    )

    # -----------------------------------------------------
    # 6.13 Normalize variable placeholders
    # -----------------------------------------------------

    # Example:
    # {idsubject}, {studentid}
    # becomes {PARAM}

    line = re.sub(
        r'\{[A-Za-z_][A-Za-z0-9_-]*\}',
        '{PARAM}',
        line
    )

    # -----------------------------------------------------
    # 6.14 Normalize IP addresses
    # -----------------------------------------------------

    # Example:
    # 153.105.55.110 -> IP_ADDRESS_TOKEN

    line = re.sub(
        r'\b(?:\d{1,3}\.){3}\d{1,3}\b',
        'IP_ADDRESS_TOKEN',
        line
    )

    # -----------------------------------------------------
    # 6.15 Mask secret values
    # -----------------------------------------------------

    # Preserve security field names while masking their values.
    #
    # Examples:
    # client_secret=abc123
    # becomes:
    # client_secret=SECRET_VALUE_TOKEN
    #
    # password=xyz789
    # becomes:
    # password=SECRET_VALUE_TOKEN
    #
    # Also supports quoted JSON-style field names.

    SECRET_ASSIGNMENT_RE = re.compile(
        r"""(?ix)
        (?P<quote>["']?)
        (?P<key>
            (?:[A-Za-z_][A-Za-z0-9_-]*[_-])?
            (?:password|passwd|token|secret|api[_-]?key|access[_-]?key)
        )
        (?P=quote)
        \s*
        (?P<separator>[:=])
        \s*
        (?P<value>
            "[^"]*"
            |
            '[^']*'
            |
            [^\s,;}\]]+
        )
        """
    )

    def mask_secret_value(match):

        field_name = match.group('key')
        separator = match.group('separator')

        return (
            field_name
            + separator
            + 'SECRET_VALUE_TOKEN'
        )

    line = SECRET_ASSIGNMENT_RE.sub(
        mask_secret_value,
        line
    )

    # Normalize authorization header values.
    line = re.sub(
        r'(?i)\b(authorization)\s*([=:])\s*'
        r'(?:Bearer|Basic)\s+\S+',
        r'\1\2AUTH_VALUE_TOKEN',
        line
    )

    # -----------------------------------------------------
    # 6.16 Normalize execution durations
    # -----------------------------------------------------

    # Examples:
    # 0.002s -> DURATION_TOKEN
    # 0.000s -> DURATION_TOKEN
    # 200ms  -> DURATION_TOKEN

    line = re.sub(
        r'\b\d+(?:[.,]\d+)?\s*'
        r'(?:ms|msec|s|sec|seconds?)\b',
        'DURATION_TOKEN',
        line,
        flags=re.IGNORECASE
    )

    # -----------------------------------------------------
    # 6.17 Remove repeated dots inside log lines
    # -----------------------------------------------------

    # Do this after path and URL normalization so that
    # ordinary file extensions and URL structures are handled.

    line = re.sub(
        r'\.{2,}',
        ' ',
        line
    )

    # -----------------------------------------------------
    # 6.18 Normalize whitespace
    # -----------------------------------------------------

    line = re.sub(r'[ \t]+', ' ', line).strip()

    return line

# -----------------------------------------------------------------------------
# Exact model architecture used in the training notebook
# -----------------------------------------------------------------------------
class CharacterCNN(nn.Module):
    def __init__(self, vocab_size, char_dim=32, d_model=128, pad_id=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, char_dim, padding_idx=pad_id)
        self.conv1 = nn.Conv1d(char_dim, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=3, padding=1)
        self.projection = nn.Linear(128, d_model)
        self.dropout = nn.Dropout(0.1)

    def forward(self, char_ids):
        valid = char_ids.ne(PAD_ID)
        z = self.embedding(char_ids).transpose(1, 2)
        z = F.gelu(self.conv1(z))
        z = F.gelu(self.conv2(z)).transpose(1, 2)
        valid_f = valid.unsqueeze(-1).to(z.dtype)
        mean_pool = (z * valid_f).sum(dim=1) / valid_f.sum(dim=1).clamp_min(1.0)
        max_pool = z.masked_fill(~valid.unsqueeze(-1), -1e4).max(dim=1).values
        has_chars = valid.any(dim=1, keepdim=True)
        max_pool = torch.where(has_chars, max_pool, torch.zeros_like(max_pool))
        pooled = torch.cat([mean_pool, max_pool], dim=-1)
        return self.dropout(self.projection(pooled))


class LogTransformerClassifier(nn.Module):
    def __init__(self, vocab_size, char_dim=32, d_model=128, n_heads=2,
                 n_layers=2, ff_dim=1024, max_window=30, num_classes=5):
        super().__init__()
        self.line_encoder = CharacterCNN(vocab_size, char_dim, d_model)
        self.position_embedding = nn.Embedding(max_window, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=ff_dim,
            dropout=0.1, batch_first=True, activation="gelu"
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.classifier = nn.Linear(d_model, num_classes)
        self.max_window = max_window

    def forward(self, char_ids, padding_mask):
        batch_size, window_size, char_length = char_ids.shape
        z = self.line_encoder(
            char_ids.reshape(batch_size * window_size, char_length)
        ).reshape(batch_size, window_size, -1)
        positions = torch.arange(window_size, device=char_ids.device).unsqueeze(0)
        z = z + self.position_embedding(positions)
        z = self.transformer(z, src_key_padding_mask=padding_mask)
        return self.classifier(z)


# The checkpoint saved by the CNN-Transformer notebook contains the model
# architecture configuration below. Defaults are retained for compatibility.
MODEL_CONFIG = {
    "char_dim": int(CHECKPOINT.get("char_dim", 32)),
    "d_model": int(CHECKPOINT.get("d_model", 128)),
    "n_heads": int(CHECKPOINT.get("n_heads", 2)),
    "n_layers": int(CHECKPOINT.get("n_layers", 2)),
    "ff_dim": int(CHECKPOINT.get("ff_dim", 1024)),
    "max_window": int(CHECKPOINT.get("max_window", 30)),
    "num_classes": int(CHECKPOINT.get("num_classes", len(CLASS_NAMES))),
}

state_dict = CHECKPOINT.get("model_state_dict")
if state_dict is None:
    raise KeyError(
        "The checkpoint does not contain 'model_state_dict'. Re-save the trained "
        "model using the notebook's model-saving cell and upload the resulting "
        "gha_log_transformer_model.pt file."
    )

model = LogTransformerClassifier(
    vocab_size=len(char_to_id),
    char_dim=MODEL_CONFIG["char_dim"],
    d_model=MODEL_CONFIG["d_model"],
    n_heads=MODEL_CONFIG["n_heads"],
    n_layers=MODEL_CONFIG["n_layers"],
    ff_dim=MODEL_CONFIG["ff_dim"],
    max_window=MODEL_CONFIG["max_window"],
    num_classes=MODEL_CONFIG["num_classes"],
)
print("[STARTUP] Initializing CNN-Transformer model", flush=True)
model.load_state_dict(state_dict, strict=True)
model.to(DEVICE)
model.eval()
print("[STARTUP] Model weights loaded successfully", flush=True)


def prepare_log_lines(raw_text):
    """Clean each line using the same operations/order used during training.

    Ground-truth annotation tags are removed from model input. Lines removed as
    blank/noise during training are also removed here. Original line numbers
    are retained for display.
    """
    prepared = []
    for original_line_number, raw_line in enumerate((raw_text or "").splitlines(), start=1):
        if not raw_line.strip():
            continue

        line = ANSI_RE.sub("", raw_line)
        line = ESC_RE.sub("", line)
        line = CONTROL_RE.sub(" ", line)
        line = line.replace("\x00", " ")
        line = TAG_RE.sub(" ", line)

        if REMOVE_LEADING_TIMESTAMPS:
            line = TIMESTAMP_RE.sub("", line)

        line = re.sub(r"[ \t]+", " ", line).strip()
        line = clean_line_edges(line)
        line = normalize_dynamic_log_fields(line)
        line = clean_line_edges(line)
        line = re.sub(r"[ \t]+", " ", line).strip()

        if not line or is_noise_line(line):
            continue

        prepared.append({
            "line_number": original_line_number,
            "original": raw_line,
            "clean": line[:2000],
        })

        if len(prepared) > MAX_RETAINED_LINES:
            raise ValueError(
                f"This demo accepts up to {MAX_RETAINED_LINES:,} retained log lines per request. "
                "Please split the log into smaller sections."
            )
    return prepared


def _softmax(values):
    shifted = values - np.max(values)
    exp_values = np.exp(shifted)
    return exp_values / exp_values.sum()


def predict_log_text(raw_text):
    if not (raw_text or "").strip():
        return [], "Paste log text or upload a .txt log file."
    if len(raw_text) > MAX_INPUT_CHARACTERS:
        return [], f"Input is too large. Limit: {MAX_INPUT_CHARACTERS:,} characters."

    items = prepare_log_lines(raw_text)
    if not items:
        return [], "No usable log lines remained after the training-time cleaning steps."

    n_lines = len(items)
    if n_lines <= WINDOW_SIZE:
        starts = [0]
    else:
        starts = list(range(0, n_lines - WINDOW_SIZE + 1, WINDOW_STEP))
        last_start = n_lines - WINDOW_SIZE
        if not starts or starts[-1] != last_start:
            starts.append(last_start)

    # Average logits for a line over every overlapping window containing it,
    # matching the notebook's predict_unique_lines evaluation procedure.
    score_sums = [np.zeros(NUM_CLASSES, dtype=np.float64) for _ in range(n_lines)]
    score_counts = np.zeros(n_lines, dtype=np.int64)
    model.eval()

    with torch.inference_mode():
        for batch_start in range(0, len(starts), INFERENCE_BATCH_WINDOWS):
            batch_starts = starts[batch_start:batch_start + INFERENCE_BATCH_WINDOWS]
            batch_x = np.zeros(
                (len(batch_starts), WINDOW_SIZE, MAX_CHARS), dtype=np.int64
            )
            batch_padding = np.ones(
                (len(batch_starts), WINDOW_SIZE), dtype=bool
            )
            batch_line_indices = np.full(
                (len(batch_starts), WINDOW_SIZE), -1, dtype=np.int64
            )

            for batch_i, window_start in enumerate(batch_starts):
                for position in range(WINDOW_SIZE):
                    line_i = window_start + position
                    if line_i >= n_lines:
                        break
                    char_ids = [
                        char_to_id.get(ch, UNK_ID)
                        for ch in items[line_i]["clean"][:MAX_CHARS]
                    ]
                    if char_ids:
                        batch_x[batch_i, position, :len(char_ids)] = char_ids
                    batch_padding[batch_i, position] = False
                    batch_line_indices[batch_i, position] = line_i

            x_tensor = torch.from_numpy(batch_x).to(DEVICE)
            padding_tensor = torch.from_numpy(batch_padding).to(DEVICE)
            batch_logits = model(x_tensor, padding_tensor).detach().cpu().numpy()

            for batch_i in range(len(batch_starts)):
                for position in range(WINDOW_SIZE):
                    line_i = int(batch_line_indices[batch_i, position])
                    if line_i < 0:
                        continue
                    score_sums[line_i] += batch_logits[batch_i, position]
                    score_counts[line_i] += 1

    rows = []
    predicted_counts = Counter()
    anomaly_lines = 0
    top_scores = []

    for i, item in enumerate(items):
        mean_logits = score_sums[i] / max(int(score_counts[i]), 1)
        probabilities = _softmax(mean_logits)
        pred_id = int(np.argmax(mean_logits))
        confidence_like_score = float(probabilities[pred_id])
        predicted_class = CLASS_NAMES[pred_id]
        predicted_counts[predicted_class] += 1
        anomaly = pred_id != 0
        anomaly_lines += int(anomaly)
        top_scores.append(confidence_like_score)
        rows.append([
            item["line_number"],
            predicted_class,
            round(confidence_like_score * 100, 2),
            "ANOMALY" if anomaly else "NORMAL",
            item["clean"][:300],
        ])

    count_text = "\n".join(
        f"- **{name}:** {predicted_counts.get(name, 0)} line(s)"
        for name in CLASS_NAMES
    )
    summary = (
        "### Analysis summary\n"
        f"- Non-empty log lines classified: **{n_lines:,}**\n"
        f"- Lines predicted anomalous: **{anomaly_lines:,}**\n"
        f"- Lines predicted normal: **{n_lines - anomaly_lines:,}**\n"
        f"- Mean top-class score: **{100 * float(np.mean(top_scores)):.2f}%**\n\n"
        "### Predicted class counts\n"
        f"{count_text}\n\n"
        "> Scores are softmax outputs, not calibrated probabilities. Predictions are model outputs, "
        "not proof that an incident exists. Each line is classified using its surrounding "
        "log-window context."
    )
    return rows, summary


def analyze_uploaded_or_pasted_log(pasted_text, uploaded_file):
    """Analyze an uploaded or pasted CI/CD log and build dashboard summaries."""
    try:
        if uploaded_file is not None:
            file_path = uploaded_file if isinstance(uploaded_file, str) else uploaded_file.name
            raw_text = Path(file_path).read_text(encoding="utf-8", errors="replace")
        else:
            raw_text = pasted_text or ""
        rows, summary = predict_log_text(raw_text)
        if not rows:
            return rows, summary, '<div class="empty-kpi">Run an analysis to see prediction statistics.</div>', []
        total = len(rows)
        normal = sum(1 for row in rows if str(row[3]).upper() == "NORMAL")
        anomalous = total - normal
        mean_score = float(np.mean([float(row[2]) for row in rows]))
        stats_html = f"""<div class="kpi-grid">
          <div class="kpi"><span>LINES ANALYZED</span><strong>{total:,}</strong><small>Retained log lines</small></div>
          <div class="kpi"><span>NORMAL</span><strong>{normal:,}</strong><small>{normal / total * 100:.1f}% of lines</small></div>
          <div class="kpi"><span>ANOMALIES</span><strong>{anomalous:,}</strong><small>{anomalous / total * 100:.1f}% of lines</small></div>
          <div class="kpi"><span>MEAN TOP SCORE</span><strong>{mean_score:.1f}%</strong><small>Not calibrated confidence</small></div>
        </div>"""
        counts = Counter(str(row[1]) for row in rows)
        class_rows = []
        for class_name in CLASS_NAMES:
            count = counts.get(class_name, 0)
            class_rows.append([class_name, count, round(100 * count / total, 2),
                               "Normal" if class_name == CLASS_NAMES[0] else "Anomaly"])
        return rows, summary, stats_html, class_rows
    except Exception as exc:
        return [], f"**Unable to analyze this input:** {type(exc).__name__}: {exc}", '<div class="empty-kpi">Analysis did not complete.</div>', []


def submit_feedback(name, email, category, message):
    """Save feedback locally and send an email notification through Web3Forms."""
    message = (message or "").strip()
    if not message:
        return "Please enter a comment before submitting."

    feedback = {
        "submitted_at_utc": datetime.now(timezone.utc).isoformat(),
        "name": (name or "Anonymous").strip() or "Anonymous",
        "email": (email or "").strip(),
        "category": category or "General comment",
        "message": message[:10000],
    }

    # Keep a local copy where the service filesystem permits it. Render Free
    # filesystems are ephemeral, so Web3Forms email is the durable delivery path.
    saved_locally = False
    local_save_note = ""
    feedback_path = BASE_DIR / "feedback.jsonl"
    try:
        with feedback_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(feedback, ensure_ascii=False) + "\n")
        saved_locally = True
    except OSError as exc:
        local_save_note = f" Local file backup was unavailable ({type(exc).__name__})."

    access_key = (os.environ.get("WEB3FORMS_ACCESS_KEY") or "").strip()
    if not access_key:
        if saved_locally:
            return ("Your comment was saved temporarily, but email delivery is not configured. "
                    "Add the Render environment variable WEB3FORMS_ACCESS_KEY and redeploy. "
                    "Render Free local files may be cleared when the service restarts.")
        return ("Feedback could not be sent because WEB3FORMS_ACCESS_KEY is not configured. "
                "Add it in Render Environment settings and redeploy." + local_save_note)

    # Web3Forms accepts JSON submissions at this HTTPS endpoint. The access key
    # is associated with the recipient email configured when the key was created.
    payload = {
        "access_key": access_key,
        "from_name": "CI/CD Log Anomaly Research Dashboard",
        "subject": f"CI/CD Classifier feedback: {feedback['category']}",
        "name": feedback["name"],
        "category": feedback["category"],
        "submitted_at_utc": feedback["submitted_at_utc"],
        "message": feedback["message"],
    }
    if feedback["email"]:
        # Web3Forms uses the submitted email as the notification's Reply-To.
        payload["email"] = feedback["email"]

    request = urllib.request.Request(
        "https://api.web3forms.com/submit",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            response_text = response.read().decode("utf-8", errors="replace")
            result = json.loads(response_text) if response_text else {}

        nested_body = result.get("body") if isinstance(result, dict) else None
        api_success = (
            isinstance(result, dict) and result.get("success") is True
        ) or (
            isinstance(nested_body, dict) and nested_body.get("success") is True
        )

        if api_success:
            return "Thank you — your feedback has been emailed successfully." + local_save_note

        detail = ""
        if isinstance(result, dict):
            detail = result.get("message") or result.get("error") or ""
            if not detail and isinstance(nested_body, dict):
                detail = nested_body.get("message") or ""
        backup = " It was saved temporarily on the app." if saved_locally else ""
        return ("Your comment was not accepted by Web3Forms. "
                + (str(detail)[:300] if detail else "Check the access key and Web3Forms settings.")
                + backup + local_save_note)

    except urllib.error.HTTPError as exc:
        error_text = exc.read().decode("utf-8", errors="replace")
        try:
            error_data = json.loads(error_text) if error_text else {}
            detail = error_data.get("message") or error_data.get("error") or ""
        except (json.JSONDecodeError, AttributeError):
            detail = ""
        backup = " Your comment was saved temporarily on the app." if saved_locally else ""
        return (f"Web3Forms returned HTTP {exc.code}. "
                + (str(detail)[:300] if detail else "Check your access key and Web3Forms settings.")
                + backup + local_save_note)
    except (urllib.error.URLError, TimeoutError) as exc:
        backup = " Your comment was saved temporarily on the app." if saved_locally else ""
        return ("The feedback email could not reach Web3Forms. Check network access and try again. "
                f"({type(exc).__name__})" + backup + local_save_note)
    except (json.JSONDecodeError, ValueError) as exc:
        backup = " Your comment was saved temporarily on the app." if saved_locally else ""
        return ("Web3Forms returned an unreadable response. Please try again. "
                f"({type(exc).__name__})" + backup + local_save_note)
    except Exception as exc:
        backup = " Your comment was saved temporarily on the app." if saved_locally else ""
        return ("An unexpected error occurred while submitting feedback. "
                f"({type(exc).__name__})" + backup + local_save_note)


CUSTOM_CSS = r"""
:root { --ink:#142238; --muted:#64748b; --line:#e2e8f0; --navy:#10233f; --blue:#2563eb; }
.gradio-container { max-width: 1440px !important; background: #f4f7fb !important; font-family: Inter, ui-sans-serif, system-ui, sans-serif !important; }
#hero { background: linear-gradient(120deg,#10233f 0%,#1c3c62 62%,#2563a6 100%); color:white; padding:30px 34px; border-radius:18px; margin-bottom:18px; }
#hero h1 { color:white; font-size:30px; letter-spacing:-.7px; margin:4px 0 8px; }
#hero p { color:#dbeafe; max-width:850px; margin:0; }
.eyebrow { color:#93c5fd; font-size:11px; font-weight:800; letter-spacing:2px; }
.kpi-grid { display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:12px; margin:8px 0 18px; }
.kpi { background:white; border:1px solid #e2e8f0; border-radius:13px; padding:17px 18px; box-shadow:0 3px 12px rgba(15,23,42,.04); }
.kpi span { display:block; color:#64748b; font-size:10px; font-weight:800; letter-spacing:1.2px; }
.kpi strong { display:block; color:#10233f; font-size:28px; margin:7px 0 2px; }
.kpi small { color:#64748b; font-size:11px; }
.empty-kpi { background:white; border:1px dashed #cbd5e1; border-radius:12px; padding:22px; color:#64748b; }
button.primary { background:#1d4ed8 !important; }
footer { display:none !important; }
@media(max-width:760px){ .kpi-grid{grid-template-columns:repeat(2,minmax(0,1fr));} #hero{padding:22px;} #hero h1{font-size:24px;} }
"""

with gr.Blocks(title="CI/CD Log Anomaly Research Dashboard", theme=gr.themes.Soft(primary_hue="blue", neutral_hue="slate"), css=CUSTOM_CSS) as demo:
    gr.HTML("""<section id="hero"><div class="eyebrow">RESEARCH PROTOTYPE · CI/CD OBSERVABILITY</div>
      <h1>CI/CD Workflow Log Anomaly Dashboard</h1>
      <p>Character-CNN + Transformer classification for line-level analysis of continuous integration and delivery logs. Upload a log or paste text to inspect model predictions and class distributions.</p></section>""")
    with gr.Tabs():
        with gr.Tab("Analysis dashboard"):
            gr.Markdown("### Log analysis workspace\n<span class='section-note'>Upload a plain-text log or paste its contents. If both are supplied, the uploaded file is used.</span>")
            with gr.Row(equal_height=True):
                with gr.Column(scale=3):
                    pasted_input = gr.Textbox(label="Paste CI/CD log", placeholder="Paste GitHub Actions or other CI/CD output here…", lines=12)
                with gr.Column(scale=1):
                    file_input = gr.File(label="Upload .txt log", file_types=[".txt"], type="filepath")
                    analyze_button = gr.Button("Analyze workflow log", variant="primary", size="lg")
                    clear_button = gr.ClearButton([pasted_input, file_input], value="Clear inputs")
            gr.Markdown("### Prediction statistics")
            stats_output = gr.HTML('<div class="empty-kpi">Run an analysis to see prediction statistics.</div>')
            with gr.Row():
                with gr.Column(scale=3):
                    gr.Markdown("### Line-by-line prediction table")
                    output_table = gr.Dataframe(headers=["Input line #", "Predicted class", "Top score (%)", "Decision", "Cleaned log text"], datatype=["number", "str", "number", "str", "str"], label="Predictions", interactive=False, wrap=True)
                with gr.Column(scale=2):
                    gr.Markdown("### Class distribution / confusion-style summary")
                    class_table = gr.Dataframe(headers=["Class", "Lines", "Share (%)", "Type"], datatype=["str", "number", "number", "str"], interactive=False, label="Predicted class counts")
                    summary_output = gr.Markdown(label="Model summary")
            analyze_button.click(fn=analyze_uploaded_or_pasted_log, inputs=[pasted_input, file_input], outputs=[output_table, summary_output, stats_output, class_table])
            gr.Markdown("**Interpretation note:** This is a predicted-class distribution, not a ground-truth confusion matrix. A true confusion matrix requires known labels for the submitted log. Softmax scores are not calibrated probabilities. Predictions are review signals, not proof of an incident. Avoid submitting passwords, access tokens, private keys, or confidential logs.")
        with gr.Tab("Class guide"):
            gr.Markdown("## Explanation of predicted classes\nThe labels below describe the five categories used by this research classifier. The model prediction should be reviewed in the context of the full workflow.")
            class_explanations = [
                ("Normal", "No anomaly category was predicted for the line. This does not guarantee that the complete workflow is free of problems."),
                ("Flaky test", "A test-related instability pattern, such as inconsistent or intermittent test execution behavior."),
                ("Configuration drift", "A possible mismatch or change in environment, dependency, build, or workflow configuration."),
                ("Security anomaly", "A log pattern associated with a potential security-related issue, such as suspicious authentication or secret-handling behavior."),
                ("Silent failure", "A possible failure pattern where execution does not produce a clear, conventional error signal."),
            ]
            for title, description in class_explanations:
                gr.Markdown(f"### {title}\n{description}")
        with gr.Tab("About"):
            gr.Markdown("## About this research dashboard\nThis web application demonstrates line-level CI/CD workflow log classification using a Character-CNN + Transformer model. It is intended for research and exploratory analysis; it is not a replacement for operational monitoring or security incident response.")
            gr.Markdown("### Developer\n**Taddele Petros**")
            gr.Markdown("### Model limitations\n- Predictions depend on the training data and preprocessing pipeline.\n- Softmax scores are not calibrated probabilities.\n- A predicted class is not proof of a real-world incident.\n- Similar log lines and surrounding context can affect predictions.")
        with gr.Tab("Feedback & comments"):
            gr.Markdown("## Share feedback\nReport a problem, suggest an improvement, or leave a comment about the research dashboard.")
            with gr.Row():
                feedback_name = gr.Textbox(label="Name (optional)", placeholder="Your name")
                feedback_email = gr.Textbox(label="Email (optional)", placeholder="you@example.com")
            feedback_category = gr.Dropdown(["General comment", "Prediction issue", "Interface issue", "Research question", "Feature suggestion"], value="General comment", label="Feedback category")
            feedback_message = gr.Textbox(label="Your comment", placeholder="Write your feedback here…", lines=6, max_lines=12)
            feedback_submit = gr.Button("Submit feedback", variant="primary")
            feedback_status = gr.Markdown()
            feedback_submit.click(fn=submit_feedback, inputs=[feedback_name, feedback_email, feedback_category, feedback_message], outputs=[feedback_status])
            gr.Markdown("<span class='section-note'>Feedback is emailed using Web3Forms. The app owner must configure WEB3FORMS_ACCESS_KEY in Render Environment settings. Please do not include passwords, access tokens, private keys, or confidential log data.</span>")
    gr.Markdown("<div style='text-align:center;color:#64748b;font-size:12px;padding:18px'>CI/CD Log Anomaly Research Dashboard · Developer: Taddele Petros</div>")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "10000"))
    print(f"[STARTUP] Launching Gradio on 0.0.0.0:{port}", flush=True)
    demo.queue(max_size=4, default_concurrency_limit=1)
    demo.launch(server_name="0.0.0.0", server_port=port, share=False, show_error=False, max_threads=2)
