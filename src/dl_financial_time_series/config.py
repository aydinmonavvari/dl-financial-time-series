"""Central configuration for the deep-learning financial time-series study."""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = PROJECT_ROOT / "figures"

# --- Reproducibility ---------------------------------------------------------
SEED = 42

# --- Study design -------------------------------------------------------------
SEQ_LEN = 30  # sequence length (trading days) fed to the sequence models
N_FOLDS = 3
TRAIN_END_FRACS = (0.50, 0.66, 0.82)  # expanding-window train boundaries (fractions of usable rows)
VAL_FRAC = 0.05  # validation window between train end and test start
EMBARGO_DAYS = 1  # rows skipped at each segment boundary (label at t needs close at t+1)

# --- Training (deep learning) -------------------------------------------------
MAX_EPOCHS = 20
PATIENCE = 5  # early-stopping patience on validation loss
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
DROPOUT = 0.2
HIDDEN_SIZE = 32  # LSTM / GRU hidden size
TRANSFORMER_D_MODEL = 32
TRANSFORMER_HEADS = 2
TRANSFORMER_FF = 64
CNN_CHANNELS = (32, 32, 32)  # 3 causal conv blocks
PARAM_BUDGET = 100_000  # hard cap on trainable parameters per architecture

# --- Economics ----------------------------------------------------------------
COST_BPS_PER_SIDE = 10.0  # transaction cost per side (buy or sell), basis points

# --- Model registry -----------------------------------------------------------
MODELS = ("lstm", "gru", "cnn", "transformer")
