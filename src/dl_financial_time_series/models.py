"""Tiny PyTorch sequence classifiers — every architecture is under 100k parameters.

All four models consume ``(batch, seq_len, n_features)`` float tensors of causal
features and return raw logits ``(batch,)`` for ``nn.BCEWithLogitsLoss``.
"""

from __future__ import annotations

import copy
import math

import numpy as np
import torch
from torch import nn

from dl_financial_time_series import config


class LSTMClassifier(nn.Module):
    def __init__(
        self,
        n_features: int,
        hidden: int = config.HIDDEN_SIZE,
        layers: int = 1,
        dropout: float = config.DROPOUT,
    ):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden, num_layers=layers, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.head(self.dropout(out[:, -1, :])).squeeze(-1)


class GRUClassifier(nn.Module):
    def __init__(
        self,
        n_features: int,
        hidden: int = config.HIDDEN_SIZE,
        layers: int = 1,
        dropout: float = config.DROPOUT,
    ):
        super().__init__()
        self.gru = nn.GRU(n_features, hidden, num_layers=layers, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(x)
        return self.head(self.dropout(out[:, -1, :])).squeeze(-1)


class TemporalCNN(nn.Module):
    """Stack of strictly causal (left-padded) 1-D conv blocks + global average pooling."""

    def __init__(
        self,
        n_features: int,
        channels: tuple[int, ...] = config.CNN_CHANNELS,
        kernel_size: int = 3,
        dropout: float = config.DROPOUT,
    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.convs = nn.ModuleList()
        in_ch = n_features
        for ch in channels:
            self.convs.append(nn.Conv1d(in_ch, ch, kernel_size))
            in_ch = ch
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(in_ch, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x.transpose(1, 2)  # (B, F, T)
        for conv in self.convs:
            h = nn.functional.pad(h, (self.kernel_size - 1, 0))  # causal: past only
            h = torch.relu(conv(h))
            h = self.dropout(h)
        h = h.mean(dim=2)  # global average over time
        return self.head(h).squeeze(-1)


class TinyTransformer(nn.Module):
    """1-2 layer Transformer encoder (d_model 32, 2 heads) with a causal mask."""

    def __init__(
        self,
        n_features: int,
        d_model: int = config.TRANSFORMER_D_MODEL,
        nhead: int = config.TRANSFORMER_HEADS,
        layers: int = 1,
        dim_ff: int = config.TRANSFORMER_FF,
        dropout: float = config.DROPOUT,
        max_len: int = 512,
    ):
        super().__init__()
        self.input_proj = nn.Linear(n_features, d_model)
        self.register_buffer("pe", self._positional_encoding(d_model, max_len), persistent=False)
        layer = nn.TransformerEncoderLayer(
            d_model,
            nhead,
            dim_feedforward=dim_ff,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.head = nn.Linear(d_model, 1)

    @staticmethod
    def _positional_encoding(d_model: int, max_len: int) -> torch.Tensor:
        pos = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        return pe

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Per-position encoder outputs (used by the causality unit test)."""
        h = self.input_proj(x) + self.pe[: x.shape[1]]
        mask = torch.triu(
            torch.ones(x.shape[1], x.shape[1], dtype=torch.bool, device=x.device), diagonal=1
        )
        return self.encoder(h, mask=mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encode(x)
        return self.head(h.mean(dim=1)).squeeze(-1)


def build_model(name: str, n_features: int) -> nn.Module:
    if name == "lstm":
        return LSTMClassifier(n_features)
    if name == "gru":
        return GRUClassifier(n_features)
    if name == "cnn":
        return TemporalCNN(n_features)
    if name == "transformer":
        return TinyTransformer(n_features)
    raise ValueError(f"unknown model: {name!r}")


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_model(
    model: nn.Module,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    *,
    epochs: int = config.MAX_EPOCHS,
    batch_size: int = config.BATCH_SIZE,
    lr: float = config.LEARNING_RATE,
    weight_decay: float = config.WEIGHT_DECAY,
    patience: int = config.PATIENCE,
    seed: int = config.SEED,
    shuffle_train: bool = True,
) -> tuple[nn.Module, list[dict]]:
    """Train with Adam + early stopping on validation loss; restore best weights.

    Minibatch shuffling happens only inside the training window (never across
    time); the shuffle generator is seeded for reproducibility.
    """
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    train_x = torch.as_tensor(np.asarray(x_train, dtype=np.float32))
    train_y = torch.as_tensor(np.asarray(y_train, dtype=np.float32))
    val_x = torch.as_tensor(np.asarray(x_val, dtype=np.float32))
    val_y = torch.as_tensor(np.asarray(y_val, dtype=np.float32))
    dataset = torch.utils.data.TensorDataset(train_x, train_y)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle_train, generator=gen
    )
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.BCEWithLogitsLoss()
    best_val, best_state, wait = math.inf, None, 0
    history: list[dict] = []
    for epoch in range(epochs):
        model.train()
        total, seen = 0.0, 0
        for xb, yb in loader:
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            total += float(loss.item()) * len(yb)
            seen += len(yb)
        model.eval()
        with torch.no_grad():
            val_loss = float(loss_fn(model(val_x), val_y))
        history.append(
            {"epoch": epoch + 1, "train_loss": total / max(seen, 1), "val_loss": val_loss}
        )
        if val_loss < best_val - 1e-5:
            best_val, wait = val_loss, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            wait += 1
            if wait >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history


def predict_proba(model: nn.Module, x: np.ndarray) -> np.ndarray:
    """Sigmoid probabilities from logits (eval mode, no grad)."""
    model.eval()
    with torch.no_grad():
        logits = model(torch.as_tensor(np.asarray(x, dtype=np.float32)))
    return torch.sigmoid(logits).cpu().numpy()
