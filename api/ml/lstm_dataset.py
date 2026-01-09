from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SequenceDataset:
    X_seq: np.ndarray  # (n, seq_len, n_features)
    y_reg: np.ndarray  # (n, n_targets)
    y_cls: np.ndarray  # (n,) 0/1
    # index_original aponta para a linha do dataframe/feature X que corresponde ao fim da janela (t)
    index_original: np.ndarray  # (n,) int


def build_x_sequences(X: pd.DataFrame, seq_len: int) -> tuple[np.ndarray, np.ndarray]:
    """Cria sequências apenas de X para inferência.

    Retorna:
    - X_seq: (n_seq, seq_len, n_features)
    - index_original: índices do ponto 't' (fim da janela) no X original
    """
    if seq_len < 2:
        raise ValueError("seq_len deve ser >= 2")
    n = len(X)
    if n <= seq_len:
        raise ValueError("Dados insuficientes para montar sequências")

    Xv = X.to_numpy(dtype=np.float32, copy=True)
    n_seq = n - (seq_len - 1)
    X_seq = np.zeros((n_seq, seq_len, Xv.shape[1]), dtype=np.float32)
    idx_orig = np.zeros((n_seq,), dtype=np.int64)

    j = 0
    for i in range(seq_len - 1, n):
        X_seq[j] = Xv[i - seq_len + 1 : i + 1]
        idx_orig[j] = i
        j += 1

    return X_seq, idx_orig


def build_sequences(
    X: pd.DataFrame,
    y_reg: pd.DataFrame,
    y_cls: pd.Series,
    seq_len: int,
) -> SequenceDataset:
    if seq_len < 2:
        raise ValueError("seq_len deve ser >= 2")
    if len(X) != len(y_reg) or len(X) != len(y_cls):
        raise ValueError("X, y_reg e y_cls devem ter o mesmo comprimento")
    n = len(X)
    if n <= seq_len:
        raise ValueError("Dados insuficientes para montar sequências")

    Xv = X.to_numpy(dtype=np.float32, copy=True)
    Yreg = y_reg.to_numpy(dtype=np.float32, copy=True)
    Ycls = y_cls.to_numpy(dtype=np.int64, copy=True)

    n_seq = n - (seq_len - 1)
    X_seq = np.zeros((n_seq, seq_len, Xv.shape[1]), dtype=np.float32)
    y_reg_out = np.zeros((n_seq, Yreg.shape[1]), dtype=np.float32)
    y_cls_out = np.zeros((n_seq,), dtype=np.int64)
    idx_orig = np.zeros((n_seq,), dtype=np.int64)

    j = 0
    for i in range(seq_len - 1, n):
        X_seq[j] = Xv[i - seq_len + 1 : i + 1]
        y_reg_out[j] = Yreg[i]
        y_cls_out[j] = Ycls[i]
        idx_orig[j] = i
        j += 1

    return SequenceDataset(X_seq=X_seq, y_reg=y_reg_out, y_cls=y_cls_out, index_original=idx_orig)


def temporal_split_indices(n: int, holdout_max: int = 500, train_ratio: float = 0.8) -> int:
    """Retorna split_idx (início da validação) para um split temporal."""
    if n <= 1:
        return 0
    split_idx = max(int(n * train_ratio), n - holdout_max)
    split_idx = max(1, min(split_idx, n - 1))
    return split_idx


def rolling_mape_from_futures(err_close: np.ndarray, real_close: np.ndarray) -> float:
    """MAPE(%) = mean(|err|/real)*100. Assume real_close > 0."""
    den = np.where(real_close == 0, 1e-9, real_close)
    return float(np.mean(np.abs(err_close) / den) * 100.0)


