from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import joblib

from ml.model_paths import LSTM_BUNDLE_PATH, LSTM_MODEL_PATH


@dataclass(frozen=True)
class LstmBundle:
    model: object  # tf.keras.Model
    scaler_x: object  # sklearn scaler
    scaler_y: object  # sklearn scaler (targets)
    feature_cols: list[str]
    target_reg_cols: list[str]
    seq_len: int


_CACHE: Optional[LstmBundle] = None


def clear_bundle_cache():
    global _CACHE
    _CACHE = None


def load_bundle(force_reload: bool = False) -> LstmBundle:
    global _CACHE
    if _CACHE is not None and not force_reload:
        return _CACHE

    meta = joblib.load(LSTM_BUNDLE_PATH)
    model_path = meta.get("model_path") or LSTM_MODEL_PATH

    # Import pesado: só aqui.
    import tensorflow as tf

    model = tf.keras.models.load_model(model_path)
    bundle = LstmBundle(
        model=model,
        scaler_x=meta["scaler_x"],
        scaler_y=meta["scaler_y"],
        feature_cols=list(meta["feature_cols"]),
        target_reg_cols=list(meta["target_reg_cols"]),
        seq_len=int(meta["seq_len"]),
    )
    _CACHE = bundle
    return bundle







