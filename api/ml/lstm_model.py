from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class LstmModelConfig:
    seq_len: int
    n_features: int
    n_reg_targets: int
    lstm_units: int = 64
    dense_units: int = 64
    dropout: float = 0.2
    learning_rate: float = 1e-3


def build_lstm_multitask_model(cfg: LstmModelConfig):
    """Modelo multi-tarefa: regressão (OHLC/amp_next) + classificação (dir_next)."""
    import tensorflow as tf

    inp = tf.keras.Input(shape=(cfg.seq_len, cfg.n_features), name="x")
    x = tf.keras.layers.LSTM(cfg.lstm_units, return_sequences=False, name="lstm")(inp)
    x = tf.keras.layers.Dropout(cfg.dropout, name="dropout")(x)
    x = tf.keras.layers.Dense(cfg.dense_units, activation="relu", name="dense")(x)

    reg = tf.keras.layers.Dense(cfg.n_reg_targets, name="reg")(x)
    cls = tf.keras.layers.Dense(1, activation="sigmoid", name="cls")(x)

    model = tf.keras.Model(inputs=inp, outputs={"reg": reg, "cls": cls}, name="btc_lstm_multitask")

    opt = tf.keras.optimizers.Adam(learning_rate=cfg.learning_rate)
    model.compile(
        optimizer=opt,
        loss={"reg": "mse", "cls": "binary_crossentropy"},
        metrics={
            "reg": [tf.keras.metrics.MeanAbsoluteError(name="mae")],
            "cls": [tf.keras.metrics.BinaryAccuracy(name="acc")],
        },
    )
    return model







