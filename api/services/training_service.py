import os
from datetime import datetime

import joblib
import pandas as pd

from core.config import settings
from core.db import pg_conn
from core.logging import log_job
from ml.features import build_features_targets, exp_sample_weights, FEATURE_COLS, TARGET_REG_COLS
from ml.lstm_dataset import build_sequences, temporal_split_indices
from ml.lstm_model import LstmModelConfig, build_lstm_multitask_model
from ml.model_paths import LSTM_BUNDLE_PATH, LSTM_MODEL_PATH


def load_candles_window(days: int) -> pd.DataFrame:
	with pg_conn() as conn:
		q = """SELECT time, open, high, low, close, volume
			   FROM btc_candles
			   WHERE time >= NOW() - INTERVAL %s
			   ORDER BY time;"""
		return pd.read_sql(q, conn, params=(f'{days} days',))


def mean_absolute_percentage_error(y_true, y_pred):
	import numpy as np
	y_true = np.asarray(y_true)
	y_pred = np.asarray(y_pred)
	den = np.where(y_true == 0, 1e-9, y_true)
	return float((np.abs((y_true - y_pred) / den)).mean() * 100.0)


def symmetric_mape(y_true, y_pred):
	import numpy as np
	y_true = np.asarray(y_true)
	y_pred = np.asarray(y_pred)
	den = (np.abs(y_true) + np.abs(y_pred))
	den = np.where(den == 0, 1e-9, den)
	return float((2.0 * np.abs(y_pred - y_true) / den).mean() * 100.0)


def train_job(days: int|None=None, alpha: float|None=None):
	days = days or settings.LOOKBACK_DAYS
	alpha = alpha or settings.ALPHA_DECAY
	start = datetime.utcnow()
	try:
		df = load_candles_window(days)
		if len(df) < 500:
			raise RuntimeError("Dados insuficientes para treino (mínimo ~500 candles).")

		df2, X, Yreg, Ycls = build_features_targets(df)

		seq_len = int(settings.LSTM_SEQ_LEN)
		ds = build_sequences(X, Yreg, Ycls, seq_len=seq_len)
		n_seq = len(ds.X_seq)
		split_idx = temporal_split_indices(n_seq, holdout_max=500, train_ratio=0.8)

		X_train_raw, X_val_raw = ds.X_seq[:split_idx], ds.X_seq[split_idx:]
		Yreg_train_raw, Yreg_val_raw = ds.y_reg[:split_idx], ds.y_reg[split_idx:]
		Ycls_train, Ycls_val = ds.y_cls[:split_idx], ds.y_cls[split_idx:]

		# Pesos exponenciais apenas no treino (mais peso ao recente)
		w_train = exp_sample_weights(len(X_train_raw), alpha).astype("float32")

		# Normalização (fit apenas no treino)
		from sklearn.preprocessing import MinMaxScaler
		import numpy as np

		scaler_x = MinMaxScaler()
		scaler_y = MinMaxScaler()

		X_train_2d = X_train_raw.reshape((X_train_raw.shape[0] * X_train_raw.shape[1], X_train_raw.shape[2]))
		scaler_x.fit(X_train_2d)
		X_train = scaler_x.transform(X_train_2d).reshape(X_train_raw.shape).astype("float32")

		X_val_2d = X_val_raw.reshape((X_val_raw.shape[0] * X_val_raw.shape[1], X_val_raw.shape[2]))
		X_val = scaler_x.transform(X_val_2d).reshape(X_val_raw.shape).astype("float32")

		scaler_y.fit(Yreg_train_raw)
		Yreg_train = scaler_y.transform(Yreg_train_raw).astype("float32")
		Yreg_val = scaler_y.transform(Yreg_val_raw).astype("float32")

		# Modelo
		cfg = LstmModelConfig(
			seq_len=seq_len,
			n_features=X.shape[1],
			n_reg_targets=Yreg.shape[1],
			learning_rate=float(settings.LSTM_LR),
		)
		model = build_lstm_multitask_model(cfg)

		import tensorflow as tf
		callbacks = [
			tf.keras.callbacks.EarlyStopping(
				monitor="val_loss",
				patience=int(settings.LSTM_PATIENCE),
				restore_best_weights=True,
			),
			tf.keras.callbacks.ModelCheckpoint(
				filepath=LSTM_MODEL_PATH,
				monitor="val_loss",
				save_best_only=True,
			),
		]

		# Treino
		hist = model.fit(
			X_train,
			{"reg": Yreg_train, "cls": Ycls_train.astype("float32")},
			validation_data=(X_val, {"reg": Yreg_val, "cls": Ycls_val.astype("float32")}),
			epochs=int(settings.LSTM_EPOCHS),
			batch_size=int(settings.LSTM_BATCH_SIZE),
			sample_weight={"reg": w_train, "cls": w_train},
			verbose=0,
			callbacks=callbacks,
		)
		history = getattr(hist, "history", {}) or {}
		epochs_ran = int(len(history.get("loss", [])) or 0)
		try:
			val_loss_best = float(min(history.get("val_loss", []))) if history.get("val_loss") else None
		except Exception:
			val_loss_best = None

		# Carregar melhor checkpoint (se o callback salvou)
		try:
			model = tf.keras.models.load_model(LSTM_MODEL_PATH)
		except Exception:
			pass

		# Avaliação no conjunto de validação (close_next)
		pred = model.predict(X_val, verbose=0)
		reg_pred_scaled = pred["reg"]
		reg_pred = scaler_y.inverse_transform(reg_pred_scaled)
		reg_true = Yreg_val_raw

		close_idx = TARGET_REG_COLS.index("close_next")
		y_true = reg_true[:, close_idx]
		y_pred = reg_pred[:, close_idx]

		from sklearn.metrics import mean_absolute_error, mean_squared_error
		mae = float(mean_absolute_error(y_true, y_pred))
		# scikit-learn 1.8+ removeu o parâmetro squared; usar sqrt() mantém compatibilidade
		rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
		mape = mean_absolute_percentage_error(y_true, y_pred)
		smape = symmetric_mape(y_true, y_pred)

		# Persistência: salvar bundle (scalers + metadados) e o caminho do modelo
		os.makedirs(os.path.dirname(LSTM_BUNDLE_PATH), exist_ok=True)
		joblib.dump(
			{
				"model_path": LSTM_MODEL_PATH,
				"scaler_x": scaler_x,
				"scaler_y": scaler_y,
				"feature_cols": FEATURE_COLS,
				"target_reg_cols": TARGET_REG_COLS,
				"seq_len": seq_len,
			},
			LSTM_BUNDLE_PATH,
		)

		msg = (
			f"Treinado {days}d, n={n_seq}, split={split_idx}/{n_seq}. "
			f"EPOCHS={epochs_ran}. "
			+ (f"VAL_LOSS={val_loss_best:.6f}. " if val_loss_best is not None else "")
			+ f"Val close_next -> MAE={mae:.4f}, RMSE={rmse:.4f}, MAPE={mape:.2f}%, SMAPE={smape:.2f}%"
		)
		log_job("train","ok", msg, start, datetime.utcnow())
		return {"status":"ok","samples":n_seq,"mae":mae,"mape":mape,"smape":smape}
	except Exception as e:
		log_job("train","error",str(e),start,datetime.utcnow())
		return {"status":"error","message":str(e)}
