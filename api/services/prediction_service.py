import numpy as np
import pandas as pd
from typing import Optional

from core.db import pg_conn
from ml.features import build_features_targets, TARGET_REG_COLS
from ml.lstm_dataset import build_x_sequences
from services.lstm_bundle_service import load_bundle


def series_data(start: Optional[str], end: Optional[str], fallback_days: int=90):
	with pg_conn() as conn:
		if start and end:
			q = """SELECT time, open, high, low, close, volume FROM btc_candles
				   WHERE time BETWEEN %s AND %s ORDER BY time;"""
			df = pd.read_sql(q, conn, params=(start,end))
		else:
			q = """SELECT time, open, high, low, close, volume FROM btc_candles
				   WHERE time >= NOW() - INTERVAL %s ORDER BY time;"""
			df = pd.read_sql(q, conn, params=(f'{fallback_days} days',))
	if df.empty or len(df) < 30: return {"points":[]}

	df2, X, Yreg, Ycls = build_features_targets(df)
	try:
		bundle = load_bundle()
		seq_len = int(bundle.seq_len)
		n = len(X)
		reg_pred = np.full((n, len(TARGET_REG_COLS)), np.nan, dtype="float32")
		prob_up = np.full((n,), np.nan, dtype="float32")

		# Batch predict
		X_feat = X[bundle.feature_cols].copy()
		X_seq, idx_orig = build_x_sequences(X_feat, seq_len=seq_len)
		X2d = X_seq.reshape((X_seq.shape[0] * X_seq.shape[1], X_seq.shape[2]))
		X_scaled = bundle.scaler_x.transform(X2d).reshape(X_seq.shape).astype("float32")

		p = bundle.model.predict(X_scaled, verbose=0, batch_size=512)
		reg_all = bundle.scaler_y.inverse_transform(p["reg"]).astype("float32")
		cls_all = p["cls"].reshape((-1,)).astype("float32")

		reg_pred[idx_orig, :] = reg_all
		prob_up[idx_orig] = cls_all

		reg_pred = pd.DataFrame(reg_pred, columns=TARGET_REG_COLS, index=X.index)
		cls_pred = (prob_up >= 0.5).astype(int)
		prob = np.vstack([1.0 - prob_up, prob_up]).T
	except Exception:
		reg_pred = cls_pred = prob = None

	out = []
	for i in range(len(df2)):
		real = {k: (float(df2.iloc[i][k]) if k!="time" else df2.iloc[i]["time"].isoformat())
				for k in ["time","open","high","low","close","volume"]}
		pred = None
		if reg_pred is not None:
			row = reg_pred.iloc[i]
			if not np.isnan(row["close_next"]):
				pred = {k: float(row[k]) for k in TARGET_REG_COLS}

		clsinfo = None
		if cls_pred is not None and prob is not None:
			pup = prob[i][1]
			if not (np.isnan(pup) or np.isnan(prob[i][0])):
				clsinfo = {"dir_next": int(cls_pred[i]), "prob_up": float(prob[i][1]), "prob_down": float(prob[i][0])}
		err = None
		if pred is not None and i+1 < len(df2):
			real_next_close = float(df2.iloc[i+1]["close"])
			real_next_amp = float(df2.iloc[i+1]["high"] - df2.iloc[i+1]["low"])
			err = {
				"close_abs": abs(pred["close_next"] - real_next_close),
				"close_signed": pred["close_next"] - real_next_close,
				"amp_abs": abs(pred["amp_next"] - real_next_amp)
			}
		out.append({"real": real, "pred": pred, "cls": clsinfo, "err": err})
	return {"points": out}
