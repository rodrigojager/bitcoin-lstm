from fastapi import APIRouter
from datetime import datetime
from core.db import pg_conn
from ml.features import build_features_targets
from core.config import settings
from models.schemas import MetricsResponse

router = APIRouter(prefix="/metrics", tags=["metrics"])


def parse_metrics(msg: str):
	# Exemplo de msg:
	# "Treinado 90d, n=25909, split=20727/25909. EPOCHS=12. VAL_LOSS=0.123456. Val close_next -> MAE=535.5278, RMSE=700.1234, MAPE=0.49%, SMAPE=0.52%"
	import re
	out = {}
	try:
		mae = re.search(r"MAE=([0-9.]+)", msg)
		rmse = re.search(r"RMSE=([0-9.]+)", msg)
		mape = re.search(r"MAPE=([0-9.]+)%", msg)
		smape = re.search(r"SMAPE=([0-9.]+)%", msg)
		n = re.search(r"n=(\d+)", msg)
		split = re.search(r"split=(\d+)/(\d+)", msg)
		epochs = re.search(r"EPOCHS=(\d+)", msg)
		# Log formata como "VAL_LOSS=0.123456." (com ponto final). Capturar apenas o número.
		val_loss = re.search(r"VAL_LOSS=([0-9]+(?:\.[0-9]+)?)", msg)
		if mae: out["mae"] = float(mae.group(1))
		if rmse: out["rmse"] = float(rmse.group(1))
		if mape: out["mape"] = float(mape.group(1))
		if smape: out["smape"] = float(smape.group(1))
		if n: out["samples"] = int(n.group(1))
		if split: out["split_train"] = int(split.group(1)); out["split_total"] = int(split.group(2))
		# response_model usa "epochs"
		if epochs: out["epochs"] = int(epochs.group(1))
		if val_loss: out["val_loss"] = float(val_loss.group(1))
	except Exception:
		pass
	return out


def compute_validation_start_iso() -> str | None:
	# Reconstroi a janela usada no treino (LOOKBACK_DAYS) e aplica a mesma regra de split
	from pandas import read_sql
	try:
		with pg_conn() as conn:
			df = read_sql(
				"""
				SELECT time, open, high, low, close, volume
				FROM btc_candles
				WHERE time >= NOW() - INTERVAL %s
				ORDER BY time
				""",
				conn,
				params=(f"{settings.LOOKBACK_DAYS} days",),
			)
		if df.empty:
			return None
		df2, X, *_ = build_features_targets(df)
		n = len(X)
		if n == 0:
			return None
		split_idx = max(int(n * 0.8), n - 500)
		if split_idx >= len(df2):
			split_idx = len(df2) - 1
		return df2.iloc[split_idx]["time"].isoformat()
	except Exception:
		return None


@router.get("", response_model=MetricsResponse, summary="Métricas do último treino", description="Extrai métricas (MAE, MAPE, SMAPE) e metadados do último treino bem-sucedido, além do início do período de validação para sombreamento no front-end.")
def get_metrics():
	with pg_conn() as conn:
		with conn.cursor() as cur:
			cur.execute(
				"""
				SELECT message, started_at, finished_at
				FROM job_logs
				WHERE job_name='train' AND status='ok'
				ORDER BY id DESC
				LIMIT 1;
				"""
			)
			row = cur.fetchone()
			if not row:
				return {"status":"empty"}
			msg, started_at, finished_at = row
			m = parse_metrics(msg)
			m.update({
				"status": "ok",
				"started_at": started_at.isoformat() if isinstance(started_at, datetime) else str(started_at),
				"finished_at": finished_at.isoformat() if isinstance(finished_at, datetime) else str(finished_at),
				"validation_start": compute_validation_start_iso(),
			})
			return m
