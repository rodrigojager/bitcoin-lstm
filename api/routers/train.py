from datetime import datetime, timezone

from fastapi import APIRouter, Query

from core.config import settings
from core.db import pg_conn
from models.schemas import TrainResponse
from services.series_cache_service import build_series_cache
from services.training_service import train_job

router = APIRouter(prefix="/train", tags=["train"])

@router.post("", response_model=TrainResponse, summary="Treino de modelos (LSTM)", description="Treina um modelo LSTM (multi-saída) para prever OHLC/amp do próximo candle e um head de classificação para direção. Retorna métricas de validação para close_next.")
def train(days: int = Query(90, ge=1, le=90)):
    return train_job(days=days)


def _last_train_finished_at() -> datetime | None:
	with pg_conn() as conn:
		with conn.cursor() as cur:
			cur.execute(
				"""
				SELECT finished_at
				FROM job_logs
				WHERE job_name='train' AND status='ok'
				ORDER BY id DESC
				LIMIT 1;
				"""
			)
			row = cur.fetchone()
			return row[0] if row else None


def _rolling_futures_mape(n: int) -> float | None:
	# MAPE(%) = mean(err_close / real_close) * 100 nos últimos N pontos com real_close válido
	n = int(max(1, n))
	with pg_conn() as conn:
		with conn.cursor() as cur:
			cur.execute(
				"""
				SELECT err_close::float8, real_close::float8
				FROM futures
				WHERE real_close IS NOT NULL AND err_close IS NOT NULL AND real_close <> 0
				ORDER BY time DESC
				LIMIT %s;
				""",
				(n,),
			)
			rows = cur.fetchall()
	if not rows:
		return None
	import numpy as np
	err = np.asarray([r[0] for r in rows], dtype="float64")
	real = np.asarray([r[1] for r in rows], dtype="float64")
	return float(np.mean(np.abs(err) / real) * 100.0)


@router.post(
	"/auto",
	summary="Treino automático (policy)",
	description=(
		"Executa treino somente quando necessário, baseado em policy:\n"
		"- Treina se o último treino foi há >= TRAIN_MAX_HOURS.\n"
		"- OU, se já passou >= TRAIN_MIN_HOURS e o MAPE(rolling) em futures >= FUTURES_MAPE_THRESHOLD.\n"
		"Os parâmetros vêm de env/arquivo (TRAIN_POLICY_PATH)."
	),
)
def train_auto(days: int = Query(90, ge=1, le=90)):
	now = datetime.now(timezone.utc).replace(tzinfo=None)
	last = _last_train_finished_at()

	if last is None:
		out = train_job(days=days)
		trained = out.get("status") == "ok"
		out.update({"auto": True, "trained": trained, "reason": "no_previous_train"})
		return out

	hours_since = (now - last).total_seconds() / 3600.0

	# Força por janela máxima
	if hours_since >= float(settings.TRAIN_MAX_HOURS):
		out = train_job(days=days)
		trained = out.get("status") == "ok"
		out.update({"auto": True, "trained": trained, "reason": "max_hours_exceeded", "hours_since_last": hours_since})
		return out

	# Gatilho por erro após janela mínima
	if hours_since >= float(settings.TRAIN_MIN_HOURS):
		try:
			mape = _rolling_futures_mape(int(settings.FUTURES_ROLLING_N))
		except Exception:
			mape = None
		if mape is not None and mape >= float(settings.FUTURES_MAPE_THRESHOLD):
			out = train_job(days=days)
			trained = out.get("status") == "ok"
			out.update(
				{
					"auto": True,
					"trained": trained,
					"reason": "mape_threshold_exceeded",
					"hours_since_last": hours_since,
					"futures_mape": mape,
				}
			)
			return out

		return {
			"status": "ok",
			"auto": True,
			"trained": False,
			"reason": "no_need",
			"hours_since_last": hours_since,
			"futures_mape": mape,
		}

	return {"status": "ok", "auto": True, "trained": False, "reason": "min_hours_not_reached", "hours_since_last": hours_since}


@router.post("/apply", summary="Materializa série consolidada pós-treino")
def apply_series(days: int = Query(90, ge=1, le=90)):
    n = build_series_cache(days)
    return {"status":"ok","materialized": n}
