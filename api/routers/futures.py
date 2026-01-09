from fastapi import APIRouter, Query
from typing import Optional
from services.futures_service import save_predictions_for_times, load_futuros_series
from models.schemas import FuturesResponse, FutUpdateResponse

router = APIRouter(prefix="/futures", tags=["futures"])

@router.post("/update", response_model=FutUpdateResponse, summary="Atualiza 'futures' para o último timestamp", description="Calcula a previsão prospectiva (t→t+1) para o último candle disponível e persiste em 'futures'.")
def futures_update():
	# Atualiza somente o último timestamp disponível para evitar retro-preenchimento
	from core.db import pg_conn
	with pg_conn() as conn:
		with conn.cursor() as cur:
			cur.execute("SELECT MAX(time) FROM btc_candles")
			row = cur.fetchone()
			last_time = row[0] if row else None
	if not last_time:
		return {"status":"ok","updated": 0}
	try:
		inserted = save_predictions_for_times([last_time])
		return {"status":"ok","updated": inserted}
	except Exception as e:
		# Se o modelo ainda não está pronto, não derruba o endpoint (ajuda monitoramento)
		return {"status":"ok","updated": 0, "message": str(e)}

@router.get("", response_model=FuturesResponse, summary="Série prospectiva 'futures'", description="Retorna a série de previsões prospectivas (pred_close × real_close × err_close) alinhadas por timestamp.")
def futures_series(start: Optional[str]=Query(None), end: Optional[str]=Query(None), limit: Optional[int]=Query(None, ge=1, le=10000)):
    return load_futuros_series(start, end, limit=limit)
