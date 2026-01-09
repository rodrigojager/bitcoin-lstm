from fastapi import APIRouter, Query
from typing import Optional
from services.ingestion_service import backfill_job
from core.config import settings
from models.schemas import BackfillResponse

router = APIRouter(prefix="/init", tags=["init"])

@router.post("/backfill", response_model=BackfillResponse, summary="Backfill histórico de candles", description="Busca candles históricos na Binance e persiste em btc_candles, respeitando limites e pausa entre chamadas.")
def backfill(
    days: Optional[int] = Query(None, ge=1, le=90),
    symbol: Optional[str] = Query(None),
    interval: Optional[str] = Query(None),
    sleep_ms: Optional[int] = Query(None, ge=0),
    limit: Optional[int] = Query(None, ge=1, le=1000)
):
    return backfill_job(
        days=days or settings.BACKFILL_DAYS,
        symbol=symbol or settings.BINANCE_SYMBOL,
        interval=interval or settings.BINANCE_INTERVAL,
        sleep_ms=sleep_ms if sleep_ms is not None else settings.BACKFILL_SLEEP_MS,
        limit=limit or 1000
    )
