from fastapi import APIRouter, Query
from typing import Optional
from services.prediction_service import series_data
from services.series_cache_service import load_series_cached
from models.schemas import SeriesResponse

router = APIRouter(prefix="/series", tags=["series"])

@router.get("", response_model=SeriesResponse, summary="Série consolidada para gráficos (on-demand)", description="Calcula on-demand a série consolidada (real × previsto). Para produção, prefira /series_cached.")
def series(start: Optional[str]=Query(None), end: Optional[str]=Query(None), fallback_days: int=90):
    return series_data(start, end, fallback_days)


@router.get("/cached", response_model=SeriesResponse, summary="Série consolidada materializada", description="Retorna a série já materializada em banco (series_cache), gerada pelo job de treino.")
def series_cached(start: Optional[str]=Query(None), end: Optional[str]=Query(None), fallback_days: int=90):
    return load_series_cached(start, end, fallback_days)


@router.post("/rebuild", summary="Recalcula e materializa a série consolidada")
def series_rebuild(days: int = Query(90, ge=1, le=90)):
    from services.series_cache_service import build_series_cache
    n = build_series_cache(days)
    return {"status":"ok","materialized": n}
