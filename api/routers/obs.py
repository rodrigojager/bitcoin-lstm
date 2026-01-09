from __future__ import annotations

from fastapi import APIRouter, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

router = APIRouter(prefix="/obs", tags=["observability"])


@router.get("/metrics", summary="Métricas Prometheus")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


