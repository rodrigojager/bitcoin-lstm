from fastapi import FastAPI
from core.config import settings
from core.observability import instrument_app
from routers import ingest, train, series, init_backfill, metrics, futures, obs

app = FastAPI(
    title="BTC ML API",
    version="1.3.0",
    description=(
        "API para ingestão de candles do BTC (Binance), treino de modelos (LSTM),\n"
        "exposição de séries para visualização (real × previsto) e métricas de validação.\n"
        "Inclui endpoints prospectivos ('futuros') usados na análise de direção/erro."
    ),
    root_path=settings.API_ROOT_PATH or "",
)

instrument_app(app)

app.include_router(ingest.router)
app.include_router(train.router)
app.include_router(series.router)
app.include_router(init_backfill.router)
app.include_router(metrics.router)
app.include_router(futures.router)
app.include_router(obs.router)

# rota raiz para indicar status da API
@app.get("/")
def read_root():
	return {"status": "ok", "message": "Visite /docs para explorar os endpoints."}
