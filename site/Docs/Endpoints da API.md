# Endpoints da API

## Visão Geral

Esta documentação descreve todos os endpoints disponíveis na FastAPI do projeto BTC (pasta `api`). A API provê ingestão de dados do Bitcoin (via Binance), treino de modelos, séries para visualização (materializadas para carregamento rápido), métricas de validação, backfill histórico e uma série prospectiva (`futures`) com previsões feitas em tempo real e comparadas ao realizado.

---

## Raiz (status)

Endpoint simples para indicar disponibilidade da API e direcionar à documentação interativa.

### Detalhes Técnicos
- **Método HTTP**: `GET`
- **Rota**: `/`
- **Content-Type**: `application/json`

### Resposta
**Sucesso (200 OK)**:
```json
{ "status": "ok", "message": "Visite /docs para explorar os endpoints." }
```

---

## Ingestão de dados (Binance)

Obtém os candles mais recentes na Binance e insere/atualiza na tabela `btc_candles`. Integra a atualização da série prospectiva `futures` usando o penúltimo timestamp (evita retropreenchimento).

### Detalhes Técnicos
- **Método HTTP**: `POST`
- **Rota**: `/ingest`
- **Content-Type**: `application/json`

### Parâmetros de Entrada
Nenhum.

### Parâmetros de Saída
**Sucesso (200 OK)**:
```json
{ "status": "ok", "inserted": 89, "futures_updated": 1 }
```

**Erro (200 OK com status de erro)**:
```json
{ "status": "error", "message": "<detalhe do erro>" }
```

### Funcionamento Interno
1. Busca klines via `GET {BINANCE_BASE}/api/v3/klines` com `symbol`, `interval`, `limit`.
2. Normaliza payload para `time, open, high, low, close, volume`.
3. Upsert em `btc_candles` (conflito por `time` é ignorado).
4. Atualiza `futures` para o último `time` com par (usa T-1 → prevê T).

---

## Treino de modelos

Treina um modelo **LSTM (TensorFlow/Keras)** com duas saídas:
- Regressão multi-saída para prever o próximo candle (t→t+1): `open_next`, `high_next`, `low_next`, `close_next`, `amp_next`
- Classificação para direção (t→t+1): `prob_up`/`prob_down` e `dir_next`

O treino usa split temporal (80/20), **EarlyStopping** e salva os artefatos em volume (`models`).

### Detalhes Técnicos
- **Método HTTP**: `POST`
- **Rota**: `/train`
- **Content-Type**: `application/json`

### Parâmetros de Entrada
**Query**:
- `days` (int, 1..90, padrão 90): janela temporal de treino

### Parâmetros de Saída
**Sucesso (200 OK)**:
```json
{ "status": "ok", "samples": 25909, "mae": 535.53, "mape": 0.49, "smape": 0.52 }
```

**Erro (200 OK com status de erro)**:
```json
{ "status": "error", "message": "<detalhe do erro>" }
```

### Funcionamento Interno
1. Carrega janela de `days` da tabela `btc_candles`.
2. Constrói features/targets; aplica split temporal (80/20).
3. Monta sequências (janelas) e normaliza (fit no treino, aplica no val).
4. Treina a LSTM com EarlyStopping/Checkpoint.
5. Calcula MAE/MAPE/SMAPE no conjunto de validação; salva modelo + bundle (scalers/metadados).

### Treino automático (policy)
O endpoint `POST /train/auto` continua disponível, mas **na Opção C** a policy de retreino fica no **Quartz (.NET)** (Treino diário + checagem de drift). Em produção, o Site chama `POST /train` quando decide treinar.

---

## Série histórica para gráficos (on-demand)

Retorna série consolidada para visualização: candles reais, previsões (t→t+1), classificação direcional e erros relativos ao próximo candle.

### Detalhes Técnicos
- **Método HTTP**: `GET`
- **Rota**: `/series`
- **Content-Type**: `application/json`

### Parâmetros de Entrada
**Query**:
- `start` (string ISO8601, opcional)
- `end` (string ISO8601, opcional)
- `fallback_days` (int, padrão 90)

### Resposta
**Sucesso (200 OK)**:
```json
{
  "points": [
    {
      "real": { "time":"2025-09-26T12:00:00Z", "open": 0.0, "high": 0.0, "low": 0.0, "close": 0.0, "volume": 0.0 },
      "pred": { "open_next": 0.0, "high_next": 0.0, "low_next": 0.0, "close_next": 0.0, "amp_next": 0.0 },
      "cls": { "dir_next": 1, "prob_up": 0.52, "prob_down": 0.48 },
      "err": { "close_abs": 35.1, "close_signed": -12.4, "amp_abs": 58.7 }
    }
  ]
}
```

Quando não há dados suficientes:
```json
{ "points": [] }
```

---

## Série histórica materializada para gráficos

Retorna a série consolidada pronta para visualização a partir da tabela `series_cache` (materializada após cada treino).

### Detalhes Técnicos
- **Método HTTP**: `GET`
- **Rota**: `/series/cached`
- **Content-Type**: `application/json`

### Parâmetros de Entrada
**Query**:
- `start` (string ISO8601, opcional)
- `end` (string ISO8601, opcional)
- `fallback_days` (int, padrão 90)

### Resposta
Mesma estrutura de `/series`.

---

## Aplicação da série consolidada (pós-treino)

Materializa a série usada pelos gráficos após o treino de modelos.

### Detalhes Técnicos
- **Método HTTP**: `POST`
- **Rota**: `/series/rebuild`
- **Query**: `days` (int, 1..90, padrão 90)

### Resposta
```json
{ "status": "ok", "materialized": 25909 }
```

---

## Métricas de validação

Retorna métricas do último treino (extraídas de `job_logs`) e o início do período de validação, para sombreamento no front-end.

### Detalhes Técnicos
- **Método HTTP**: `GET`
- **Rota**: `/metrics`
- **Content-Type**: `application/json`

### Resposta
**Sucesso (200 OK)**:
```json
{
  "status": "ok",
  "mae": 535.53,
  "mape": 0.49,
  "smape": 0.52,
  "samples": 25909,
  "split_train": 20727,
  "split_total": 25909,
  "started_at": "2025-09-27T00:00:00Z",
  "finished_at": "2025-09-27T00:02:00Z",
  "validation_start": "2025-09-26T18:00:00Z"
}
```

Quando não há treino ainda:
```json
{ "status": "empty" }
```

---

## Observabilidade (Prometheus)

Endpoint para scraping Prometheus (latência e volume HTTP):
- **Método HTTP**: `GET`
- **Rota**: `/obs/metrics`

---

## Backfill histórico

Executa backfill de candles históricos na Binance para preencher lacunas e histórico definido.

### Detalhes Técnicos
- **Método HTTP**: `POST`
- **Rota**: `/init/backfill`
- **Content-Type**: `application/json`

### Parâmetros de Entrada
**Query (opcionais)**:
- `days` (int, 1..90) — padrão: `settings.BACKFILL_DAYS`
- `symbol` (string) — padrão: `settings.BINANCE_SYMBOL`
- `interval` (string) — padrão: `settings.BINANCE_INTERVAL`
- `sleep_ms` (int, >=0) — padrão: `settings.BACKFILL_SLEEP_MS`
- `limit` (int, 1..1000) — padrão: 1000

### Resposta
**Sucesso (200 OK)**:
```json
{ "status":"ok", "fetched": 5400, "inserted": 5200, "calls": 6, "days": 30 }
```

**Erro (200 OK com status de erro)**:
```json
{ "status":"error", "message":"<detalhe do erro>" }
```

---

## Futures (série prospectiva)

Série de previsões prospectivas (feitas em t−1 e comparadas ao real em t), usada na aba de Futuros e para métricas direcionais.

### Atualização da última amostra
- **Método HTTP**: `POST`
- **Rota**: `/futures/update`
- **Resposta**:
```json
{ "status":"ok", "updated": 1 }
```

### Consulta
- **Método HTTP**: `GET`
- **Rota**: `/futures`
- **Query (opcionais)**: `start`, `end` (ISO8601), `limit` (int; retorna os últimos N pontos)
- **Resposta**:
```json
{
  "points": [
    { "time":"2025-09-26T12:00:00Z", "pred_close": 64210.2, "real_close": 64195.7, "err_close": 14.5 }
  ]
}
```

---

## Modelo de Dados (principais tabelas)

- `btc_candles(time TIMESTAMP PRIMARY KEY, open NUMERIC, high NUMERIC, low NUMERIC, close NUMERIC, volume NUMERIC)`
- `job_logs(id SERIAL, job_name TEXT, status TEXT, message TEXT, started_at TIMESTAMP, finished_at TIMESTAMP)`
- `series_cache(time TIMESTAMP PRIMARY KEY, open NUMERIC, high NUMERIC, low NUMERIC, close NUMERIC, volume NUMERIC, pred_open_next NUMERIC, pred_high_next NUMERIC, pred_low_next NUMERIC, pred_close_next NUMERIC, pred_amp_next NUMERIC, cls_dir_next INTEGER, prob_up NUMERIC, prob_down NUMERIC, err_close_abs NUMERIC, err_close_signed NUMERIC, err_amp_abs NUMERIC)`
- `futures(time TIMESTAMP PRIMARY KEY, pred_close NUMERIC, real_close NUMERIC, err_close NUMERIC)`
