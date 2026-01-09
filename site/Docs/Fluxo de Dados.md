# Fluxo de Dados e Decisões de Projeto

Este documento apresenta o fluxo completo da solução: de onde os dados nascem, por quais camadas de código passam, como são persistidos, como o modelo é treinado e como as previsões são consumidas na interface.

## 1) Aquisição de dados (Binance → API Python)

- Ponto de entrada (FastAPI): `routers/ingest.py` (POST `/ingest`)
  - Responsável por iniciar a cadeia de ingestão: chama os serviços de obtenção e persistência.
- Serviço de ingestão: `services/ingestion_service.py`
  - `fetch_binance_klines(...)` → Faz requisição REST à Binance (`/api/v3/klines`) usando `requests` com `symbol`, `interval` e `limit` vindos de variáveis de ambiente (ver `core/config.py`).
  - `normalize_klines_payload(...)` → Converte o payload para DataFrame com colunas padronizadas `[time, open, high, low, close, volume]`, ordena por tempo e remove campos não utilizados.
  - `upsert_candles(df)` → Insere no PostgreSQL (tabela `btc_candles`) com `INSERT ... ON CONFLICT (time) DO NOTHING`, garantindo idempotência.

## 2) Persistência (PostgreSQL)

- Tabela `btc_candles(time, open, high, low, close, volume)` → Série OHLCV “limpa” para consumo por treino e previsões.
- Tabela `job_logs(...)` → Registro de execução, métricas e mensagens de auditoria.
- Tabela `futures(time, pred_close, real_close, err_close)` → Série prospectiva: previsões feitas em t−1 comparadas com o real em t (sem retro-preencher passado).

## 3) Janela e features (engenharia de atributos)

- Arquivo: `ml/features.py`
  - Deriva features a partir de OHLCV (ex.: `ret`, `acc`, `amp`, `vol_rel`) e targets (`open_next`, `high_next`, `low_next`, `close_next`, `amp_next`, e `dir_next`).
  - Retorna DataFrame alinhado (descarta `NaN` decorrente de `shift`) pronto para treino/previsão.
- Decisão: features simples e interpretáveis para um horizonte curto (5 min). `vol_rel` normaliza volume; `amp` captura volatilidade local.

## 4) Treinamento (modelo e avaliação)

- Ponto de entrada (FastAPI): `routers/train.py` (POST `/train`)
- Serviço: `services/training_service.py`
  - Carrega janela temporal (configurável por `LOOKBACK_DAYS`).
  - `build_features_targets(df)` para obter `X` (features), `Yreg` (targets contínuos) e `Ycls` (direção).
  - Monta **sequências** para alimentar o LSTM (`lstm_seq_len`), preservando a ordem temporal.
  - Split temporal 80/20 (sem embaralhar), limitando validação mínima (até 500 amostras) para estabilidade.
  - Treino: **LSTM (TensorFlow/Keras)** com duas saídas:
    - Regressão multi-saída (open/high/low/close/amp do t+1)
    - Classificação direcional (prob_up/prob_down em t+1)
  - Usa EarlyStopping + checkpoint e exporta **modelo + scalers/metadados** para o volume.
  - Métricas no conjunto de validação (out-of-sample): MAE, MAPE, SMAPE (em `close_next`).
  - Persistência dos modelos (via `joblib`) e log das métricas (`job_logs`).
- Decisão: split temporal para evitar “vazamento temporal”; LSTM multi-saída mantém consistência entre os alvos e simplifica o deploy.

## 5) Previsões em série (histórico + t→t+1)

- Ponto de entrada (FastAPI): `routers/series.py` (GET `/series`)
- Serviço: `services/prediction_service.py`
  - Carrega `btc_candles` (intervalo pedido ou janela padrão), monta features e carrega modelos salvos.
  - Para cada linha i, prevê o próximo candle (i+1) e calcula erro relativo ao `close` real (quando disponível).
  - Entrega uma lista de pontos contendo: `real` (OHLCV), `pred` (targets i+1), `cls` (direção/probabilidades) e `err`.
- Decisão: alinhar previsão i→(i+1) para refletir “uso real” e facilitar construção de gráficos consistentes.

## 6) Série prospectiva “Futuros” (avaliação contínua)

- Pontos de entrada (FastAPI): `routers/futures.py`
  - POST `/futures/update` → Atualiza a linha prospectiva do último timestamp disponível (t).
  - GET `/futures` → Retorna a série prospectiva para análise (pred vs real vs erro).
- Serviço: `services/futures_service.py`
  - Garante que a previsão armazenada para o tempo t foi de fato gerada em t−1 (mapeando `time_next → idx_prev`).
  - Evita retro-preencher histórico: apenas timestamps prospectivos.
- Decisão: separar avaliação “on-line” de validação (split 80/20) para ter duas visões: robustez histórica e desempenho recente/operacional.

## 7) Métricas públicas (para UI)

- Ponto de entrada (FastAPI): `routers/metrics.py` (GET `/metrics`)
  - Extrai do último `job_logs` de treino as métricas e reconstruções auxiliares (início da validação) para sombreamento no front.
- Decisão: manter a UI independente da lógica de treino; a API expõe métrica consolidada, e a UI apenas consome.

## 8) Apresentação (Site ASP.NET Core 9.0)

- Camada de UI: `site/Views/Charts/Index.cshtml`
  - Busca `/series`, `/metrics`, `/futures` e constrói gráficos (ECharts) com OHLC, close, erro%, volume, futuros e direção.
  - Sombreamento da região de validação: ajuda leitura das métricas em contexto.
  - Spinners de loading simples: exibidos até as séries serem carregadas.
- Documentação: `DocsController` + `MarkdownService` renderizando `.md` (Docs); Swagger da API Python embutido em view dedicada.
- Jobs (Quartz.NET): `IngestJob`, `TrainDailyJob`, `TrainDriftJob` e `BackfillJob` disparam periodicamente os endpoints Python.
- Decisão: separar visualização (site) de processamento (API) simplifica implantação e escalabilidade.

## 9) Orquestração e Configuração

- Docker Compose: sobe `db` (Postgres), `pyapi` (FastAPI), `site` (ASP.NET Core) e `adminer`.
- Variáveis de ambiente: senhas e endpoints são lidos de `.env` na raiz (não versionado). O site injeta a `ConnectionStrings__DefaultConnection` por env var.
- Volumes: `./api/models` para persistir artefatos do modelo entre reinícios.
- Decisão: compose simplifica dev/ops; `.env` evita segredos em repositório; healthchecks garantem dependências.

## 9.1 Política de retreino (produção) — Opção C (Quartz)

A policy de retreino fica no **Quartz (.NET)**:
- **Treino diário fixo** (`TrainDailyJob`): chama `POST /train` e `POST /series/rebuild`.
- **Checagem de drift** (`TrainDriftJob`): roda periodicamente e treina somente se:
  - passou >= `TrainMinHours` desde o último treino e
  - `MAPE(rolling)` em `futures` >= `FuturesMapeThreshold` (calculado sobre os últimos `FuturesRollingN` pontos).

Configuração (sem UI) em `site/appsettings.json` (seção `PythonApi`).

## 10) Considerações de Design

- Temporal split (80/20) ao invés de K-fold: evita vazamento de futuro e reflete uso em produção.
- Modelo único (LSTM multi-saída): prevê múltiplos alvos do próximo candle e também uma probabilidade direcional (head de classificação).
- WDA e avaliação prospectiva: métricas direcionais complementam MAPE/SMAPE (nível) e aproximam utilidade econômica.
- Postgres: SQL padrão, consultas simples, integrável a Adminer; bom equilibrio entre robustez e praticidade.
- ECharts: controle fino de séries temporais, performance aceitável, conectividade de zoom/pan.

## 11) Caminhos no Código (resumo)

- Aquisição e ingestão: `api/services/ingestion_service.py` ← `api/routers/ingest.py`
- Features/targets: `api/ml/features.py`
- Treino: `api/services/training_service.py` ← `api/routers/train.py`
- Previsões de série: `api/services/prediction_service.py` ← `api/routers/series.py`
- Futuros (prospectivo): `api/services/futures_service.py` ← `api/routers/futures.py`
- Métricas: `api/routers/metrics.py`
- UI (gráficos): `site/Views/Charts/Index.cshtml`
- Jobs: `site/Jobs/*.cs`
- Config/env: `api/core/config.py`, `.env` (raiz), `docker-compose.yml`
