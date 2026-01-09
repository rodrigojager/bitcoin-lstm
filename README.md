## Apresentação

Este projeto apresenta uma solução completa que inclui uma API em Python que tem endpoints que servem para obter os dados referentes aos klines/candlesticks/velas dos preços do BitCoin conforme fornecido pela Binance, criar um modelo de regressão para predizer klines, modelo de classificação para dizer se a previsão é de subida ou descida e trazer informações para gerar gráficos para melhorar acompanhar os resultados. Existe um endpoint que serve para popular inicialmente o banco de dados com informações dos ultimos 90 dias, caso não haja dados suficientes recentes, executado uma única vez e endpoint para alimentar o banco de dados com o valor atual, sob demanda. A lista completa de endpoints com suas funcionalidades estão disponíveis no item do menu da documentação "Endpoints da API".

Além da API, tem um projeto em Asp .NET 9.0 MVC que além de servir esse site, tem jobs configurados através do Quartz.NET para fazer requisições para os endpoints da API para alimentar com dados atualizados o banco de dados em PostgreSQL a cada 5 minutos. O treinamento do modelo é orquestrado pelo Quartz (Opção C): um job diário executa `POST /train` e `POST /series/rebuild`, e outro job faz checagem de drift via MAPE(rolling) em `futures` para decidir se treina antes do diário. Tanto a API em Python, quanto a aplicação do site e os jobs em ASP .NET, quanto o PostgreSQL e o Adminer para consultar os dados gravados estão todos dentro de containers Docker, agrupados com Docker Compose e hospedados em minha VPS, gerenciados com Portainer e usando o Traefik para proxy reverso.

A API, a documentação com o site e os gráficos estão disponívelem https://rodrigojager.com/

---

## 1. Motivação

O desenvolvimento dessa solução visa atender ao Tech Challenge **Fase 04** do curso de Machine Learning Engineering da FIAP.

---

## 2. Resumo

Este projeto implementa uma solução completa de *machine learning* para Bitcoin, com:

- **Ingestão contínua de candles** (Binance) para um **PostgreSQL**.
- **Treino de um modelo LSTM** (TensorFlow/Keras) para:
  - **Regressão**: prever valores do **próximo candle** (t→t+1), incluindo `close_next` (e também `open/high/low` e `amp`).
  - **Classificação**: prever **direção** (subir/descer) do próximo candle (probabilidade de alta).
- **Deploy em uma API FastAPI** (Python) que:
  - expõe séries para gráficos (real × previsto),
  - materializa uma série “cacheada” para carregar rápido,
  - mantém uma série prospectiva (`futures`) para acompanhar erro *online*.
- **Orquestração via Quartz.NET** no **Site ASP.NET Core**:
  - um job para ingestão periódica,
  - um job de treino diário,
  - um job de checagem de drift (MAPE em `futures`) que dispara retreino quando necessário,
  - um job de backfill inicial quando a aplicação sobe.

---

## 3. O que é LSTM?

Um **LSTM** é um tipo de rede neural desenhada para **dados em sequência** (tempo), como candles.

- Ele lê uma sequência **passo a passo** (candle por candle).
- Mantém uma **memória interna** (*cell state*) e um estado curto (*hidden state*).
- Usa “**portas**” (*gates*) para decidir:
  - **o que guardar** do passado,
  - **o que esquecer**,
  - **o que usar agora** para gerar a previsão.

Ideia prática: ao invés de prever olhando apenas o candle atual, o LSTM tenta prever usando um **histórico recente** para capturar padrões temporais (tendência, aceleração, volatilidade, comportamento do volume etc.).

---

## 4. O que está sendo feito neste projeto (pipeline completo)

### 4.1 Dados de entrada

- **Fonte**: candles do BTC na Binance.
- **Intervalo**: `5m` (configurado em `docker-compose.yml` via `BINANCE_INTERVAL=5m`).

### 4.2 Features (X) e Targets (y)

As **features** (entradas do modelo) são calculadas a partir de candles reais e ficam em `api/ml/features.py`:

- `close` (fechamento)
- `ret` (retorno percentual do close)
- `acc` (diferença do retorno = “aceleração”)
- `amp` (amplitude do candle = high - low)
- `vol_rel` (volume relativo = volume / média móvel)

Os **targets** (o que o modelo aprende a prever) são do **próximo candle** (t→t+1):

- **Regressão** (`reg`): `open_next`, `high_next`, `low_next`, `close_next`, `amp_next`
- **Classificação** (`cls`): `dir_next = 1 se close_next > close_atual, senão 0`

### 4.3 Sequências (SEQ_LEN)

O LSTM recebe uma **janela** de tamanho `SEQ_LEN` (padrão 48) e prevê o próximo passo:

- Entrada do modelo: matriz \(48 \times n\_features\)
- Interpretação: como o candle é de 5 minutos, `SEQ_LEN=48` ≈ **4 horas** de histórico para prever o próximo candle de 5 minutos.

Isso é gerado em `api/ml/lstm_dataset.py`.

### 4.4 Split temporal (treino/validação)

O split é **temporal** (não embaralha):

- começo da série → treino
- final da série → validação

Isso simula melhor o “mundo real” (passado → futuro).

### 4.5 Normalização (scalers)

Redes neurais treinam melhor com valores em escalas parecidas.

Em `api/services/training_service.py`:

- `scaler_x` (MinMaxScaler) é ajustado **apenas no treino** e aplicado em treino/validação.
- `scaler_y` (MinMaxScaler) é ajustado **apenas no treino** para os targets de regressão.

### 4.6 Treino “multi-tarefa”

Em `api/ml/lstm_model.py` o modelo é “um tronco + duas cabeças”:

- **Tronco**: LSTM + camadas densas
- **Cabeça reg**: 5 saídas contínuas (valores do próximo candle)
- **Cabeça cls**: 1 saída sigmoide (probabilidade de alta)

### 4.7 EarlyStopping e checkpoint (quantas épocas rodar)

Em `api/services/training_service.py`:

- **EarlyStopping** monitora `val_loss`:
  - se `val_loss` **não melhora** por `patience` épocas, o treino para
  - e restaura os **melhores pesos**
- **ModelCheckpoint** salva em disco o **melhor modelo** (menor `val_loss`)

Isso define automaticamente “quantas épocas” o modelo realmente roda (ele para antes do máximo se não estiver melhorando).

### 4.8 Artefatos salvos (para inferência)

Ao final do treino (em volume/disco):

- `api/models/lstm_model.keras`: modelo treinado
- `api/models/lstm_bundle.joblib`: *bundle* com:
  - `scaler_x`, `scaler_y`
  - `feature_cols`, `target_reg_cols`
  - `seq_len`
  - path do modelo

Isso permite que inferência use **a mesma preparação** do treino.

---

## 5. Arquitetura

### Componentes

- **Site ASP.NET Core 9.0** (`site/`)
  - UI (docs + gráficos)
  - **Quartz.NET** para orquestrar as etapas (ingest, treino, drift, backfill)
- **API Python (FastAPI)** (`api/`)
  - ingestão (Binance → Postgres)
  - treino (Postgres → LSTM → disco)
  - inferência/series para gráficos
  - observabilidade (Prometheus)
- **PostgreSQL**
  - armazena candles, logs e séries materializadas
- **Disco/Volume (models)**
  - armazena modelo `.keras` e o bundle `.joblib`

### Frequências

- **Ingest**: job periódico (no projeto, o padrão é “a cada 5 minutos”).
- **Treino**:
  - **1x ao dia** (job `TrainDailyJob`)
  - **+ checagem de drift** (job `TrainDriftJob`, padrão a cada 1 hora) que só treina se MAPE(rolling) em `futures` passar do limiar e o tempo mínimo desde o último treino já tiver passado.
- **Backfill**: roda **uma vez** quando a aplicação sobe (se detectar baixa cobertura de dados).

---

## 6. Como o código está organizado

### API (Python)

- **Config**: `api/core/config.py` (+ `api/train_policy.json`)
- **Extração/transformação de dados ML**:
  - `api/ml/features.py`: features e targets
  - `api/ml/lstm_dataset.py`: janelas/sequências (SEQ_LEN)
  - `api/ml/lstm_model.py`: arquitetura do modelo LSTM
- **Treino e persistência do modelo**:
  - `api/services/training_service.py`: treino, métricas, saving em disco, log em `job_logs`
- **Carregamento do bundle para inferência**:
  - `api/services/lstm_bundle_service.py`: carrega `lstm_model.keras` + `lstm_bundle.joblib`
- **Séries para gráficos**:
  - `api/services/prediction_service.py`: série on-demand
  - `api/services/series_cache_service.py`: materialização em `series_cache` (batch predict)
- **Futuros (prospectivo)**:
  - `api/services/futures_service.py`: mantém tabela `futures` e calcula previsões “T-1 → T”
- **Rotas FastAPI**: `api/routers/*` (ingest, train, series, futures, metrics, obs)

### Site (ASP.NET Core)

- **Jobs Quartz.NET**: `site/Jobs/*`
  - `IngestJob.cs`: chama ingestão periódica na API
  - `TrainDailyJob.cs`: dispara treino diário + rebuild + update de futures
  - `TrainDriftJob.cs`: calcula MAPE rolling via `futures` e dispara retreino se necessário
  - `BackfillJob.cs`: preenche histórico inicial e dispara um treino completo quando necessário
- **Tela de gráficos**:
  - `site/Views/Charts/Index.cshtml`: consome as séries e mostra um **card visual** com métricas do modelo

---

## 7. Persistência (banco e disco)

### 7.1 PostgreSQL

- **`btc_candles`**
  - `time` (PK), `open`, `high`, `low`, `close`, `volume`
  - Definida em `api/schema.sql`

- **`job_logs`**
  - `id` (PK), `job_name`, `status`, `message`, `started_at`, `finished_at`
  - Definida em `api/schema.sql`

- **`series_cache`** (materialização para gráficos)
  - `time` (PK)
  - dados reais: `open`, `high`, `low`, `close`, `volume`
  - previsões: `pred_open_next`, `pred_high_next`, `pred_low_next`, `pred_close_next`, `pred_amp_next`
  - classificação: `cls_dir_next`, `prob_up`, `prob_down`
  - erros: `err_close_abs`, `err_close_signed`, `err_amp_abs`
  - Criada automaticamente por `api/services/series_cache_service.py`

- **`futures`** (série prospectiva)
  - `time` (PK), `pred_close`, `real_close`, `err_close`
  - Criada automaticamente por `api/services/futures_service.py`

### 7.2 Disco/volume

No Docker Compose, a pasta local `./api/models` é montada dentro do container Python em `/app/models`.

Arquivos principais:

- `api/models/lstm_model.keras`
- `api/models/lstm_bundle.joblib`

---

## 8. Limitações do modelo (para este problema)

- **Horizonte curto**: o modelo prevê apenas **o próximo candle** (5 minutos). Isso não significa que ele “entende” movimentos de longo prazo.
- **Sinal fraco x ruído**: mercados têm muito ruído; métrica boa em validação curta pode não se manter em produção.
- **Direção é difícil**: mesmo com acerto bom em `close_next`, acertar “sobe/desce” consistentemente pode ser instável.
- **Não inclui custos/execução**: o projeto não modela *slippage*, taxa, spread, liquidez nem estratégia; então **não dá para concluir** “compra/venda” só pelo modelo.
- **Risco de overfitting**: LSTM é mais expressivo que modelos lineares; sem controle (regularização, dados suficientes, validação correta) pode aprender padrões espúrios.
- **Retreino não é “garantia”**: retreinar por drift pode ajudar, mas também pode piorar se o período recente estiver anômalo.

---

## 9 Conclusões

### 9.1 Métricas do último treino

Obtidas via `GET /metrics` na API:

- **MAPE (validação, close_next)**: **0.14%**
- **MAE (close_next)**: **131.7572**
- **RMSE (close_next)**: **170.0316**
- **Épocas rodadas (EarlyStopping)**: **23**
- **val_loss (melhor)**: **0.693365**
- **Amostras**: **1879** (split **1503/1879**)
- **Início da validação**: **2026-01-07T02:25:00**

Interpretação rápida:

- MAPE baixo sugere boa aproximação para **t→t+1** (curto prazo), mas isso **não implica** utilidade direta para trade sem considerar risco/custos.

### 9.2 Drift (MAPE rolling em `futures`)

Calculando MAPE rolling nos dados atuais de `futures` (mesma ideia do job de drift):

- **MAPE rolling**: **0.7388%** (n=95 pontos válidos)

Isso é o tipo de número que o `TrainDriftJob` compara com o limiar configurado.

### 9.3 Dados para gráficos

Na série materializada `series_cache`, na janela **2026-01-07 → 2026-01-08**:

- **396 pontos** retornados pela API (suficiente para gráficos do período).

### 9.4 Observabilidade

- O endpoint Prometheus (`/obs/metrics`) está ativo, então dá para acompanhar volume/latência HTTP (útil para detectar gargalos quando /series/rebuild roda).

---

## 10. Melhorias recomendadas (próximos passos)

- **Retenção/purga de dados**: criar um mecanismo explícito para apagar dados antigos por data (candles, cache, futures, logs) para manter banco leve.
- **Métricas de classificação**: além de MAPE/MAE/RMSE (regressão), acompanhar acurácia/F1/ROC-AUC do `dir_next` e calibrar probabilidade.
- **Separar datasets/validação**: ampliar validação (ex.: *walk-forward validation*) para reduzir chance de “validar bem por sorte”.
- **Monitoramento de recursos**: incluir CPU/memória (ex.: via `psutil` + Prometheus) e alertas (latência alta, OOM).
- **Controles de risco (fora do modelo)**: se o objetivo for decisão de compra/venda, incluir camada de estratégia, custos e *backtest*.

