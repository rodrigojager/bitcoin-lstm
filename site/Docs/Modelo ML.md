# Modelo de Previsão de Preço (BTC)

## 1) Lógica / Funcionamento do modelo

Este projeto treina um modelo de aprendizado de máquina para prever o próximo "candle" (próxima amostra de tempo) do Bitcoin em janelas de 5 minutos. Um candle contém: preço de abertura (open), máxima (high), mínima (low), fechamento (close) e volume negociado no período.

- Onde o código faz isso:
  - Extração e ingestão de dados: `api/routers/ingest.py` + `api/services/ingestion_service.py`
  - Construção de variáveis (features) e alvos (targets): `api/ml/features.py`
  - Treino: `api/services/training_service.py`
  - Predição em série e métricas: `api/services/prediction_service.py`, `api/routers/metrics.py`
- Persistência de previsões prospectivas (`futures`): `api/services/futures_service.py`

### 1.1 Como as variáveis são formadas (features)
O arquivo `api/ml/features.py` cria as seguintes colunas a partir de OHLCV:
- `close`: o preço de fechamento atual. É usado como base do estado atual.
- `ret` (retorno): variação percentual do fechamento em relação ao candle anterior. Fórmula: `ret = close[t] / close[t-1] - 1`.
- `acc` (aceleração do retorno): variação do retorno entre dois candles. Fórmula: `acc = ret[t] - ret[t-1]`.
- `amp` (amplitude): diferença entre máxima e mínima do candle. Fórmula: `amp = high - low`. Dá uma noção de volatilidade local.
- `vol_rel` (volume relativo): volume atual dividido pela média móvel de 10 períodos do volume. Normaliza o volume.

Essas variáveis são usadas como "entradas" (X) do modelo.

### 1.2 O que o modelo prevê (targets)
Também em `api/ml/features.py`, definimos como alvos (Y) o próximo candle (t+1):
- `open_next`, `high_next`, `low_next`, `close_next`, `amp_next`.
- Adicionalmente, criamos `dir_next` para classificar a direção do fechamento: 1 (subiu) quando close[t+1] > close[t], 0/−1 conforme o caso.

Na prática, treinamos:
- Um conjunto de regressões (uma por alvo) para prever valores contínuos de t+1.
- Um classificador para direção (subiu/baixou) em t+1.

### 1.3 Como o treino é feito
O treino ocorre em `api/services/training_service.py`:
- Seleciona uma janela temporal de N dias (configurável por `LOOKBACK_DAYS`).
- Constrói features e alvos com `build_features_targets`.
- Faz "split" temporal: ~80% primeiras amostras para treino, ~20% finais para validação (fora do treino).
- Monta sequências (janelas) e treina um **modelo LSTM (TensorFlow/Keras)** multi-tarefa:
  - Head de regressão multi-saída (open/high/low/close/amp do t+1)
  - Head de classificação para direção (prob_up/prob_down do t+1)
- Calcula métricas no conjunto de validação (out-of-sample): MAE (erro absoluto), MAPE (erro percentual), SMAPE (erro percentual simétrico) para `close_next`.
- Salva o modelo Keras e um bundle com scalers/metadados para uso posterior pela API.

Observação: em produção, o treino é orquestrado pelo Quartz (.NET) em dois jobs:

- **Treino diário fixo**: chama `POST /train` e `POST /series/rebuild`.
- **Checagem de drift**: calcula MAPE(rolling) em `futures` e dispara o treino apenas se ultrapassar um limiar após um tempo mínimo.

Os parâmetros ficam em `site/appsettings.json` (seção `PythonApi`).

### 1.4 Como as previsões são usadas na interface
- A página de gráficos (`site/Views/Charts/Index.cshtml`) consulta:
  - `/series` — série com candles e previsões (t→t+1) ao longo da janela.
  - `/metrics` — métricas de validação e início do período de teste (o trecho mais novo, sombreado na UI).
  - `/futures` — série prospectiva com previsões feitas em tempo t−1 e comparadas ao real em t (sem retropreencher o passado). Essa série reflete o que teria acontecido no tempo.
- Gráficos exibidos:
  - Velas reais × previstas (t→t+1), close real × previsto e erro%.
  - Região sombreada indica a parte "de teste" (validação) do conjunto mais recente.
  - Futuros: close previsto × real × erro% prospectivos.
  - Direção (sobreposta): direção prevista × direção real e erro (0/1), com acurácia e métricas ponderadas.

## 2) Potencial do modelo (o que esperar e limitações)
- Com apenas OHLCV (sem livro de ordens, notícias, macroeventos), conseguimos capturar padrões básicos:
  - Relação volume–volatilidade (candles mais “largos” tendem a vir com mais volume).
  - Persistência/“clusters” de volatilidade (períodos mais voláteis em sequência).
  - Sazonalidades intradiárias (padrões de horário) e efeitos de curto prazo.
- Para horizonte de 5 minutos, o ruído de mercado é alto. Prever o valor exato do próximo fechamento (`close_next`) pode ter erro percentual baixo (MAPE sub-1%), mas prever a direção bruta muitas vezes é mais difícil (hit rate próximo de 50%).
- O modelo atua como um "estimador de tendência curta e nível", útil para visualizar cenários e monitorar mudanças abruptas, mas não substitui sinal com edges estáveis sem filtragem/custos.

Limitações estruturais atuais:
- Sem dados de livro de ordens (order book), fluxo de ordens ou latência de execução.
- Sem features de notícias/eventos/volatilidade implícita.
- Sem filtragem por custos de transação/slippage.
- Validação feita por split temporal simples (ainda não implementamos walk-forward na métrica de exposição pública).

### 2.1 Como o livro de ordens pode ajudar
- Profundidade e spread: spreads (diferença entre o preço do ask e o bid) estreitos e boa profundidade no lado comprador reduzem o custo de subir preço; spreads abertos e pouca profundidade no bid tornam quedas mais prováveis.
- Desequilíbrio (order imbalance): se o somatório de volumes no bid (oferta de compra) ≫ ask (oferta de venda) (em múltiplos níveis), há maior “pressão” de compra; o inverso sinaliza pressão de venda.
- Dinâmica de fila (queue/priority): cancelamentos massivos do lado comprador ou “spoofing” podem anteceder micro-movimentos.
- Impacto esperado: features como spread, mid-price, profundidade agregada por nível, e um índice de desequilíbrio (ex.: (BidVol−AskVol)/(BidVol+AskVol)) tendem a melhorar previsão de direção/magnitude em curtos prazos.

### 2.2 Exemplos de notícias e macroeventos (e direção esperada, ceteris paribus)
- Aprovação de ETF spot de BTC ou grandes influxos em ETFs: aumenta demanda institucional → viés altista.
- Hack/pausa de exchange relevante: risco sistêmico/saída de liquidez → viés baixista.
- Endurecimento regulatório (ex.: ação da SEC contra grandes players): incerteza regulatória → viés baixista.
- CPI/inflação acima do esperado (hawkish): aperto de condições financeiras → risco-off → viés baixista para ativos de risco.
- Queda global nos dividendos de ações ou taxação extra em ações: realocação para alternativas como BTC → possível viés altista (dependente do contexto).
- Halving/atualizações de rede: redução de emissão/melhorias → narrativa altista (efeito depende de precificação prévia).

### 2.3 Volume, volatilidade e impacto no preço
- Volume–volatilidade: positivamente correlacionados; picos de volume geralmente vêm com maior range (`amp`).
- Volume e direção: “mais volume” por si só não garante alta/baixa; o sinal depende do sentido do fluxo (agressão ao ask vs. ao bid). Volume alto com agressão compradora tende a elevar o close; com agressão vendedora, a reduzi-lo.

### 2.4 Custos de transação e slippage
- Custos: taxas, funding, spreads. A previsão só é “acionável” se |E[ret]| > custos + margem.
- Slippage: diferença entre preço teórico (top-of-book) e preço efetivo de execução, causada por impacto de mercado e liquidez disponível. Aumenta com pressa (ordem a mercado), tamanho e baixa profundidade.
- Relação com o previsto: ambientes de spread largo/baixa profundidade (alto slippage) exigem maior magnitude prevista; em alta liquidez/spread estreito, sinais menores podem ser viáveis.

### 2.5 Split temporal simples vs. walk-forward
- Split temporal simples: separar um bloco inicial (treino) e um bloco final (validação) respeitando a ordem do tempo (sem embaralhar). É o que usamos hoje (~80/20).
- Walk-forward: treinos repetidos em janelas móveis (ex.: treina [t0,t1], valida [t1,t2]; move a janela e repete). Mede estabilidade ao longo do tempo e reduz risco de “acertar” um período específico; útil para seleção de hiperparâmetros e early stopping.

## 3) Avaliação da performance do modelo

### 3.1 Métricas de validação (conjunto fora do treino)
- As métricas exibidas na UI vêm de `/metrics` e refletem o split temporal do último treino. Exemplo recente (ilustrativo):
  - MAE ≈ 535 USD
  - MAPE ≈ 0,49%
  - SMAPE ≈ 0,49%
Isso indica um erro percentual médio abaixo de 1% para `close_next` no período de validação — bom para horizonte de 5m.

### 3.2 Métricas prospectivas na tabela `futures`
Comandos SQL (já executados no ambiente) sintetizam as métricas direcional e ponderada:
- Acurácia direcional (hit rate): ~45,36%
- WDA (Weighted Directional Accuracy): ~47,95%
- Erro ponderado (100 − WDA): ~52,05%
- |ret| médio quando acerta: ~0,029%
- |ret| médio quando erra: ~0,026%

Interpretação:
- Direção ~45%: pior que um baseline 50% se a base for balanceada, mas na prática pode haver muitos "estáveis" (0) — por isso complementamos com WDA.
- WDA ~48%: acertos ponderados por magnitude dos retornos. Estar abaixo de 50% sugere que, mesmo ponderando pelos movimentos, o sinal direcional simples não tem edge sustentável no período analisado.
- Magnitudes médias (|ret|) em acertos e erros próximas (0,029% vs 0,026%) indicam que, sem filtros (thresholds) e sem custos, o sinal bruto não se converteria facilmente em PnL consistente.

Nota sobre edge sustentável:
- Chamamos de “edge sustentável” quando a vantagem se mantém em janelas walk-forward (janela móvel), após custos e slippage, e com drawdown controlado. Os números atuais não caracterizam edge sustentável; servem mais como monitor de nível/tendência curta.

Qualitativamente:
- O modelo capta níveis (erro de close baixo em %) melhor do que direção bruta. Isso é comum em horizontes curtos com ruído.
- A utilidade operacional cresce quando aplicamos filtros de confiança/probabilidade e magnitude esperada do movimento.

## 4) Sugestões de melhorias

### 4.1 Melhorias de dados e features
- Incluir OHLCV lags e janelas múltiplas: retornos em 1/3/5/10 candles, ATR/volatilidade realizada, EMAs, RSI/MACD, bandas de Bollinger, amplitude normalizada (amp/close), volume bruto e relativo em várias janelas.
- Codificar hora-do-dia/dia-da-semana (efeitos intradiários/semanais).
- Se possível, integrar dados de ordem (book), desequilíbrio (order imbalance), fluxo de negócios (tick data) e eventos/notícias.

Detalhes dos indicadores citados:
- ATR (Average True Range) / volatilidade realizada: medem amplitude típica/variância dos retornos. Maior ATR indica maior risco e custo para mover preço; como feature, ajuda a diferenciar regimes (calmo vs. turbulento) e a ajustar stops/targets.
- EMA (Exponential Moving Average): média móvel exponencial (mais peso ao recente). Cruzes de EMAs (rápida × lenta) capturam momentum; como feature, sinaliza tendência local.
- RSI (Relative Strength Index): oscilador 0–100 baseado em ganhos/perdas recentes; extremos (70/30) sugerem sobrecompra/sobrevenda e potenciais reversões curtas.
- MACD (Moving Average Convergence/Divergence): diferença entre EMAs (rápida−lenta) e sua “signal line”; captura aceleração/desaceleração de tendência.
- Bandas de Bollinger: envelope em torno de uma média (ex.: SMA) a k desvios-padrão. Toques/saídas sinalizam compressões/expansões de volatilidade e possíveis reversões/continuações.

Order imbalance, fluxo e tick data:
- Tick data: dados por negócio/negociação (trade-by-trade) e por cotações (updates do book) — granularidade da microestrutura.
- Fluxo de negócios (order flow): direção e intensidade de agressões (compras a mercado vs. vendas a mercado). Aumento de agressão compradora, ceteris paribus, tende a elevar o close; agressão vendedora tende a reduzi-lo.
- Order imbalance (desequilíbrio): diferença relativa entre volumes no bid e no ask (em vários níveis). Aumento do desequilíbrio pró-bid tende a pressão altista; pró-ask, baixista. Features úteis: spread, mid-price, profundidade agregada (níveis 1–5), imbalance normalizado, taxa de cancelamentos/modificações.

### 4.2 Modelagem e validação
- Separar claramente:
  - Regressão (multi-saída) para prever o próximo candle e a magnitude (incluindo `close_next`).
  - Classificador calibrado para probabilidade de alta/baixa com banda neutra (|ret| abaixo de limiar).
- Walk-forward (janelas móveis) para avaliação mais robusta e seleção de hiperparâmetros com early stopping.
- Calibração das probabilidades (isotônica/Platt) e curvas de calibração.

Curvas de calibração e calibração de probabilidades:
- O que são: gráficos que comparam a probabilidade prevista (ex.: “prob. de alta”) com a frequência observada. Um classificador bem calibrado tem curva próxima da diagonal (p=0,7 → ~70% de altas).
- Como calibrar: Platt (sigmoide) ajusta uma função logística sobre as pontuações; Isotônica faz mapeamento monótono não paramétrico (mais flexível, exige mais dados).
- Por que melhora: decisões baseadas em probabilidade (ex.: operar só quando p_alta > 0,55 e |E[ret]| > custos) ficam consistentes; melhora o dimensionamento de posições e a gestão de risco. Avaliar com Brier score/ECE e curvas de confiabilidade.

### 4.3 Estratégia e decisão (day trading/trading)
- Definir limiar de entrada: operar apenas quando |E[ret]| > custos + margem ("banda morta").
- Dimensionamento (position sizing) por confiança e volatilidade: menor alavancagem em baixa confiança/alta volatilidade.
- Regras de gestão de risco: stop/target adaptativos pela volatilidade (ex.: múltiplos de ATR), limite de perdas/drawdown.
- Backtest simples de PnL: a partir das previsões, simular uma estratégia long/short/flat com custos, para verificar relevância econômica (não apenas acertos).

### 4.4 Visualização/monitoramento
- Exibir baseline(s) de comparação no gráfico de direção: “sempre estável”, “repete direção anterior”.
- Mostrar PnL acumulado de uma regra simples (com custos) ao lado de WDA e hit rate.
- Destacar zonas de baixa confiança (probabilidade próxima de 50% ou |ret| esperado pequeno) e evitar operar nelas.

## 5) Conclusão a respeito do modelo

O modelo tem um erro muito baixo para prever o preço de fechamento de cada vela de 5 minutos de duração, porém isso é algo relativamente fácil de obter, porque mesmo com um ativo tão volátil como o Bitcoin dificilmente sofre grandes alterações em menos de 5 minutos, é uma janela muito pequena para que o modelo consiga errar muito e logo esse modelo está longe de ser um bom indicador para tomada de decisão de compra e venda de Bitcoins em day trade. Ele foi feito com a API gratuita e aberta da binance, que fornece um conjunto muito limitado de informações/dados para um modelo mais robusto. Alguns deles estão disponíveis na API da Binance também gratuita, mas fechada (requer cadastro para obteção de API key e respectiva autenticação). Além disso a própria estratégia utilizada para esses dados que obtemos podem evoluir, mas isso acaba indo além da proposta que tem função acadêmica de demonstrar a obtenção e treinamento de um modelo de Machine Learning, que foi devidamente implementado e demonstrado usando um regressor e um classificador. Além disso, atividades econômicas, tanto a nível macro quanto a nível micro, principalmente em um cenário extremamente especulativo sofrem influências de diversos critérios subjetivos por parte de todos os players envolvidos, de forma que a própria análise estatística e respectiva atuação dela influenciam os comportamentos e consequentemente os dados da análise, sendo um problema extremamente complexo de maneira que pode ser difícil mesmo com modelos muito mais robustos obter sugestões de investimentos adequados. Por mais que o modelo precisaria a princípio e em teoria apenas ser melhor do que um humano sem esses recursos para mostrar sua validade, existem milhares de investidores que investiram tempo, dinheiro e tecnologia para fazer melhores previsões e mais rápidas, bem como movimentando um volume em escala muito maior do que a maioria dos entusiastas que se aventuram nesse meio, munidos de informações de ciência de dados e machine learning, sendo a própria participação desses players algo que influenciaria os modelos e seu estado caótico.

---

Glossário de siglas mais usadas: 
- OHLCV: Open (abertura), High (máxima), Low (mínima), Close (fechamento), Volume.
- MAPE/SMAPE/MAE: métricas de erro (percentual/percentual simétrico/absoluto) — quanto menores, melhor.
- WDA: acertos ponderados pela magnitude do retorno (avalia a importância econômica dos acertos).
- Candle: bloco de tempo (aqui 5 minutos) com preços e volume desse período.
