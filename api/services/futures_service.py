from datetime import datetime
from typing import Iterable, List, Optional
import pandas as pd
from core.db import pg_conn
from ml.features import build_features_targets
from services.lstm_bundle_service import load_bundle


def ensure_table():
    conn = pg_conn()
    old_autocommit = conn.autocommit
    try:
        conn.autocommit = True
        with conn.cursor() as cur:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS futures (
                  time        TIMESTAMP PRIMARY KEY,
                  pred_close  NUMERIC,
                  real_close  NUMERIC,
                  err_close   NUMERIC
                );
                """
            )
    finally:
        conn.autocommit = old_autocommit
        conn.close()


def save_predictions_for_times(times: Iterable[datetime]):
    """Para cada time em 'times', calcula a previsão de close_next baseada no candle anterior
    e insere (pred, real, erro) em 'futuros'. Ignora tempos já existentes.
    """
    times = sorted(set([t if isinstance(t, datetime) else pd.to_datetime(t).to_pydatetime() for t in times]))
    if not times:
        return 0
    ensure_table()
    min_time = min(times)
    with pg_conn() as conn:
        df = pd.read_sql(
            """
            SELECT time, open, high, low, close, volume
            FROM btc_candles
            WHERE time >= %s - INTERVAL '3 days'
            ORDER BY time
            """,
            conn,
            params=(min_time,),
        )
    if df.empty or len(df) < 3:
        return 0
    # Monta features com dropna (remove o último da janela consultada, mantendo pares prev->next)
    df2, X, Yreg, _ = build_features_targets(df)
    # Mapa: time_next -> idx_prev (features em T-1 geram target em T)
    next_to_prev = {}
    for i in range(len(df2)-1):
        T_next = df2.iloc[i+1]["time"]
        next_to_prev[T_next] = i
    bundle = load_bundle()
    seq_len = int(bundle.seq_len)
    close_idx = bundle.target_reg_cols.index("close_next")
    # Construir inserts apenas quando houver par (T-1, T)
    inserts: List[tuple] = []
    for T in times:
        T = pd.to_datetime(T).to_pydatetime()
        idx_prev = next_to_prev.get(pd.Timestamp(T))
        if idx_prev is None:
            continue
        # Sem janela suficiente, não conseguimos prever
        if idx_prev < seq_len - 1:
            continue
        idx_next = idx_prev + 1
        win = X.iloc[idx_prev - seq_len + 1 : idx_prev + 1][bundle.feature_cols].to_numpy(dtype="float32")
        win_scaled = bundle.scaler_x.transform(win).reshape((1, seq_len, len(bundle.feature_cols)))
        p = bundle.model.predict(win_scaled, verbose=0)
        reg = bundle.scaler_y.inverse_transform(p["reg"])[0]
        pred_close = float(reg[close_idx])
        real_close = float(df2.iloc[idx_next]["close"])
        err = abs(pred_close - real_close)
        inserts.append((T, pred_close, real_close, err))
    if not inserts:
        return 0
    with pg_conn() as conn:
        with conn.cursor() as cur:
            cur.executemany(
                """
                INSERT INTO futures(time, pred_close, real_close, err_close)
                VALUES (%s,%s,%s,%s)
                ON CONFLICT (time) DO UPDATE SET
                  pred_close = EXCLUDED.pred_close,
                  real_close = EXCLUDED.real_close,
                  err_close = EXCLUDED.err_close
                """,
                inserts,
            )
            return cur.rowcount


def load_futuros_series(start: Optional[str], end: Optional[str], limit: Optional[int] = None):
    ensure_table()
    params = []
    where = []
    if start and end:
        where.append("time BETWEEN %s AND %s")
        params.extend([start, end])
    query = "SELECT time, pred_close, real_close, err_close FROM futures"
    if where:
        query += " WHERE " + " AND ".join(where)
    if limit is not None:
        # para pegar os últimos N pontos sem varrer tudo, ordena DESC, limita e reordena em memória
        query += " ORDER BY time DESC LIMIT %s"
        params.append(int(limit))
    else:
        query += " ORDER BY time"
    with pg_conn() as conn:
        df = pd.read_sql(query, conn, params=tuple(params))
    if df.empty:
        return {"points": []}
    if limit is not None:
        df = df.sort_values("time")
    # Sanitiza NaN/Inf para None para compatibilidade com JSON
    def f(x):
        try:
            import math
            if x is None:
                return None
            xv = float(x)
            if math.isnan(xv) or math.isinf(xv):
                return None
            return xv
        except Exception:
            return None
    points = []
    for _, r in df.iterrows():
        points.append({
            "time": pd.to_datetime(r["time"]).isoformat(),
            "pred_close": f(r.get("pred_close")),
            "real_close": f(r.get("real_close")),
            "err_close": f(r.get("err_close")),
        })
    return {"points": points}
