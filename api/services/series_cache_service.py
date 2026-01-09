from typing import Optional, List, Tuple
import pandas as pd
import numpy as np
from core.db import pg_conn
from core.config import settings
from ml.features import build_features_targets, TARGET_REG_COLS
from ml.lstm_dataset import build_x_sequences
from services.lstm_bundle_service import load_bundle


def ensure_table() -> None:
    """Create cached series table if not exists (materialized series for charts)."""
    conn = pg_conn()
    old_autocommit = conn.autocommit
    try:
        conn.autocommit = True
        with conn.cursor() as cur:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS series_cache (
                  time                TIMESTAMP PRIMARY KEY,
                  open                NUMERIC,
                  high                NUMERIC,
                  low                 NUMERIC,
                  close               NUMERIC,
                  volume              NUMERIC,
                  pred_open_next      NUMERIC,
                  pred_high_next      NUMERIC,
                  pred_low_next       NUMERIC,
                  pred_close_next     NUMERIC,
                  pred_amp_next       NUMERIC,
                  cls_dir_next        INTEGER,
                  prob_up             NUMERIC,
                  prob_down           NUMERIC,
                  err_close_abs       NUMERIC,
                  err_close_signed    NUMERIC,
                  err_amp_abs         NUMERIC
                );
                """
            )
    finally:
        conn.autocommit = old_autocommit
        conn.close()


def _predict_lstm_for_series(X: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Retorna (reg_pred_df, cls_pred, prob_2col) alinhados por índice de X."""
    bundle = load_bundle()
    seq_len = int(bundle.seq_len)
    n = len(X)
    reg_pred = np.full((n, len(TARGET_REG_COLS)), np.nan, dtype="float32")
    prob_up = np.full((n,), np.nan, dtype="float32")

    # Batch predict: muito mais rápido que chamar predict() ponto-a-ponto
    X_feat = X[bundle.feature_cols].copy()
    X_seq, idx_orig = build_x_sequences(X_feat, seq_len=seq_len)
    X2d = X_seq.reshape((X_seq.shape[0] * X_seq.shape[1], X_seq.shape[2]))
    X_scaled = bundle.scaler_x.transform(X2d).reshape(X_seq.shape).astype("float32")

    p = bundle.model.predict(X_scaled, verbose=0, batch_size=512)
    reg_all = bundle.scaler_y.inverse_transform(p["reg"]).astype("float32")
    cls_all = p["cls"].reshape((-1,)).astype("float32")

    reg_pred[idx_orig, :] = reg_all
    prob_up[idx_orig] = cls_all

    reg_df = pd.DataFrame(reg_pred, columns=TARGET_REG_COLS, index=X.index)
    cls_pred = (prob_up >= 0.5).astype(int)
    prob = np.vstack([1.0 - prob_up, prob_up]).T
    return reg_df, cls_pred, prob


def build_series_cache(days: Optional[int] = None) -> int:
    """Recalcula a série utilizada pelos gráficos e materializa na tabela series_cache.
    Retorna número de linhas upsertadas.
    """
    ensure_table()
    days = days or settings.LOOKBACK_DAYS
    with pg_conn() as conn:
        df = pd.read_sql(
            """
            SELECT time, open, high, low, close, volume
            FROM btc_candles
            WHERE time >= NOW() - %s::interval
            ORDER BY time
            """,
            conn,
            params=(f"{days} days",),
        )
    if df.empty or len(df) < 3:
        return 0

    df2, X, Yreg, Ycls = build_features_targets(df)
    try:
        reg_pred, cls_pred, prob = _predict_lstm_for_series(X)
    except Exception:
        # se modelos não existirem ainda, materializa somente o real
        reg_pred = cls_pred = prob = None

    # construir linhas alinhadas i (real em i) com pred de i (para i+1) e erros conforme real i+1
    import math
    def safe_float(x):
        try:
            xv = float(x)
            if math.isnan(xv) or math.isinf(xv):
                return None
            return xv
        except Exception:
            return None
    inserts: List[Tuple] = []
    for i in range(len(df2)):
        real = df2.iloc[i]
        time_i = pd.to_datetime(real["time"]).to_pydatetime()
        open_i  = safe_float(real["open"]) 
        high_i  = safe_float(real["high"]) 
        low_i   = safe_float(real["low"]) 
        close_i = safe_float(real["close"]) 
        vol_i   = safe_float(real["volume"]) 

        pred_open = pred_high = pred_low = pred_close = pred_amp = None
        cls_dir = None
        p_up = p_down = None
        err_close_abs = err_close_signed = err_amp_abs = None

        if reg_pred is not None:
            pred_open  = safe_float(reg_pred.iloc[i]["open_next"]) if i < len(reg_pred) else None
            pred_high  = safe_float(reg_pred.iloc[i]["high_next"]) if i < len(reg_pred) else None
            pred_low   = safe_float(reg_pred.iloc[i]["low_next"])  if i < len(reg_pred) else None
            pred_close = safe_float(reg_pred.iloc[i]["close_next"]) if i < len(reg_pred) else None
            pred_amp   = safe_float(reg_pred.iloc[i]["amp_next"]) if i < len(reg_pred) else None
        if cls_pred is not None:
            cls_dir = int(cls_pred[i]) if i < len(cls_pred) else None
            if prob is not None and i < len(prob):
                p_down = safe_float(prob[i][0])
                p_up   = safe_float(prob[i][1])

        if pred_close is not None and i + 1 < len(df2):
            real_next_close = safe_float(df2.iloc[i + 1]["close"]) 
            real_next_amp = safe_float(df2.iloc[i + 1]["high"] - df2.iloc[i + 1]["low"]) 
            if real_next_close is not None:
                err_close_abs = safe_float(abs(pred_close - real_next_close))
                err_close_signed = safe_float(pred_close - real_next_close)
            if pred_amp is not None and real_next_amp is not None:
                err_amp_abs = safe_float(abs(pred_amp - real_next_amp))

        inserts.append(
            (
                time_i,
                open_i, high_i, low_i, close_i, vol_i,
                pred_open, pred_high, pred_low, pred_close, pred_amp,
                cls_dir, p_up, p_down,
                err_close_abs, err_close_signed, err_amp_abs,
            )
        )

    if not inserts:
        return 0

    with pg_conn() as conn:
        with conn.cursor() as cur:
            cur.executemany(
                """
                INSERT INTO series_cache(
                  time, open, high, low, close, volume,
                  pred_open_next, pred_high_next, pred_low_next, pred_close_next, pred_amp_next,
                  cls_dir_next, prob_up, prob_down,
                  err_close_abs, err_close_signed, err_amp_abs
                ) VALUES (
                  %s,%s,%s,%s,%s,%s,
                  %s,%s,%s,%s,%s,
                  %s,%s,%s,
                  %s,%s,%s
                )
                ON CONFLICT (time) DO UPDATE SET
                  open = EXCLUDED.open,
                  high = EXCLUDED.high,
                  low  = EXCLUDED.low,
                  close= EXCLUDED.close,
                  volume= EXCLUDED.volume,
                  pred_open_next = EXCLUDED.pred_open_next,
                  pred_high_next = EXCLUDED.pred_high_next,
                  pred_low_next  = EXCLUDED.pred_low_next,
                  pred_close_next= EXCLUDED.pred_close_next,
                  pred_amp_next  = EXCLUDED.pred_amp_next,
                  cls_dir_next   = EXCLUDED.cls_dir_next,
                  prob_up        = EXCLUDED.prob_up,
                  prob_down      = EXCLUDED.prob_down,
                  err_close_abs  = EXCLUDED.err_close_abs,
                  err_close_signed=EXCLUDED.err_close_signed,
                  err_amp_abs    = EXCLUDED.err_amp_abs
                """,
                inserts,
            )
            return cur.rowcount


def load_series_cached(start: Optional[str], end: Optional[str], fallback_days: int = 90):
    ensure_table()
    params = []
    where = []
    if start and end:
        where.append("time BETWEEN %s AND %s")
        params.extend([start, end])
    else:
        where.append("time >= NOW() - %s::interval")
        params.append(f"{fallback_days} days")
    q = """
        SELECT time, open, high, low, close, volume,
               pred_open_next, pred_high_next, pred_low_next, pred_close_next, pred_amp_next,
               cls_dir_next, prob_up, prob_down,
               err_close_abs, err_close_signed, err_amp_abs
        FROM series_cache
    """
    if where:
        q += " WHERE " + " AND ".join(where)
    q += " ORDER BY time"

    with pg_conn() as conn:
        df = pd.read_sql(q, conn, params=tuple(params))
    if df.empty:
        return {"points": []}

    import math
    def safe(x):
        if pd.isna(x):
            return None
        try:
            v = float(x)
            if math.isnan(v) or math.isinf(v):
                return None
            return v
        except:
            return None
    
    points = []
    for _, r in df.iterrows():
        real = {
            "time": pd.to_datetime(r["time"]).isoformat(),
            "open": safe(r["open"]),
            "high": safe(r["high"]),
            "low": safe(r["low"]),
            "close": safe(r["close"]),
            "volume": safe(r["volume"]),
        }
        pred = None
        pcn = safe(r["pred_close_next"])
        if pcn is not None:
            pred = {
                "open_next": safe(r["pred_open_next"]),
                "high_next": safe(r["pred_high_next"]),
                "low_next": safe(r["pred_low_next"]),
                "close_next": pcn,
                "amp_next": safe(r["pred_amp_next"]),
            }
        cls = None
        cdn = safe(r["cls_dir_next"])
        if cdn is not None:
            cls = {
                "dir_next": int(cdn),
                "prob_up": safe(r["prob_up"]),
                "prob_down": safe(r["prob_down"]),
            }
        err = None
        eca = safe(r["err_close_abs"])
        if eca is not None:
            err = {
                "close_abs": eca,
                "close_signed": safe(r["err_close_signed"]),
                "amp_abs": safe(r["err_amp_abs"]),
            }
        points.append({"real": real, "pred": pred, "cls": cls, "err": err})
    return {"points": points}
