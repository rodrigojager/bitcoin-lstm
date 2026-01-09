import time, requests, pandas as pd
from datetime import datetime, timedelta, timezone
from core.config import settings
from core.db import pg_conn
from core.logging import log_job

def fetch_binance_klines(symbol=None, interval=None, limit=None) -> pd.DataFrame:
    symbol = symbol or settings.BINANCE_SYMBOL
    interval = interval or settings.BINANCE_INTERVAL
    limit = limit or settings.BINANCE_LIMIT
    url = f"{settings.BINANCE_BASE}/api/v3/klines"
    headers = {}
    r = requests.get(url, params={"symbol":symbol,"interval":interval,"limit":limit}, headers=headers, timeout=30)
    r.raise_for_status()
    data = r.json()
    return normalize_klines_payload(data)

def upsert_candles(df: pd.DataFrame) -> int:
    sql = """INSERT INTO btc_candles (time, open, high, low, close, volume)
             VALUES (%s,%s,%s,%s,%s,%s)
             ON CONFLICT (time) DO NOTHING;"""
    rows = list(df.itertuples(index=False, name=None))
    with pg_conn() as conn:
        with conn.cursor() as cur:
            cur.executemany(sql, rows)
            inserted = cur.rowcount
    return inserted

def normalize_klines_payload(data: list) -> pd.DataFrame:
    cols = ["open_time","open","high","low","close","volume","close_time",
            "quote_asset_volume","trades","taker_buy_base","taker_buy_quote","ignore"]
    df = pd.DataFrame(data, columns=cols)
    for c in ["open","high","low","close","volume"]:
        df[c] = df[c].astype(float)
    df["time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True).dt.tz_convert(None)
    df = df[["time","open","high","low","close","volume"]].sort_values("time").reset_index(drop=True)
    return df

def interval_to_ms(interval: str) -> int:
    unit = interval[-1]; val = int(interval[:-1])
    return {"m":60_000, "h":3_600_000, "d":86_400_000, "w":7*86_400_000}[unit]*val

def fetch_klines_window(symbol: str, interval: str, start_ms: int, limit: int=1000, api_key: str|None=None):
    url = f"{settings.BINANCE_BASE}/api/v3/klines"
    headers = {"X-MBX-APIKEY": api_key} if api_key else {}
    resp = requests.get(url, params={"symbol":symbol,"interval":interval,"limit":limit,"startTime":start_ms},
                        headers=headers, timeout=30)
    if resp.status_code in (418,429):
        time.sleep(2.0)
        resp = requests.get(url, params={"symbol":symbol,"interval":interval,"limit":limit,"startTime":start_ms},
                            headers=headers, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    if not data: return [], None
    return data, data[-1][0]

def backfill_job(days: int|None=None, symbol: str|None=None, interval: str|None=None,
                 sleep_ms: int|None=None, limit: int=1000):
    start_ts = datetime.utcnow()
    try:
        days = days or settings.BACKFILL_DAYS
        symbol = symbol or settings.BINANCE_SYMBOL
        interval = interval or settings.BINANCE_INTERVAL
        sleep_ms = sleep_ms if sleep_ms is not None else settings.BACKFILL_SLEEP_MS

        now_ms = int(datetime.now(timezone.utc).timestamp()*1000)
        start_ms = now_ms - days*86_400_000
        interval_ms = interval_to_ms(interval)
        total_fetched = total_inserted = loops = 0
        current_ms = start_ms
        while True:
            data, last_open = fetch_klines_window(symbol, interval, current_ms, limit=limit, api_key=settings.BINANCE_API_KEY)
            loops += 1
            if not data: break
            df = normalize_klines_payload(data)
            total_inserted += upsert_candles(df)
            total_fetched += len(df)
            current_ms = last_open + interval_ms
            if current_ms >= now_ms: break
            time.sleep(sleep_ms/1000.0)

        msg = f"Backfill {symbol} {interval} {days}d: fetched={total_fetched}, inserted={total_inserted}, calls={loops}"
        log_job("backfill","ok",msg,start_ts,datetime.utcnow())
        return {"status":"ok","fetched":total_fetched,"inserted":total_inserted,"calls":loops,"days":days}
    except Exception as e:
        log_job("backfill","error",str(e),start_ts,datetime.utcnow())
        return {"status":"error","message":str(e)}
