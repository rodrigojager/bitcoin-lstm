
CREATE TABLE IF NOT EXISTS btc_candles (
  time   TIMESTAMP PRIMARY KEY,
  open   NUMERIC NOT NULL,
  high   NUMERIC NOT NULL,
  low    NUMERIC NOT NULL,
  close  NUMERIC NOT NULL,
  volume NUMERIC NOT NULL
);

CREATE TABLE IF NOT EXISTS job_logs (
  id          BIGSERIAL PRIMARY KEY,
  job_name    TEXT NOT NULL,
  status      TEXT NOT NULL,
  message     TEXT,
  started_at  TIMESTAMP NOT NULL,
  finished_at TIMESTAMP NOT NULL
);
