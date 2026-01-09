import json
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


def _read_json_file(path: str | None) -> dict:
    if not path:
        return {}
    try:
        p = Path(path)
        if not p.is_file():
            return {}
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _env_int(name: str, default: int | None = None) -> int | None:
    v = os.getenv(name)
    if v is None:
        return default
    try:
        return int(v)
    except Exception:
        return default


def _env_float(name: str, default: float | None = None) -> float | None:
    v = os.getenv(name)
    if v is None:
        return default
    try:
        return float(v)
    except Exception:
        return default


class Settings:
    def __init__(self):
        # Arquivo opcional de policy/config (mantém controle fora do Site)
        self.TRAIN_POLICY_PATH = os.getenv("TRAIN_POLICY_PATH", "train_policy.json")
        cfg = _read_json_file(self.TRAIN_POLICY_PATH)

        # DB
        self.PG_DB = os.getenv("PG_DB")
        self.PG_USER = os.getenv("PG_USER")
        self.PG_PWD = os.getenv("PG_PWD")
        self.PG_HOST = os.getenv("PG_HOST")
        self.PG_PORT = _env_int("PG_PORT") or 5432

        # Binance
        self.BINANCE_BASE = os.getenv("BINANCE_BASE")
        self.BINANCE_SYMBOL = os.getenv("BINANCE_SYMBOL")
        self.BINANCE_INTERVAL = os.getenv("BINANCE_INTERVAL")
        self.BINANCE_LIMIT = _env_int("BINANCE_LIMIT", 1000) or 1000
        self.BINANCE_API_KEY = os.getenv("BINANCE_API_KEY")

        # Janela base de dados
        self.LOOKBACK_DAYS = _env_int("LOOKBACK_DAYS", 90) or 90
        self.ALPHA_DECAY = _env_float("ALPHA_DECAY", 0.999) or 0.999

        # Novos caminhos para LSTM
        self.LSTM_MODEL_PATH = os.getenv("LSTM_MODEL_PATH", "/app/models/lstm_model.keras")
        self.LSTM_BUNDLE_PATH = os.getenv("LSTM_BUNDLE_PATH", "/app/models/lstm_bundle.joblib")

        # Hiperparâmetros LSTM (podem vir do arquivo)
        self.LSTM_SEQ_LEN = int(cfg.get("lstm_seq_len", _env_int("LSTM_SEQ_LEN", 48) or 48))
        self.LSTM_EPOCHS = int(cfg.get("lstm_epochs", _env_int("LSTM_EPOCHS", 50) or 50))
        self.LSTM_BATCH_SIZE = int(cfg.get("lstm_batch_size", _env_int("LSTM_BATCH_SIZE", 64) or 64))
        self.LSTM_LR = float(cfg.get("lstm_lr", _env_float("LSTM_LR", 1e-3) or 1e-3))
        self.LSTM_PATIENCE = int(cfg.get("lstm_patience", _env_int("LSTM_PATIENCE", 8) or 8))

        # Política de retreino (24h ou 12h se MAPE(futures) > limiar)
        self.TRAIN_MAX_HOURS = float(cfg.get("train_max_hours", _env_float("TRAIN_MAX_HOURS", 24.0) or 24.0))
        self.TRAIN_MIN_HOURS = float(cfg.get("train_min_hours", _env_float("TRAIN_MIN_HOURS", 12.0) or 12.0))
        self.FUTURES_MAPE_THRESHOLD = float(
            cfg.get("futures_mape_threshold", _env_float("FUTURES_MAPE_THRESHOLD", 0.8) or 0.8)
        )
        self.FUTURES_ROLLING_N = int(cfg.get("futures_rolling_n", _env_int("FUTURES_ROLLING_N", 288) or 288))

        # Backfill
        self.BACKFILL_DAYS = _env_int("BACKFILL_DAYS", 90) or 90
        self.BACKFILL_SLEEP_MS = _env_int("BACKFILL_SLEEP_MS", 500) or 500

        # Subcaminho quando servido atrás de proxy reverso (Traefik) ex.: /fase4
        self.API_ROOT_PATH = os.getenv("API_PATH_PREFIX", "")


settings = Settings()

