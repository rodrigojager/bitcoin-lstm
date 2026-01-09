from datetime import datetime
from core.db import pg_conn

def log_job(job_name: str, status: str, message: str, started_at: datetime, finished_at: datetime):
    with pg_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """INSERT INTO job_logs(job_name, status, message, started_at, finished_at)
                   VALUES (%s,%s,%s,%s,%s);""",
                (job_name, status, message, started_at, finished_at)
            )
