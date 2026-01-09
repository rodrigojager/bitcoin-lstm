import psycopg2
from core.config import settings

def pg_conn():
    return psycopg2.connect(
        dbname=settings.PG_DB,
        user=settings.PG_USER,
        password=settings.PG_PWD,
        host=settings.PG_HOST,
        port=settings.PG_PORT,
    )
