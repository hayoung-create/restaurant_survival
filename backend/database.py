"""
SQLite 연결 및 세션 관리.

- SQLAlchemy ORM을 사용해 원문 SQL 문자열 조립을 피한다 (SQL Injection 방지).
- SQLite는 기본적으로 동시 쓰기에 약하므로 WAL 모드를 켜서 읽기 동시성을 개선한다.
  (이 서비스는 조회/예측이 대부분이라 WAL만으로 충분하다. 쓰기가 잦아지면 Postgres 등으로 이관 검토.)
"""
from sqlalchemy import create_engine, event
from sqlalchemy.orm import declarative_base, sessionmaker
from config import DB_PATH

DB_PATH.parent.mkdir(parents=True, exist_ok=True)

engine = create_engine(
    f"sqlite:///{DB_PATH}",
    connect_args={"check_same_thread": False},  # FastAPI는 요청마다 스레드가 다를 수 있음
)


@event.listens_for(engine, "connect")
def _set_sqlite_pragma(dbapi_connection, connection_record):
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.close()


SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def get_db():
    """FastAPI Depends()로 주입할 세션. 요청 종료 시 반드시 close."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
