"""
전역 설정 모듈.

- 환경변수(.env)로 값을 오버라이드할 수 있게 하여, 로컬/운영 환경을 코드 수정 없이 분리한다.
- 1인 개발 기준: 외부 설정 라이브러리(pydantic-settings 등) 버전 충돌을 피하기 위해
  표준 라이브러리 os.environ + python-dotenv 조합만 사용한다.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# backend/ 디렉터리 기준 경로 고정 (실행 위치와 무관하게 항상 동일한 파일을 참조)
BASE_DIR = Path(__file__).resolve().parent

load_dotenv(BASE_DIR / ".env")


def _env_path(name: str, default: str) -> Path:
    return Path(os.getenv(name, str(BASE_DIR / default)))


# SQLite DB 파일 경로
DB_PATH: Path = _env_path("DB_PATH", "data/restaurants.db")

# 학습 완료된 모델(pickle/joblib) 저장 디렉터리 (1년/3년/5년 각각)
MODEL_DIR: Path = _env_path("MODEL_DIR", "models")
MODEL_FILES = {
    1: os.getenv("MODEL_FILE_1Y", "model_1y.pkl"),
    3: os.getenv("MODEL_FILE_3Y", "model_3y.pkl"),
    5: os.getenv("MODEL_FILE_5Y", "model_5y.pkl"),
}

# 프론트엔드(Streamlit) Origin만 허용 (CORS) — 운영 배포 시 실제 도메인으로 교체
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "http://localhost:8501").split(",")

# 예측 대상 연차
HORIZON_YEARS = [1, 3, 5]
