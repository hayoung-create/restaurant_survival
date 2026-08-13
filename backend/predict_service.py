"""
1/3/5년 생존율 예측.

요건: "학습한 결과 모델은 별도 구성되어 있어서 생략"
-> 이 서비스는 모델을 학습하지 않고, 이미 만들어진 pipeline(pkl)을 로드해서
   FE 요청 시점에 predict_proba만 호출한다.

기대하는 모델 형식: sklearn Pipeline([("preprocessor", ColumnTransformer(...)), ("model", ...)])
(이전 대화에서 만든 get_accuracy_score2 학습 코드가 만든 pipeline과 동일 구조)
FEATURE_COLUMNS 순서/이름이 학습 시 target_cols와 다르면 반드시 맞춰야 한다.
"""
import threading
from typing import Dict

import joblib
import pandas as pd

from config import HORIZON_YEARS, MODEL_DIR, MODEL_FILES
from models import Gu, Dong, Business
from sqlalchemy.orm import Session
from sqlalchemy import select, func


FEATURE_COLUMNS = ["동", "소재지면적", "업태구분명", "동종업체수", "면적당종사자수", "구", "총종사자수"]

_model_cache: Dict[int, object] = {}
_lock = threading.Lock()


def _get_model(year: int):
    if year in _model_cache:
        return _model_cache[year]
    with _lock:
        if year not in _model_cache:  # double-checked locking (동시 요청 시 중복 로드 방지)
            path = MODEL_DIR / MODEL_FILES[year]
            if not path.exists():
                raise FileNotFoundError(
                    f"{year}년 모델 파일이 없습니다: {path}. "
                    "학습 파이프라인 산출물(pkl)을 backend/models/ 에 넣어주세요."
                )
            _model_cache[year] = joblib.load(path)
    return _model_cache[year]


def preload_models() -> None:
    """앱 기동 시 1회 호출 — 첫 예측 요청의 지연(cold load)을 없앤다."""
    for year in HORIZON_YEARS:
        try:
            _get_model(year)
        except FileNotFoundError as e:
            print(f"[경고] {e}")


def predict_survival_rates(
    gu: str, dong: str, business_type: str, area: float, derived: dict
) -> Dict[int, float]:
    row = pd.DataFrame(
        [
            {
                "동": dong,
                "소재지면적": area,
                "업태구분명": business_type,
                "동종업체수": derived["동종업체수"],
                "면적당종사자수": derived["면적당종사자수"],
                "구": gu,
                "총종사자수": derived["총종사자수"],
            }
        ]
    )[FEATURE_COLUMNS]

    results: Dict[int, float] = {}
    for year in HORIZON_YEARS:
        model = _get_model(year)
        proba_survive = model.predict_proba(row)[0, 1]  # label=1(영업중) 확률
        results[year] = round(float(proba_survive) * 100, 1)
    return results


def get_success_probability(model, input_df: pd.DataFrame) -> float:
    """
    모델의 성공 클래스(1)에 대한 확률을 반환한다.
    학습 라벨에서 성공이 1인 경우를 기준으로 작성.
    """

    # predict_proba 결과에서 클래스 1의 위치를 찾아 확률 반환
    if 1 not in model.classes_:
        raise ValueError(
            f"모델 클래스에 성공 라벨 1이 없습니다. 현재 클래스: {model.classes_}"
        )

    success_index = list(model.classes_).index(1)
    probability = model.predict_proba(input_df)[0][success_index]

    return round(float(probability) * 100, 2)



def get_all_baselines_dataframe(session: Session):
    stmt = select(
        Gu.gu_name,
        Dong.dong_name,
        Business.business_type,
        Business.label_1,
        Business.label_3,
        Business.label_5
    ).join(
        Dong, Business.dong_id == Dong.dong_id
    ).join(
        Gu, Dong.gu_id == Gu.gu_id
    ).where(
        Business.label_1.isnot(None)
    )

    # 데이터 로드 (SQLAlchemy 2.0+ 버전 지원)
    df = pd.read_sql(stmt, session.bind)

    # 동 + 업종별 집계
    baseline_df = df.groupby(['gu_name', 'dong_name', 'business_type']).agg(
        count=('label_1', 'count'),
        avg_1y=('label_1', 'mean'),
        avg_3y=('label_3', 'mean'),
        avg_5y=('label_5', 'mean')
    ).reset_index()

    return baseline_df