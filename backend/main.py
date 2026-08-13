"""
음식점 생존율 예측 서비스 - BE (FastAPI)

엔드포인트:
  GET  /api/gu                     - 구 리스트
  GET  /api/dong?gu=...             - 선택한 구에 속한 동 리스트
  GET  /api/business-types          - 업종(업태구분명) 리스트
  GET  /api/area-range              - 소재지면적 최저~최고 (사전입력 범위 안내용)
  POST /api/predict                 - 1/3/5년 생존율 예측 + DB 실측 비교 + 동일 구 비교

실행: __main__ 에 포함
  uvicorn main:app --reload --port 8000
"""
import math
import joblib
import numpy as np
import pandas as pd
import predict_service

from contextlib import asynccontextmanager
from typing import List, Optional
from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import distinct, func, case
from sqlalchemy.orm import Session

from config import ALLOWED_ORIGINS, HORIZON_YEARS, MODEL_DIR, MODEL_FILES
from database import get_db, SessionLocal
from models import Gu, Dong, Business
from schemas import (
    GuResponse, DongResponse, BusinessTypesResponse, 
    AreaRange, PredictRequest, PredictResponse
)


# NaN/Inf 값을 안전하게 None 또는 float으로 변환하는 헬퍼 함수
def sanitize_float(val) -> Optional[float]:
    if val is None:
        return None
    try:
        f_val = float(val)
        if math.isnan(f_val) or math.isinf(f_val) or pd.isna(f_val):
            return None
        return f_val
    except (ValueError, TypeError):
        return None


# 모델 로드
model_1 = joblib.load(MODEL_DIR / MODEL_FILES[1])
model_3 = joblib.load(MODEL_DIR / MODEL_FILES[3])
model_5 = joblib.load(MODEL_DIR / MODEL_FILES[5])

df_baseline = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 콜드 스타트 지연 방지 (모델 파일 없으면 경고만 출력하고 계속 기동)
    predict_service.preload_models()  

    # db에서 기초 데이터 로딩하기
    global df_baseline
    db = SessionLocal()
    try:
        print("Loading baseline data from DB...")
        df_baseline = predict_service.get_all_baselines_dataframe(session=db)
        print("Baseline data loaded successfully!")
    finally:
        db.close() # 작업 후 세션 닫기
    yield


app = FastAPI(title="음식점 생존율 예측 API", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,   # 운영 배포 시 실제 FE 도메인만 나열 (와일드카드 금지)
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/api/gu", response_model=List[str])
def get_gu_list(db: Session = Depends(get_db)):
    rows = (
        db.query(distinct(Gu.gu_name))
        .order_by(Gu.gu_name)
        .all()
    )
    return [row[0] for row in rows]


@app.get("/api/dong", response_model=List[str])
def get_dong_list(
    gu: str = Query(..., min_length=1),
    db: Session = Depends(get_db),
):
    rows = (
        db.query(distinct(Dong.dong_name))
        .join(Gu, Dong.gu_id == Gu.gu_id)
        .filter(Gu.gu_name == gu)
        .order_by(Dong.dong_name)
        .all()
    )
    return [row[0] for row in rows]


@app.get("/api/business-types", response_model=List[str])
def get_business_types(db: Session = Depends(get_db)):
    rows = db.query(distinct(Business.business_type)).order_by(Business.business_type).all()
    return [r[0] for r in rows]


@app.get("/api/area-range", response_model=AreaRange)
def get_area_range(db: Session = Depends(get_db)):
    # 전체 데이터셋 기준 최소/최대 면적 조회
    min_area, max_area = db.query(
        func.min(Business.site_area), 
        func.max(Business.site_area)
    ).one()

    return AreaRange(
        min_area=min_area or 0.0, 
        max_area=max_area or 0.0
    )


@app.post("/api/predict", response_model=PredictResponse)
def predict(payload: PredictRequest, db: Session = Depends(get_db)):
    # 1. 입력 검증
    exists_type = (
        db.query(Business.business_id).filter(Business.business_type == payload.business_type).first()
    )
    if not exists_type:
        raise HTTPException(status_code=400, detail=f"알 수 없는 업종입니다: {payload.business_type}")

    # 2. 동종업체수 집계 (안전한 예외 처리)
    count_1yr, count_3yr, count_5yr = 0, 0, 0

    try:
        if payload.gu and payload.dong and payload.business_type:
            business_count = (
                db.query(
                    func.coalesce(
                        func.sum(case((Business.label_1 == 1, 1), else_=0)), 0
                    ).label("count_1yr"),
                    func.coalesce(
                        func.sum(case((Business.label_3 == 1, 1), else_=0)), 0
                    ).label("count_3yr"),
                    func.coalesce(
                        func.sum(case((Business.label_5 == 1, 1), else_=0)), 0
                    ).label("count_5yr"),
                )
                .select_from(Business)
                .join(Dong, Business.dong_id == Dong.dong_id)
                .join(Gu, Dong.gu_id == Gu.gu_id)
                .filter(
                    Gu.gu_name == payload.gu,
                    Dong.dong_name == payload.dong,
                    Business.business_type == payload.business_type,
                )
                .first()
            )

            if business_count:
                count_1yr = int(getattr(business_count, "count_1yr", 0) or 0)
                count_3yr = int(getattr(business_count, "count_3yr", 0) or 0)
                count_5yr = int(getattr(business_count, "count_5yr", 0) or 0)

    except Exception as e:
        print(f"[Warning] 동종업체수 조회 중 오류 발생: {e}")
        count_1yr, count_3yr, count_5yr = 0, 0, 0

    # 3. Payload 필드 접근 안전성 확보 (area, total_employees 또는 소재지면적, 총종사자수)
    area_val = getattr(payload, "area", None) or getattr(payload, "소재지면적", 0.0)
    workers_val = getattr(payload, "total_employees", None) or getattr(payload, "총종사자수", 1)

    # 4. ML 모델 입력용 DataFrame 구성 (안전하게 산출된 count_1yr, count_3yr, count_5yr 변수 사용)
    input_df_1year = pd.DataFrame([{
        "동": payload.dong,
        "소재지면적": area_val,
        "업태구분명": payload.business_type,
        "동종업체수": count_1yr,
        "구": payload.gu,
        "총종사자수": workers_val,
    }])

    input_df_3year = pd.DataFrame([{
        "동": payload.dong,
        "소재지면적": area_val,
        "업태구분명": payload.business_type,
        "동종업체수": count_3yr,
        "구": payload.gu,
        "총종사자수": workers_val,
    }])

    input_df_5year = pd.DataFrame([{
        "동": payload.dong,
        "소재지면적": area_val,
        "업태구분명": payload.business_type,
        "동종업체수": count_5yr,
        "구": payload.gu,
        "총종사자수": workers_val,
    }])

    try:
        # ML 예측
        ml_1year = predict_service.get_success_probability(model_1, input_df_1year)
        ml_3year = predict_service.get_success_probability(model_3, input_df_3year)
        ml_5year = predict_service.get_success_probability(model_5, input_df_5year)

        # DB 통계 평균 계산
        db_1year, db_3year, db_5year = None, None, None
        sample_count = 0

        if df_baseline is not None:
            matched = df_baseline[
                (df_baseline['gu_name'] == payload.gu) &
                (df_baseline['dong_name'] == payload.dong) &
                (df_baseline['business_type'] == payload.business_type)
            ]
            
            if not matched.empty:
                row = matched.iloc[0]
                
                # sanitize_float를 활용해 NaN/Inf 발생 시 안전하게 None으로 대체
                raw_1y = row.get('avg_1y')
                raw_3y = row.get('avg_3y')
                raw_5y = row.get('avg_5y')

                if raw_1y is not None and not pd.isna(raw_1y):
                    db_1year = round(float(raw_1y) * 100, 2)
                if raw_3y is not None and not pd.isna(raw_3y):
                    db_3year = round(float(raw_3y) * 100, 2)
                if raw_5y is not None and not pd.isna(raw_5y):
                    db_5year = round(float(raw_5y) * 100, 2)

                sample_count = int(row.get('count', 0))

        # 최종 응답 생성 (NaN 검증 통과용 sanitize_float 적용)
        return {
            "input": {
                "동": payload.dong,
                "소재지면적": area_val,
                "업태구분명": payload.business_type,
                "동종업체수": max(count_1yr, count_3yr, count_5yr),
                "구": payload.gu,
                "총종사자수": workers_val,
            },
            "ml_success_rate": {
                "1year": sanitize_float(ml_1year),
                "3year": sanitize_float(ml_3year),
                "5year": sanitize_float(ml_5year),
            },
            "db_success_rate": {
                "1year": sanitize_float(db_1year),
                "3year": sanitize_float(db_3year),
                "5year": sanitize_float(db_5year),
                "sample_count": sample_count
            },
            "unit": "percent"
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"예측 중 오류가 발생했습니다: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)