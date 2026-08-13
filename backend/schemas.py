"""
API 요청/응답 스키마 (Pydantic).

FastAPI가 이 스키마로 입력을 자동 검증하므로,
타입이 안 맞거나 범위를 벗어난 요청은 서비스 로직에 닿기 전에 422로 걸러진다.
"""
from typing import Dict, Literal, Optional
from pydantic import BaseModel, ConfigDict, Field


class GuResponse(BaseModel):
    # gu_id: int
    gu_name: str


class DongResponse(BaseModel):
    # dong_id: int
    dong_name: str


class BusinessTypesResponse(BaseModel):
    # dong_id: int
    business_type: str


class AreaRange(BaseModel):
    min_area: float
    max_area: float


class PredictRequest(BaseModel):
    gu: str = Field(..., alias="구", min_length=1, max_length=20)
    dong: str = Field(..., alias="동", min_length=1, max_length=20)
    business_type: str = Field(..., alias="업태구분명", min_length=1, max_length=30)
    area: float = Field(..., alias="소재지면적", gt=0, le=10000)
    total_employees: float = Field(..., alias="총종사자수", gt=0)

    model_config = ConfigDict(populate_by_name=True)


class PredictInputResponse(BaseModel):
    gu: str = Field(alias="구")
    dong: str = Field(alias="동")
    business_type: str = Field(alias="업태구분명")
    area: float = Field(alias="소재지면적")
    same_business_count: int = Field(alias="동종업체수")
    total_employees: float = Field(alias="총종사자수")

    model_config = ConfigDict(populate_by_name=True)


# ML 모델 예측 생존율 스키마
class MLSuccessRateResponse(BaseModel):
    year_1: float = Field(alias="1year", ge=0, le=100)
    year_3: float = Field(alias="3year", ge=0, le=100)
    year_5: float = Field(alias="5year", ge=0, le=100)

    model_config = ConfigDict(populate_by_name=True)


# DB 통계 실측 생존율 스키마 (데이터가 없는 경우 None 허용)
class DBSuccessRateResponse(BaseModel):
    year_1: Optional[float] = Field(default=None, alias="1year", ge=0, le=100)
    year_3: Optional[float] = Field(default=None, alias="3year", ge=0, le=100)
    year_5: Optional[float] = Field(default=None, alias="5year", ge=0, le=100)
    sample_count: int = Field(default=0, alias="sample_count", ge=0)

    model_config = ConfigDict(populate_by_name=True)


# 최종 응답 스키마
class PredictResponse(BaseModel):
    input: PredictInputResponse
    ml_success_rate: MLSuccessRateResponse
    db_success_rate: DBSuccessRateResponse
    unit: Literal["percent"] = "percent"

    model_config = ConfigDict(populate_by_name=True)
    