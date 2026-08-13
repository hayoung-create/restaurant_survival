# Restaurant Survival

Machine Learning project

서울 지역 음식점의 **1년·3년·5년 생존율**을 예측하고, 같은 지역·업종의 과거 DB 통계와 비교해 보여 주는 머신러닝 기반 웹 서비스입니다.

사용자는 구, 동, 업태, 매장 면적, 종사자 수를 입력합니다. 백엔드는 저장된 분류 모델로 연차별 생존 확률을 계산하고, 프론트엔드는 예측 결과와 실제 데이터 기반 평균 생존율을 시각화합니다.

> 이 프로젝트의 예측값은 데이터와 모델에 기반한 참고용 정보이며, 실제 창업 성과를 보장하지 않습니다.

## 주요 기능

- **1년 / 3년 / 5년 생존율 예측**: 각 기간별로 학습된 모델의 `predict_proba()` 결과를 백분율로 반환합니다.
- **지역·업종 기반 입력**: 구를 선택하면 해당 동 목록을 불러오고, DB에 저장된 업태를 선택할 수 있습니다.
- **동종업체 수 반영**: 선택한 구·동·업태에서 `영업/정상` 상태인 사업장 수를 계산해 모델 입력값으로 사용합니다.
- **DB 실측 통계 비교**: 같은 구·동·업태 표본의 1·3·5년 평균 생존율과 표본 수를 함께 제공합니다.
- **입력값 검증**: FastAPI와 Pydantic이 문자열 길이, 면적 및 종사자 수 범위를 검증합니다.
- **반응형 대시보드**: Streamlit과 Plotly로 연차별 예측 추세와 DB 통계 추세를 그래프로 표시합니다.
- **API 응답 캐싱**: 프론트엔드에서 지역·업종 조회 결과를 5분 동안 캐싱합니다.

## 기술 스택

| 구분 | 기술 |
|---|---|
| Frontend | Streamlit, Plotly, Pandas, Requests |
| Backend | FastAPI, Uvicorn, Pydantic |
| ML | scikit-learn Pipeline, XGBoost, Joblib |
| Database | SQLite, SQLAlchemy ORM |
| 환경 설정 | python-dotenv |

## 프로젝트 구조

```text
├── modeling/
│    └── main.py                # 년간 생존율 모델 생성
│    └── dbutil.py               # db 생성
│    └── 식품_일반음식점_서울특별시.zip               # 원본데이터
├── frontend/
│    └── app.py                 # Streamlit 프론트엔드
├── backend/
│    └── main.py                # FastAPI 앱 및 API 엔드포인트
│    └── predict_service.py    # 모델 로드·생존 확률·DB 기준 통계 처리
│    └── schemas.py            # 요청/응답 Pydantic 스키마
│    └── models.py              # SQLAlchemy ORM 모델
│    └── database.py            # SQLite 엔진 및 DB 세션 관리
│    └── config.py               # 환경변수, DB·모델 경로 설정
│    └── .env                     # 로컬 환경변수 (Git에 커밋하지 않음)
├── data/
│    └── restaurants.db     # SQLite 데이터베이스
└── models/
    ├── model_1y.pkl       # 1년 생존율 모델
    ├── model_3y.pkl       # 3년 생존율 모델
    └── model_5y.pkl       # 5년 생존율 모델
```

## 실행 방법

### 1. 준비 사항

- Python 3.10 이상 권장
- 원본데이터(식품_일반음식점_서울특별시.zip) 준비 : 압축해제 필요
- modeling 폴더 안의 main.py 를 실행하여 database 파일 생성 및 1년, 3년, 5년 모델 파일 생성

### 2. 가상환경 및 패키지 설치

```bash
python -m venv .venv
```

macOS / Linux:

```bash
source .venv/bin/activate
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

### 3. 모델과 데이터 배치

아래 기본 경로에 파일을 준비합니다.

```text
models/model_1y.pkl
models/model_3y.pkl
models/model_5y.pkl
data/restaurants.db
```

각 모델은 `predict_proba()`를 지원하는 scikit-learn 호환 Pipeline이어야 합니다. 학습에 사용한 피처 이름과 순서가 서비스의 입력 피처와 호환되어야 합니다.

### 4. 환경변수 설정

프로젝트 루트에 `.env` 파일을 만들고 환경에 맞게 수정합니다.

```env
DB_PATH=data/restaurants.db
MODEL_DIR=models
MODEL_FILE_1Y=model_1y.pkl
MODEL_FILE_3Y=model_3y.pkl
MODEL_FILE_5Y=model_5y.pkl
ALLOWED_ORIGINS=http://localhost:8501
```

프론트엔드가 다른 서버의 백엔드와 통신해야 할 경우 `API_BASE_URL`을 설정합니다.

macOS / Linux:

```bash
export API_BASE_URL=http://localhost:8001
```

Windows PowerShell:

```powershell
$env:API_BASE_URL="http://localhost:8001"
```

설정하지 않으면 기본값 `http://localhost:8001`을 사용합니다.

### 5. 백엔드 실행

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8001
```

정상적으로 실행되면 다음 주소에서 상태를 확인할 수 있습니다.

```text
http://localhost:8001/health
```

FastAPI 자동 문서는 다음 주소에서 확인합니다.

```text
http://localhost:8001/docs
```

### 6. 프론트엔드 실행

별도 터미널에서 실행합니다.

```bash
streamlit run app.py
```

기본적으로 브라우저의 `http://localhost:8501`에서 서비스를 이용할 수 있습니다.

## 예측 흐름

1. 프론트엔드가 구·동·업태 목록 및 업태별 면적 범위를 API로 조회합니다.
2. 사용자가 구, 동, 업태, 소재지 면적, 총 종사자 수를 선택합니다.
3. `POST /api/predict`가 같은 구·동·업태의 영업 중 동종업체 수를 계산합니다.
4. 1년·3년·5년 모델이 각각 생존 클래스(`1`)의 확률을 예측합니다.
5. DB에서 같은 구·동·업태의 연차별 생존 라벨 평균과 표본 수를 집계합니다.
6. 프론트엔드는 모델 예측값, DB 통계, 추세 그래프를 표시합니다.

### 모델 입력 피처

| 피처 | 설명 |
|---|---|
| `구` | 음식점이 위치한 자치구 |
| `동` | 음식점이 위치한 행정동 |
| `업태구분명` | 음식점 업태 분류 |
| `소재지면적` | 영업장 면적 |
| `총종사자수` | 총 종사자 수 |
| `동종업체수` | 같은 구·동·업태에서 영업 중인 사업장 수 |

## API 명세

| Method | Endpoint | 설명 |
|---|---|---|
| `GET` | `/health` | 서버 상태 확인 |
| `GET` | `/api/gu` | 구 목록 조회 |
| `GET` | `/api/dong?gu={구}` | 선택한 구의 동 목록 조회 |
| `GET` | `/api/business-types` | 업태 목록 조회 |
| `GET` | `/api/area-range` | 면적 최솟값·최댓값 조회 |
| `POST` | `/api/predict` | 1·3·5년 예측 생존율과 DB 통계 반환 |

### 예측 요청 예시

```json
{
  "구": "강남구",
  "동": "역삼동",
  "업태구분명": "한식",
  "소재지면적": 80.0,
  "총종사자수": 4
}
```

### 예측 응답 예시

```json
{
  "input": {
    "구": "강남구",
    "동": "역삼동",
    "업태구분명": "한식",
    "소재지면적": 80.0,
    "동종업체수": 120,
    "총종사자수": 4
  },
  "ml_success_rate": {
    "1year": 82.34,
    "3year": 61.27,
    "5year": 43.18
  },
  "db_success_rate": {
    "1year": 79.5,
    "3year": 58.2,
    "5year": 40.1,
    "sample_count": 210
  },
  "unit": "percent"
}
```

> 위 수치는 API 형식을 설명하기 위한 예시이며 실제 모델 결과가 아닙니다.

## 데이터베이스 구성

서비스는 SQLite DB의 다음 테이블을 사용합니다.

| 테이블 | 역할 | 주요 컬럼 |
|---|---|---|
| `gu` | 자치구 정보 | `gu_id`, `gu_name` |
| `dong` | 행정동 정보 | `dong_id`, `gu_id`, `dong_name` |
| `business` | 음식점 및 생존 라벨 정보 | 업태, 면적, 종사자 수, 영업 상태, `label_1`, `label_3`, `label_5` |

`label_1`, `label_3`, `label_5`는 각 관측 기간의 생존 여부를 나타내는 이진 라벨입니다. 같은 구·동·업태의 라벨 평균을 계산해 DB 실측 생존율로 활용합니다.

## 주의 사항

- `.env`, SQLite DB 원본, 모델 파일은 민감하거나 용량이 클 수 있으므로 저장소에 직접 커밋하지 않는 것을 권장합니다.
- 모델의 입력 컬럼 구성은 학습 시 사용한 Pipeline과 반드시 일치해야 합니다.
- 지역·업종 조합의 DB 표본이 없으면 `db_success_rate`의 연차별 값은 `null`로 반환될 수 있습니다.
- 예측 요청에서 존재하지 않는 업태를 보내면 서버는 `400 Bad Request`를 반환합니다.
- 면적은 0보다 크고 10,000 이하, 종사자 수는 0보다 커야 합니다.

## 개선 아이디어

- 모델 성능 지표(Accuracy, F1-score, ROC-AUC)와 학습 데이터 기간을 화면에 공개
- 업종·지역별 표본 수가 작을 때 신뢰구간 또는 표본 부족 경고 제공
- 매출, 유동인구, 임대료, 경쟁 강도 등 설명 변수를 추가한 모델 고도화
- Docker Compose로 프론트엔드·백엔드·DB 실행 환경 통합
- 테스트 코드와 CI/CD 파이프라인 추가

## License

학습 및 포트폴리오 목적의 프로젝트입니다. 데이터와 모델 파일의 사용 조건은 각 원본 데이터의 라이선스를 별도로 확인하세요.
