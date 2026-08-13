import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
import dbutil


# 원본 데이터를 읽고 필요한 데이터 형태로 가공
def get_default_data() -> pd.DataFrame:
    """
    원본 데이터가 담긴 csv 파일을 읽은 후 데이터를 처리하기 편하게 가공한다.
    """
    df_tmp = pd.read_csv('식품_일반음식점_서울특별시.csv', encoding='cp949', low_memory=False)

    print(df_tmp.shape)

    # 불필요한 컬럼 제거
    df_tmp = df_tmp.drop(columns=['소재지우편번호', '도로명우편번호', '데이터갱신구분', '건물소유구분명',
        '공장사무직직원수', '공장생산직직원수', '공장판매직직원수', '급수시설구분명', '다중이용업소여부',
        '데이터갱신시점', '도로명주소', '등급구분명', '보증액', '본사직원수', '영업장주변구분명', '월세액',
        '전통업소주된음식', '전통업소지정번호', '전화번호', '좌표정보(X)', '좌표정보(Y)', '홈페이지', '시설총규모',
        '개방자치단체코드','관리번호', '상세영업상태명', '상세영업상태코드', '영업상태코드', '위생업태명'])

    # 파일 로드 후 컬럼명의 앞뒤 공백 일괄 제거
    # df_tmp.columns = df_tmp.columns.str.strip()

    # 날짜 형식의 데이터 타입 정리
    df_tmp['인허가일자'] = pd.to_datetime(df_tmp['인허가일자'], errors='coerce')
    df_tmp['폐업일자'] = pd.to_datetime(df_tmp['폐업일자'], errors='coerce')
    df_tmp['최종수정시점'] = pd.to_datetime(df_tmp['최종수정시점'], errors='coerce')

    # 인허가일자 기준으로 2000년 이후(2000년 1월 1일 포함) 데이터만 필터링
    df_2000 = df_tmp[df_tmp['인허가일자'] >= '2000-01-01'].copy()

    # 결측치가 있는 데이터 제거 (소재지면적, 지번주소) 및 데이터 타입 변경
    df_2000.dropna(subset=['소재지면적'], inplace=True)
    df_2000['소재지면적'] = pd.to_numeric(df_2000['소재지면적'], errors='coerce')

    df_2000.dropna(subset=['지번주소'], inplace=True)
    # print(df_2000['소재지면적'].dtypes)

    # 결측치 보정 (남성종사자수, 여성종사자수)
    for col in ['남성종사자수', '여성종사자수']:
        # 1. 숫자가 아닌 값(공백, 이상치 등)은 NaN으로 만들고 숫자형으로 변환
        df_2000[col] = pd.to_numeric(df_2000[col], errors='coerce')
        
        # 2. NaN(결측치)을 0으로 채우고 정수형(int)으로 변환
        df_2000[col] = df_2000[col].fillna(0).astype(int)

    # 구와 동을 분류
    df_2000['구'] = df_2000['지번주소'].str.split().str.get(1)
    df_2000['동'] = df_2000['지번주소'].str.split().str.get(2)

    # 결과 확인
    print(f"기본 데이터 건수: {len(df_tmp)}")
    # print(f"2000년 이후 데이터 건수: {len(df_2000)}")
    # print(df_after_2000[['상호명', '인허가일자']].head())

    print(f"2000년 이후 데이터 Shape : {df_2000.shape}")

    return df_2000


def get_yearly_data(df_input: pd.DataFrame, years_data: int=1, char_type: int=0) -> pd.DataFrame:
    """
    기본 데이터 중 '최종수정시점'에서 '인허가일자'를 빼서 1년, 3년, 5년 데이터 구성
    df_input : 기본 데이터
    years_data : 1, 3, 5
    show_chart : 0(안보여줌), 1(feature importance), 2(shap)
    """
    # 입력된 years_data 기준으로 데이터 재분류
    df_years = df_input[df_input['인허가일자'] + pd.DateOffset(years=years_data) <= df_input['최종수정시점']].copy()

    # 영업상태명이 폐업이면 0, 그 이외에는 1로 lable 구성
    df_years['label'] = (df_years['영업상태명'] != '폐업').astype(int)

    # 남성종사자수 및 여성종사자수의 결측치를 0으로 채우고, 총종사자수를 재구성
    df_years["총종사자수"] = df_years["남성종사자수"].fillna(0) + df_years["여성종사자수"].fillna(0)

    # 면적당 종사자수 신규 구성
    # df_years["면적당종사자수"] = df_years["총종사자수"] / (df_years["소재지면적"] + 0.01)
    
    # 동종업체 경쟁 밀도 피처 생성
    # df_years["동종업체수"] = df_years.groupby(["동", "업태구분명"])["사업장명"].transform("count")
    df_years["동종업체수"] = df_years.groupby(["구", "동", "업태구분명"])["label"].transform("sum")

    # ratio = (df_2000['영업상태명'] == '폐업').mean()
    ratio = (df_years['label'] == 0).mean()
    print(f"2000년 이후 {years_data}년 이내 폐업 비율: {ratio * 100:.2f}%")  # 예: 25.34%
    # print(df_years['label'].value_counts())
    
    # 수정된 데이터 저장
    # df_years.to_csv('last_data.csv', encoding='cp949', index=False)

    plt.rcParams['font.family'] = 'Malgun Gothic'  # 맑은 고딕
    plt.rcParams['axes.unicode_minus'] = False     # 음수 기호 깨짐 방지

    if char_type != 0:
        show_chart(df_years, years_data, char_type)

    return df_years


def show_chart(df_years: pd.DataFrame, years_data: int=1, chart_type: int=0):
    # 데이터셋 중 인허가 시점 이후에 결정되는 사후 정보를 모델에 입력하면 해당 변수만으로 100% 정답을 맞춰버려 진짜 유의미한 특징을 찾을 수 없음
    leakage_cols = [
        "영업상태명",
        "폐업일자",
        "최종수정시점",
        "사업장명",
        "지번주소",
        "인허가일자"
    ]
    feature_df = df_years.drop(columns=[col for col in leakage_cols if col in df_years.columns])

    # 범주형 변수 타입 변환 (LightGBM 지원)
    cat_cols = ["구", "동", "업태구분명"]
    for col in cat_cols:
        if col in feature_df.columns:
            feature_df[col] = feature_df[col].astype("category")

    X = feature_df.drop(columns=["label"])
    y = feature_df["label"]

    # 5. Train/Test 분할 및 모델 학습
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    if chart_type == 1:
        # 모델 생성 시 importance_type='gain' 설정 (추천)
        model = LGBMClassifier(
            n_estimators=200,
            importance_type="gain",  # 'split'(분할횟수) 대신 'gain'(기여도) 사용
            random_state=42,
            verbose=-1,
        )
        model.fit(X_train, y_train)

        # 피처 중요도 추출 및 데이터프레임 정리
        feature_imp = pd.DataFrame(
            {"Feature": X_train.columns, "Importance": model.feature_importances_}
        ).sort_values(by="Importance", ascending=False)

        # 시각화
        plt.figure(figsize=(10, 6))
        sns.barplot(x="Importance", y="Feature", data=feature_imp)
        plt.title(f"{years_data}년 이내 - Feature Importance (Gain 기준)", fontsize=14, pad=20)
        plt.xlabel("Importance Score")
        plt.ylabel("Features")
        plt.tight_layout()
        plt.show()
    elif chart_type == 2:
        model = LGBMClassifier(n_estimators=200, random_state=42, verbose=-1)
        model.fit(X_train, y_train)

        # 최신 Explainer 객체 생성 (X_train 또는 X_test 기반)
        explainer = shap.TreeExplainer(model)
        shap_values = explainer(X_test)

        shap.summary_plot(shap_values, X_test, show=False)

        plt.title(f"{years_data}년 이내 - SHAP Feature Importance", fontsize=14, pad=20)
        plt.tight_layout()
        plt.show()


def get_accuracy_score(df_years: pd.DataFrame, model_type: int=0):
    feature_order = [
        "동",
        "소재지면적",
        "업태구분명",
        "동종업체수",
        # "면적당종사자수",
        "구",
        "총종사자수",
        # "여성종사자수",
        # "남성종사자수"
    ]

    X = df_years[feature_order].copy()
    y = df_years['label']

    # 2. 학습용·테스트용 데이터 분리
    # stratify=y: 생존/폐업 비율을 두 데이터에 비슷하게 유지
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # results = []

    # 3. 상위 5개 → 9개까지 순서대로 변수 추가
    for n_features in range(7, 8):
        selected_features = feature_order[:n_features]

        X_train_selected = X_train[selected_features]
        X_test_selected = X_test[selected_features]

        # 숫자형·범주형 변수 자동 구분
        numeric_features = X_train_selected.select_dtypes(
            include=["int64", "float64"]
        ).columns.tolist()

        categorical_features = [
            col for col in selected_features
            if col not in numeric_features
        ]

        # 결측치 처리 및 범주형 원-핫 인코딩
        numeric_transformer = Pipeline([
            ("imputer", SimpleImputer(strategy="median"))
        ])

        categorical_transformer = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ])

        preprocessor = ColumnTransformer([
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features)
        ])

        print("==================================================")

        # 모델 설정
        if model_type == 0:
            # RandomForest
            print(f"model : RandomForest")
            model = RandomForestClassifier(
                n_estimators=300,
                max_depth=10,
                min_samples_leaf=3,
                random_state=42,
                class_weight="balanced"
            )
        elif model_type == 1:
            # Gradient Boosting
            print(f"model : GradientBoosting")
            model = GradientBoostingClassifier(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=5,
                random_state=42
            )
        elif model_type == 2:
            # Logistic Regression
            print(f"model : LogisticRegression")
            model = LogisticRegression(
                solver='saga',
                max_iter=5000,
                random_state=42,
                class_weight='balanced'
            )

        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("model", model)
        ])

        # 학습
        pipeline.fit(X_train_selected, y_train)

        # 예측 및 평가
        y_pred = pipeline.predict(X_test_selected)
        y_prob = pipeline.predict_proba(X_test_selected)[:, 1]

        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)

        # print("==================================================")
        # print(f"사용 변수 수 : {n_features}")
        # print(f"추가된 변수 : {feature_order[n_features - 1]}")
        # print(f"사용 변수 : {', '.join(selected_features)}")
        print(f"Accuracy : {round(accuracy, 4)}")
        print(f"F1-score : {round(f1, 4)}")
        # print(f"ROC-AUC : {round(auc, 4)}")
        print("==================================================\n")



def train_random_forest_models(
    df_1years,
    df_3years,
    df_5years
):

    models = [
        ("1년", df_1years),
        ("3년", df_3years),
        ("5년", df_5years)
    ]

    feature_order = [
            "동",
            "소재지면적",
            "업태구분명",
            "동종업체수",
            "면적당종사자수",
            "구",
            "총종사자수"
        ]

    results = []

    for period, df_years in models:

        print("\n" + "=" * 70)
        print(f"{period} Random Forest")
        print("=" * 70)

        # -------------------------
        # X / y
        # -------------------------
        X = df_years[feature_order].copy()
        y = df_years["label"]

        # -------------------------
        # Train / Test
        # -------------------------
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=y
        )

        # -------------------------
        # 숫자형 / 범주형 구분
        # -------------------------
        numeric_features = X_train.select_dtypes(
            include=["int64", "float64"]
        ).columns.tolist()

        categorical_features = [
            col for col in feature_order
            if col not in numeric_features
        ]

        # -------------------------
        # 전처리
        # -------------------------
        numeric_transformer = Pipeline([
            ("imputer", SimpleImputer(strategy="median"))
        ])

        categorical_transformer = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ])

        preprocessor = ColumnTransformer([
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features)
        ])

        # -------------------------
        # Random Forest
        # -------------------------
        rf = RandomForestClassifier(
            n_estimators=300,
            max_depth=10,
            min_samples_leaf=3,
            random_state=42,
            class_weight="balanced"
        )

        # -------------------------
        # Pipeline
        # -------------------------
        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("model", rf)
        ])

        # -------------------------
        # 학습
        # -------------------------
        pipeline.fit(X_train, y_train)

        # -------------------------
        # 예측
        # -------------------------
        y_pred = pipeline.predict(X_test)
        y_prob = pipeline.predict_proba(X_test)[:, 1]

        # -------------------------
        # 평가
        # -------------------------
        accuracy = accuracy_score(y_test, y_pred)

        f1 = f1_score(
            y_test,
            y_pred,
            zero_division=0
        )

        auc = roc_auc_score(
            y_test,
            y_prob
        )

        # -------------------------
        # 결과 출력
        # -------------------------
        print(f"사용 Feature : {feature_order}")
        print(f"Accuracy     : {accuracy:.4f}")
        print(f"F1-score     : {f1:.4f}")
        print(f"ROC-AUC      : {auc:.4f}")

        # -------------------------
        # 결과 저장
        # -------------------------
        results.append({
            "기간": period,
            "Feature수": len(feature_order),
            "Features": ", ".join(feature_order),
            "Accuracy": accuracy,
            "F1": f1,
            "ROC-AUC": auc
        })

    # -------------------------
    # 최종 결과표
    # -------------------------
    result_df = pd.DataFrame(results)

    print("\n")
    print("=" * 90)
    print("최종 Random Forest 결과")
    print("=" * 90)

    print(
        result_df[
            [
                "기간",
                "Feature수",
                "Accuracy",
                "F1",
                "ROC-AUC"
            ]
        ].to_string(index=False)
    )

    return result_df


def create_random_forest_model_and_save(df_years: pd.DataFrame, year_data: int):
    """
    선정된 6개 Feature를 사용하여
    1년 / 3년 / 5년 Random Forest 모델을 학습하고 저장한다.
    """

    # 선정된 7개 Feature
    selected_features = [
        "동",
        "소재지면적",
        "업태구분명",
        "동종업체수",
        # "면적당종사자수",
        "구",
        "총종사자수"
    ]

    # X, y
    X = df_years[selected_features].copy()
    y = df_years["label"]

    # 학습 / 테스트 분리
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # 숫자형 / 범주형 구분
    numeric_features = X_train.select_dtypes(
        include=["int64", "float64"]
    ).columns.tolist()

    categorical_features = [
        col for col in selected_features
        if col not in numeric_features
    ]

    # 숫자형 전처리
    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median"))
    ])

    # 범주형 전처리
    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    # 전처리 설정
    preprocessor = ColumnTransformer([
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ])

    # Random Forest
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=10,
        min_samples_leaf=3,
        random_state=42,
        class_weight="balanced"
    )

    # 전처리 + Random Forest
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    # ==============================
    # 모델 학습
    # ==============================

    pipeline.fit(X_train, y_train)

    # ==============================
    # 모델 저장
    # ==============================
    os.makedirs("./models", exist_ok=True)

    model_path = f"./models/model_{year_data}y.pkl"

    joblib.dump(
        pipeline,
        model_path
    )

    print(f"{year_data}년 모델 저장 완료")
    print(f"저장 위치 : {model_path}")



def save_database(df_default: pd.DataFrame):
    df_temp = df_default.copy()

    df_temp["총종사자수"] = df_temp["남성종사자수"].fillna(0) + df_temp["여성종사자수"].fillna(0)
    df_temp = df_temp.drop(columns=["남성종사자수", "여성종사자수"])
    
    # 각 기간 이상 지난 데이터 조건
    mask_1 = (
        df_temp["인허가일자"] + pd.DateOffset(years=1)
        <= df_temp["최종수정시점"]
    )

    mask_3 = (
        df_temp["인허가일자"] + pd.DateOffset(years=3)
        <= df_temp["최종수정시점"]
    )

    mask_5 = (
        df_temp["인허가일자"] + pd.DateOffset(years=5)
        <= df_temp["최종수정시점"]
    )

    # 우선 모든 라벨을 결측값으로 생성
    df_temp["label_1"] = pd.NA
    df_temp["label_3"] = pd.NA
    df_temp["label_5"] = pd.NA

    # 각 기간을 충족한 행에만 라벨 부여
    # 폐업이 아니면 1(영업 성공), 폐업이면 0
    df_temp.loc[mask_1, "label_1"] = (
        df_temp.loc[mask_1, "영업상태명"] != "폐업"
    ).astype(int)

    df_temp.loc[mask_3, "label_3"] = (
        df_temp.loc[mask_3, "영업상태명"] != "폐업"
    ).astype(int)

    df_temp.loc[mask_5, "label_5"] = (
        df_temp.loc[mask_5, "영업상태명"] != "폐업"
    ).astype(int)

    # 정수형 nullable dtype으로 변환
    df_temp["label_1"] = df_temp["label_1"].astype("Int64")
    df_temp["label_3"] = df_temp["label_3"].astype("Int64")
    df_temp["label_5"] = df_temp["label_5"].astype("Int64")

    dbutil.build_sqlite_database(
        df=df_temp,
        db_path="restaurant.db"
    )



if __name__ == "__main__":

    # 원본 데이터를 기초 가공한 기본 데이터로 로딩
    df_default = get_default_data()

    # 기초 데이터를 DB에 저장
    # save_database(df_default)

    # 기본 데이터에서 '최종수정시점'에서 '인허가일자'를 빼서 1년, 3년, 5년 데이터 구성
    df_1years = get_yearly_data(df_default, 1, 0)
    df_3years = get_yearly_data(df_default, 3, 0)
    df_5years = get_yearly_data(df_default, 5, 0)

    # df_3years.to_csv('data_3_years.csv', encoding='cp949', index=False)

    # 정확도 측정, 년도별 데이터를 3개의 기준 모델로 정확도를 평가함
    # 0 : RandomForestClassifier, 1 : GradientBoostingClassifier, 2 : LogisticRegression
    # get_accuracy_score(df_1years, 0)
    # get_accuracy_score(df_1years, 1)
    # get_accuracy_score(df_1years, 2)
    
    # get_accuracy_score(df_3years, 0)
    # get_accuracy_score(df_3years, 1)
    # get_accuracy_score(df_3years, 2)

    # get_accuracy_score(df_5years, 0)
    # get_accuracy_score(df_5years, 1)
    # get_accuracy_score(df_5years, 2)

    # 모델 생성 및 저장
    create_random_forest_model_and_save(df_1years, 1)
    create_random_forest_model_and_save(df_3years, 3)
    create_random_forest_model_and_save(df_5years, 5)