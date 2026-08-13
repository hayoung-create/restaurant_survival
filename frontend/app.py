import os
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
import streamlit.components.v1 as components


API_BASE = os.getenv("API_BASE_URL", "http://localhost:8000")

YEARS = ["1년", "3년", "5년"]


# --------------------------------------------------
# API 캐시 함수 정의 (show_spinner=False 로 기본 running 문구 비활성화)
# --------------------------------------------------

@st.cache_data(ttl=300, show_spinner=False)
def fetch_gu():
    r = requests.get(f"{API_BASE}/api/gu", timeout=5)
    r.raise_for_status()
    return r.json()


@st.cache_data(ttl=300, show_spinner=False)
def fetch_dong(gu: str):
    r = requests.get(
        f"{API_BASE}/api/dong",
        params={"gu": gu},
        timeout=5,
    )
    r.raise_for_status()
    return r.json()


@st.cache_data(ttl=300, show_spinner=False)
def fetch_business_types():
    r = requests.get(
        f"{API_BASE}/api/business-types",
        timeout=5,
    )
    r.raise_for_status()
    return r.json()


@st.cache_data(ttl=300, show_spinner=False)
def fetch_area_worker_range():
    r = requests.get(
        f"{API_BASE}/api/area-range",
        timeout=5,
    )
    r.raise_for_status()
    return r.json()


def call_predict(
    gu: str,
    dong: str,
    business_type: str,
    area: float,
    workers: int,
):
    payload = {
        "구": gu,
        "동": dong,
        "업태구분명": business_type,
        "소재지면적": area,
        "총종사자수": workers,
    }

    r = requests.post(
        f"{API_BASE}/api/predict",
        json=payload,
        timeout=15,
    )
    r.raise_for_status()
    return r.json()


# --------------------------------------------------
# 페이지 기본 설정
# --------------------------------------------------

st.set_page_config(
    page_title="Restaurant Survival",
    layout="wide",
)

# --------------------------------------------------
# 1. 테마별 CSS 설정
# --------------------------------------------------
st.markdown(
    """
    <style>
    /* Streamlit 기본 Status Widget / Toast 숨김 처리 */
    [data-testid="stStatusWidget"],
    .stStatusWidget,
    [data-testid="stToast"] {
        display: none !important;
    }

    /* [좌측 영역] 배경색 및 개별 테두리 제거 */
    .st-key-left_sidebar_box {
        border-radius: 12px !important;
        border: none !important;
        padding: 1.5rem !important;
        transition: background-color 0.2s ease;
    }

    .st-key-left_sidebar_box div,
    .st-key-left_sidebar_box [data-testid="stVerticalBlock"] > div {
        border: none !important;
        box-shadow: none !important;
    }

    /* 라이트 모드: 좌측 영역 배경색 */
    body[data-my-theme="light"] .st-key-left_sidebar_box {
        background-color: #dbeafe !important;
    }

    /* 다크 모드: 좌측 영역 배경색 */
    body[data-my-theme="dark"] .st-key-left_sidebar_box {
        background-color: #1e1f20 !important;
    }

    /* [우측 연차별 생존율 예측 카드 (3개 박스)] */
    div[class*="st-key-metric_card"] {
        border-radius: 12px !important;
        transition: background-color 0.2s ease, border-color 0.2s ease;
    }

    body[data-my-theme="light"] div[class*="st-key-metric_card"] {
        background-color: #eff6ff !important;
        border: 1px solid #bfdbfe !important;
    }
    body[data-my-theme="light"] div[class*="st-key-metric_card"] [data-testid="stMetricValue"] {
        color: #1d4ed8 !important;
    }

    body[data-my-theme="dark"] div[class*="st-key-metric_card"] {
        background-color: #2b2d30 !important;
        border: 1px solid #3c4043 !important;
    }
    body[data-my-theme="dark"] div[class*="st-key-metric_card"] [data-testid="stMetricValue"] {
        color: #8ab4f8 !important;
    }

    /* [우측 차트 영역 카드 (2개 박스)] */
    div[class*="st-key-chart_box"] {
        border-radius: 12px !important;
        padding: 0.5rem !important;
        transition: background-color 0.2s ease, border-color 0.2s ease;
    }

    body[data-my-theme="light"] div[class*="st-key-chart_box"] {
        background-color: #f8fafc !important;
        border: 1px solid #e2e8f0 !important;
    }

    body[data-my-theme="dark"] div[class*="st-key-chart_box"] {
        background-color: #2b2d30 !important;
        border: 1px solid #3c4043 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------
# 2. 실시간 테마 감지 JavaScript 주입
# --------------------------------------------------
components.html(
    """
    <script>
    function detectAndApplyTheme() {
        const parentDoc = window.parent.document;
        const app = parentDoc.querySelector('.stApp');
        if (!app) return;

        const bgColor = window.getComputedStyle(app).backgroundColor;
        const rgb = bgColor.match(/\\d+/g);

        if (rgb && rgb.length >= 3) {
            const r = parseInt(rgb[0]);
            const g = parseInt(rgb[1]);
            const b = parseInt(rgb[2]);
            
            const brightness = (r * 299 + g * 587 + b * 114) / 1000;
            const target = parentDoc.body;

            if (brightness < 128) {
                target.setAttribute('data-my-theme', 'dark');
            } else {
                target.setAttribute('data-my-theme', 'light');
            }
        }
    }

    detectAndApplyTheme();

    const appEl = window.parent.document.querySelector('.stApp');
    if (appEl) {
        const observer = new MutationObserver(detectAndApplyTheme);
        observer.observe(appEl, { attributes: true, childList: true, subtree: true });
    }
    </script>
    """,
    height=0,
)


# --------------------------------------------------
# 화면 레이아웃 (좌: 메뉴 / 우: 메인 결과)
# --------------------------------------------------

left, right = st.columns([0.85, 2.15], gap="large")


# ==================================================
# 좌측 - 제목 및 조건 입력 메뉴
# ==================================================

with left:
    with st.container(key="left_sidebar_box"):
        st.title("Restaurant Survival")
        st.caption("AI 기반 매장 연차별 생존율 예측")
        st.markdown("---")

        st.subheader("⚙️ 매장 조건 입력")

        try:
            with st.spinner("데이터를 불러오는 중..."):
                with ThreadPoolExecutor() as executor:
                    future_gu = executor.submit(fetch_gu)
                    future_bt = executor.submit(fetch_business_types)
                    future_range = executor.submit(fetch_area_worker_range)

                    gu_list = future_gu.result()
                    business_types = future_bt.result()
                    range_info = future_range.result()

        except requests.RequestException:
            st.error(
                "서버(BE)에 연결할 수 없습니다. "
                "FastAPI가 실행 중인지 확인해주세요."
            )
            st.stop()

        if not gu_list or not business_types:
            st.warning("DB에 데이터가 없습니다.")
            st.stop()

        min_area = float(range_info.get("min_area", 1.0))
        max_area = float(range_info.get("max_area", 1000.0))
        min_workers = int(range_info.get("min_workers", 1))     # min_workers 응답은 없음. 기본값 1
        max_workers = int(range_info.get("max_workers", 100))   # max_workers 응답은 없음. 기본값 100

        # [수정] 초기 매장 규모를 1.0으로 고정 (min_area가 1.0보다 큰 예외 상황 대비)
        default_area = max(1.0, min_area)

        # 위치
        st.markdown("**위치가 어디신가요?**")

        gu = st.selectbox(
            "구",
            gu_list,
            label_visibility="collapsed",
        )

        with st.spinner("데이터를 불러오는 중..."):
            dong_list = fetch_dong(gu)

        dong = (
            st.selectbox(
                "동",
                dong_list,
                label_visibility="collapsed",
            )
            if dong_list
            else None
        )

        st.markdown("")

        # 업종
        st.markdown("**어떤 업종인가요?**")

        business_type = st.selectbox(
            "업종",
            business_types,
            label_visibility="collapsed",
        )

        st.markdown("")

        # 매장 규모 (초기값 value를 default_area=1.0 으로 설정)
        st.markdown("**매장 규모**")
        st.caption(f"{min_area:.1f}㎡ ~ {max_area:.1f}㎡")

        area = st.number_input(
            "사업장 면적",
            min_value=min_area,
            max_value=max_area,
            value=default_area,  # [수정] 1.0 (또는 min_area가 더 크면 min_area)
            step=0.5,
            label_visibility="collapsed",
        )

        st.markdown("")

        # 함께 일할 사람
        st.markdown("**함께 일할 사람**")
        st.caption(f"{min_workers}명 ~ {max_workers}명")

        workers = st.number_input(
            "총종사자수",
            min_value=min_workers,
            max_value=max_workers,
            value=min_workers,
            step=1,
            label_visibility="collapsed",
        )

        st.markdown("---")

        predict_clicked = st.button(
            "생존율 분석하기",
            type="primary",
            use_container_width=True,
            disabled=not dong,
        )


# ==================================================
# 우측 - 생존 예측 결과 및 입력 데이터
# ==================================================

with right:
    st.subheader("📊 생존 예측 결과")

    if not predict_clicked:
        st.info(
            "👈 좌측에서 매장 조건(위치, 업종, 면적, 종사자 수)을 입력하고 "
            "**'생존율 분석하기'** 버튼을 클릭하세요."
        )

    else:
        try:
            with st.spinner("데이터를 불러오는 중..."):
                result = call_predict(
                    gu,
                    dong,
                    business_type,
                    area,
                    workers
                )

        except requests.HTTPError as e:
            st.error(
                f"예측 요청 실패: "
                f"{e.response.status_code} "
                f"{e.response.text}"
            )
            st.stop()

        except requests.RequestException as e:
            st.error(
                f"예측 요청 실패: {e}"
            )
            st.stop()

        # --------------------------------------------------
        # ① 우측 상단: 분석 입력 데이터
        # --------------------------------------------------
        st.markdown("#### 📝 분석 입력 데이터")
        
        input_data = result.get("input", {})

        target_keys = ["구", "동", "업태구분명", "소재지면적", "동종업체수", "총종사자수"]
        
        table_rows = []
        for key in target_keys:
            raw_val = input_data.get(key, "-")
            
            if key == "소재지면적" and isinstance(raw_val, (int, float)):
                formatted_val = f"{raw_val:.1f} ㎡"
            elif key == "동종업체수" and isinstance(raw_val, (int, float)):
                formatted_val = f"{raw_val:,} 개"
            elif key == "총종사자수" and isinstance(raw_val, (int, float)):
                formatted_val = f"{raw_val:,} 명"
            else:
                formatted_val = str(raw_val)

            table_rows.append({"항목": key, "설정 / 산출 값": formatted_val})

        df_summary = pd.DataFrame(table_rows)
        st.dataframe(df_summary, use_container_width=True, hide_index=True)
        st.markdown("---")

        # --------------------------------------------------
        # ② 연차별 생존율 예측 결과
        # --------------------------------------------------
        success_keys = [
            "1year",
            "3year",
            "5year",
        ]

        model_vals = [
            result["ml_success_rate"][key]
            for key in success_keys
        ]

        st.markdown("#### 🎯 연차별 생존율 예측 결과")
        year_cols = st.columns(3)

        for i, year in enumerate(YEARS):
            with year_cols[i]:
                with st.container(border=True, key=f"metric_card_{i}"):
                    st.metric(
                        label=f"{year} 생존율",
                        value=f"{model_vals[i]:.1f}%"
                    )

        st.markdown("")

        # --------------------------------------------------
        # ③ 그래프 시각화 (모델 예측 vs 연차별 데이터 평균)
        # --------------------------------------------------
        col1, col2 = st.columns(2)

        # [col1] 모델 예측 그래프
        with col1:
            with st.container(border=True, key="chart_box_1"):
                fig1 = go.Figure()

                fig1.add_trace(
                    go.Scatter(
                        x=YEARS,
                        y=model_vals,
                        mode="lines+markers+text",
                        name="모델 예측",
                        text=[f"{value:.1f}%" for value in model_vals], 
                        textposition="top center",
                    )
                )

                fig1.update_layout(
                    title=dict(
                        text="모델 예측 생존율",
                        x=0,
                        xanchor="left",
                    ),
                    yaxis=dict(
                        range=[0, 100],
                        dtick=20,
                        title="생존율(%)",
                    ),
                    xaxis_title="경과 연차",
                    height=320,
                    margin=dict(l=20, r=20, t=40, b=20),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                )

                st.plotly_chart(
                    fig1,
                    use_container_width=True,
                )

        # [col2] 연차별 데이터 평균 그래프
        db_rate_info = result.get("db_success_rate", {})
        db_vals = [db_rate_info.get(k) for k in success_keys]
        sample_count = db_rate_info.get("sample_count", 0)

        with col2:
            with st.container(border=True, key="chart_box_2"):
                fig2 = go.Figure()

                if all(v is not None for v in db_vals):
                    fig2.add_trace(
                        go.Scatter(
                            x=YEARS,
                            y=db_vals,
                            mode="lines+markers+text",
                            name="데이터 평균",
                            text=[f"{value:.1f}%" for value in db_vals], 
                            textposition="top center",
                            line=dict(color="#FF7F0E")
                        )
                    )
                    title_text = f"데이터 평균 생존율 (표본 {sample_count}개)" if sample_count else "데이터 평균 생존율"
                    fig2.update_layout(
                        title=title_text,
                        yaxis=dict(range=[0, 100], dtick=20, title="생존율(%)"),
                        xaxis_title="경과 연차",
                        height=320,
                        margin=dict(l=20, r=20, t=40, b=20),
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                    )
                else:
                    fig2.update_layout(
                        title="데이터 평균 생존율",
                        yaxis=dict(range=[0, 100], dtick=20, title="생존율(%)"),
                        xaxis_title="경과 연차",
                        height=320,
                        margin=dict(l=20, r=20, t=40, b=20),
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        annotations=[
                            dict(
                                text="해당 지역/업종의 데이터가 없습니다",
                                xref="paper",
                                yref="paper",
                                x=0.5,
                                y=0.5,
                                showarrow=False,
                                font=dict(size=14, color="gray"),
                            )
                        ],
                    )
                st.plotly_chart(fig2, use_container_width=True)