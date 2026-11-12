import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.cluster import KMeans
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.inspection import permutation_importance
from datetime import datetime

st.set_page_config(page_title="기후 연구소 | Climate Lab", layout="wide")

# ---------- 디자인 ----------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Pretendard:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Pretendard', sans-serif;
}
.stApp {
    background: #f4f7fb;
}

/* 상단 콘텐츠 잘림 방지 */
section.main > div.block-container {
    padding-top: 3.2rem !important;
    padding-bottom: 3rem !important;
    max-width: 1500px;
}
header[data-testid="stHeader"] {
    background: transparent;
}

/* 사이드바 */
[data-testid="stSidebar"] {
    background: #111827;
}
[data-testid="stSidebar"] * {
    color: #e5e7eb;
}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
    color: #cbd5e1;
}
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] {
    color: #94a3b8;
}
.sidebar-brand {
    font-size: 23px;
    font-weight: 800;
    letter-spacing: -0.5px;
    color: #f8fafc;
    margin: 8px 0 4px 0;
}
.sidebar-description {
    color: #94a3b8;
    font-size: 13px;
    margin-bottom: 22px;
}

/* 서비스형 사이드바 내비게이션 */
[data-testid="stSidebar"] [data-testid="stRadio"] > label {
    display: none !important;
}
[data-testid="stSidebar"] [data-testid="stRadio"] [role="radiogroup"] {
    display: flex;
    flex-direction: column;
    gap: 8px;
    width: 100%;
}
[data-testid="stSidebar"] [data-testid="stRadio"] [role="radiogroup"] > label {
    box-sizing: border-box;
    display: flex;
    align-items: center;
    width: 100%;
    min-height: 48px;
    padding: 0 15px;
    margin: 0;
    border: 1px solid rgba(148, 163, 184, 0.13);
    border-radius: 11px;
    background: rgba(255, 255, 255, 0.035);
    cursor: pointer;
    transition: background .18s ease, border-color .18s ease,
                transform .18s ease, box-shadow .18s ease;
}
[data-testid="stSidebar"] [data-testid="stRadio"] [role="radiogroup"] > label:hover {
    background: rgba(255, 255, 255, 0.085);
    border-color: rgba(148, 163, 184, 0.32);
    transform: translateX(2px);
}
[data-testid="stSidebar"] [data-testid="stRadio"] [role="radiogroup"] > label:has(input[type="radio"]:checked) {
    background: linear-gradient(100deg, rgba(15, 159, 154, .24), rgba(15, 159, 154, .10));
    border-color: rgba(45, 212, 191, .62);
    box-shadow: inset 3px 0 0 #2dd4bf, 0 4px 14px rgba(0, 0, 0, .12);
}
[data-testid="stSidebar"] [data-testid="stRadio"] [role="radiogroup"] > label > div:first-child {
    display: none !important;
}
[data-testid="stSidebar"] [data-testid="stRadio"] [role="radiogroup"] > label p {
    margin: 0;
    color: #cbd5e1;
    font-size: 14px;
    font-weight: 600;
    letter-spacing: -.15px;
    line-height: 1.4;
}
[data-testid="stSidebar"] [data-testid="stRadio"] [role="radiogroup"] > label:has(input[type="radio"]:checked) p {
    color: #f8fafc;
    font-weight: 750;
}
[data-testid="stSidebar"] [data-testid="stRadio"] input[type="radio"] {
    position: absolute;
    opacity: 0;
    width: 1px;
    height: 1px;
}
[data-testid="stSidebar"] [data-testid="stRadio"] [role="radiogroup"] > label:focus-within {
    outline: 2px solid rgba(45, 212, 191, .75);
    outline-offset: 2px;
}
/* 메인 화면 */
.topbar {
    background: #171c2d;
    padding: 20px 24px;
    border-radius: 14px;
    color: white;
    margin: 0 0 26px 0;
    overflow: visible;
}
.brand {
    font-size: 21px;
    font-weight: 800;
    letter-spacing: .2px;
    line-height: 1.5;
    color: #ffffff;
}
.brand span {
    color: #38bdf8;
}
.subbrand {
    color: #aab4c5;
    font-size: 13px;
    line-height: 1.6;
    margin-top: 4px;
}
.page-title {
    font-size: 27px;
    font-weight: 800;
    color: #172033;
    margin: 4px 0 3px;
}
.page-sub {
    color: #778196;
    font-size: 13px;
    margin-bottom: 20px;
}
.panel {
    background: white;
    border: 1px solid #e8edf4;
    border-radius: 14px;
    padding: 18px 20px;
}
.metric-card {
    background: white;
    border: 1px solid #e8edf4;
    border-radius: 13px;
    padding: 16px 18px;
}
.metric-label {
    color: #778196;
    font-size: 12px;
    font-weight: 600;
}
.metric-value {
    color: #172033;
    font-size: 25px;
    font-weight: 800;
    margin-top: 5px;
}
.metric-note {
    color: #16a085;
    font-size: 11px;
    margin-top: 3px;
}
.section-label {
    color: #253047;
    font-size: 15px;
    font-weight: 750;
    margin: 5px 0 12px;
}
div.stButton > button {
    border-radius: 9px;
    font-weight: 650;
}
div[data-testid="stTabs"] button {
    font-weight: 650;
}
[data-testid="stDataFrame"] {
    border-radius: 10px;
}
.small-muted {
    color: #8490a3;
    font-size: 12px;
}
</style>
""", unsafe_allow_html=True)

if "datasets" not in st.session_state:
    st.session_state.datasets = {}
if "models" not in st.session_state:
    st.session_state.models = {}
if "active_dataset" not in st.session_state:
    st.session_state.active_dataset = None
if "active_model" not in st.session_state:
    st.session_state.active_model = None
if "history" not in st.session_state:
    st.session_state.history = []

def synthetic_climate_data():
    """교육용 예시 데이터. 실제 관측 자료가 아닌 합성 데이터임."""
    rng = np.random.default_rng(42)
    years = np.arange(1980, 2025)
    rows = []
    for region, offset in [("동아시아", 0.15), ("유럽", -0.10), ("북아메리카", 0.05), ("남아시아", 0.35)]:
        for year in years:
            co2 = 338 + (year - 1980) * 2.05 + rng.normal(0, 1.7)
            temp = 14.0 + (year - 1980) * 0.025 + (co2 - 338) * 0.012 + offset + rng.normal(0, 0.16)
            rainfall = 950 + 0.8 * (year - 1980) + rng.normal(0, 65)
            humidity = 63 + rng.normal(0, 5)
            rows.append([year, region, co2, temp, rainfall, humidity])
    return pd.DataFrame(rows, columns=["연도", "지역", "이산화탄소농도(ppm)", "평균기온(℃)", "연강수량(mm)", "평균습도(%)"])

def current_df():
    name = st.session_state.active_dataset
    return st.session_state.datasets.get(name) if name else None

def numeric_columns(df):
    return df.select_dtypes(include=np.number).columns.tolist()

def model_pipeline(model_name, X):
    numeric = X.select_dtypes(include=np.number).columns.tolist()
    categorical = X.select_dtypes(exclude=np.number).columns.tolist()
    transformers = []

    if numeric:
        numeric_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler())
        ])
        transformers.append(("num", numeric_pipeline, numeric))

    if categorical:
        categorical_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ])
        transformers.append(("cat", categorical_pipeline, categorical))

    if not transformers:
        raise ValueError("학습에 사용할 수 있는 숫자형 또는 범주형 변수가 없습니다.")

    prep = ColumnTransformer(transformers, remainder="drop")
    models = {
        "선형 회귀": LinearRegression(),
        "의사결정나무": DecisionTreeRegressor(max_depth=6, random_state=42),
        "랜덤 포레스트": RandomForestRegressor(n_estimators=160, max_depth=None, random_state=42),
        "그래디언트 부스팅": GradientBoostingRegressor(random_state=42),
        "K-최근접 이웃": KNeighborsRegressor(n_neighbors=5)
    }
    return Pipeline([("전처리", prep), ("모델", models[model_name])])

# ---------- 사이드바 ----------
with st.sidebar:
    st.markdown('<div class="sidebar-brand">기후 연구소</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-description">기후 데이터 지능형 분석 시스템</div>', unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("작업 공간")
    page = st.radio(
        " ",
        ["대시보드", "데이터셋", "모델 학습", "테스트 · 시각화", "군집 분석", "실험 보고서"],
        label_visibility="collapsed"
    )
    st.markdown("---")
    st.markdown("**등록된 데이터셋**")
    if st.session_state.datasets:
        for n in st.session_state.datasets:
            st.caption("▦ " + n)
    else:
        st.caption("아직 등록된 데이터가 없습니다.")
    st.markdown("---")
    st.caption("버전 1.0 · 동아리 탐구 프로젝트")

# ---------- 상단 브랜드 ----------
st.markdown("""
<div class="topbar">
  <div class="brand">기후<span>연구소</span> <span style="color:#aab4c5;font-size:12px;font-weight:500;"> / 데이터 분석 플랫폼</span></div>
  <div class="subbrand">데이터를 탐색하고, 모델을 학습시키고, 패턴을 발견하세요.</div>
</div>
""", unsafe_allow_html=True)

# ---------- 대시보드 ----------
if page == "대시보드":
    st.markdown('<div class="page-title">연구 대시보드</div><div class="page-sub">기후 데이터와 머신러닝 실험을 한곳에서 관리합니다.</div>', unsafe_allow_html=True)
    a,b,c,d = st.columns(4)
    with a:
        st.markdown(f'<div class="metric-card"><div class="metric-label">등록된 데이터셋</div><div class="metric-value">{len(st.session_state.datasets)}</div><div class="metric-note">데이터 저장소</div></div>', unsafe_allow_html=True)
    with b:
        st.markdown(f'<div class="metric-card"><div class="metric-label">생성된 모델</div><div class="metric-value">{len(st.session_state.models)}</div><div class="metric-note">학습 모델</div></div>', unsafe_allow_html=True)
    with c:
        st.markdown(f'<div class="metric-card"><div class="metric-label">완료된 실험</div><div class="metric-value">{len(st.session_state.history)}</div><div class="metric-note">실험 기록</div></div>', unsafe_allow_html=True)
    with d:
        st.markdown(f'<div class="metric-card"><div class="metric-label">현재 작업</div><div class="metric-value" style="font-size:18px;">{st.session_state.active_dataset or "없음"}</div><div class="metric-note">활성 데이터셋</div></div>', unsafe_allow_html=True)
    st.write("")
    left, right = st.columns([1.45, 1])
    with left:
        st.markdown("**새로운 데이터로 분석 시작하기**")
        st.write("기후 관련 CSV 파일을 업로드하거나 교육용 예시 데이터를 불러오세요.")
        if st.button("교육용 기후 데이터 불러오기", type="primary", use_container_width=True):
            st.session_state.datasets["기후 변화 예시 데이터"] = synthetic_climate_data()
            st.session_state.active_dataset = "기후 변화 예시 데이터"
            st.success("교육용 합성 데이터를 불러왔습니다. 실제 관측 데이터가 아닌 실습용 데이터입니다.")
            st.rerun()
    with right:
        st.markdown('<div class="section-label">탐구 진행 순서</div>', unsafe_allow_html=True)
        st.markdown('<div class="panel">1. 공개 데이터 수집 및 업로드<br><br>2. 변수 선택과 데이터 탐색<br><br>3. 머신러닝 모델 학습<br><br>4. 예측 결과 및 오차 분석<br><br>5. 실험 결과 비교 및 보고서 작성</div>', unsafe_allow_html=True)
    st.write("")
    st.markdown('<div class="section-label">최근 실험 기록</div>', unsafe_allow_html=True)
    if st.session_state.history:
        st.dataframe(pd.DataFrame(st.session_state.history).tail(8), use_container_width=True, hide_index=True)
    else:
        st.info("아직 실험 기록이 없습니다. 데이터셋을 불러온 뒤 모델 학습을 진행해 보세요.")

# ---------- 데이터셋 ----------
elif page == "데이터셋":
    st.markdown('<div class="page-title">데이터셋 관리</div><div class="page-sub">공개 기후 데이터를 업로드하고 분석 가능한 형태로 확인합니다.</div>', unsafe_allow_html=True)
    up_col, info_col = st.columns([1.2, 1])
    with up_col:
        st.markdown('<div class="section-label">데이터 가져오기</div>', unsafe_allow_html=True)
        uploaded = st.file_uploader("CSV 파일 선택", type=["csv"], help="첫 번째 행은 열 이름이어야 합니다.")
        dataset_name = st.text_input("데이터셋 이름", value=(uploaded.name.rsplit(".",1)[0] if uploaded else "새 기후 데이터"))
        if uploaded is not None and st.button("데이터셋 등록", type="primary"):
            try:
                df = pd.read_csv(uploaded)
                st.session_state.datasets[dataset_name] = df
                st.session_state.active_dataset = dataset_name
                st.success(f"{len(df):,}개 행의 데이터셋을 등록했습니다.")
                st.rerun()
            except Exception as e:
                st.error(f"CSV를 읽을 수 없습니다: {e}")
        if st.button("교육용 예시 데이터 추가"):
            st.session_state.datasets["기후 변화 예시 데이터"] = synthetic_climate_data()
            st.session_state.active_dataset = "기후 변화 예시 데이터"
            st.rerun()
    with info_col:
        st.markdown('<div class="section-label">데이터 출처 추천</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="panel">
        <b>NASA 기후 자료</b><br><a href="https://climate.nasa.gov/vital-signs/global-temperature/" target="_blank">전 지구 기온 변화</a><br><br>
        <b>Our World in Data</b><br><a href="https://ourworldindata.org/co2-and-greenhouse-gas-emissions" target="_blank">온실가스 및 CO₂ 데이터</a><br><br>
        <b>World Bank</b><br><a href="https://data.worldbank.org/" target="_blank">국가별 기후·경제 지표</a>
        </div>
        """, unsafe_allow_html=True)
    if st.session_state.datasets:
        selected = st.selectbox("현재 분석할 데이터셋", list(st.session_state.datasets.keys()),
                                index=list(st.session_state.datasets.keys()).index(st.session_state.active_dataset) if st.session_state.active_dataset in st.session_state.datasets else 0)
        st.session_state.active_dataset = selected
        df = current_df()
        st.markdown('<div class="section-label">데이터 미리보기</div>', unsafe_allow_html=True)
        m1,m2,m3 = st.columns(3)
        m1.metric("전체 행", f"{len(df):,}")
        m2.metric("변수 수", len(df.columns))
        m3.metric("결측값", f"{int(df.isna().sum().sum()):,}")
        st.dataframe(df.head(100), use_container_width=True, height=300)
        st.markdown('<div class="section-label">기초 통계</div>', unsafe_allow_html=True)
        st.dataframe(df.describe(include="all").transpose(), use_container_width=True)
        st.markdown('<div class="section-label">결측값 현황</div>', unsafe_allow_html=True)
        miss = df.isna().sum().reset_index()
        miss.columns = ["변수", "결측값 수"]
        fig = px.bar(miss, x="변수", y="결측값 수", color="결측값 수", color_continuous_scale="Teal")
        fig.update_layout(template="plotly_white", height=300, margin=dict(l=10,r=10,t=20,b=10))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("먼저 CSV 파일을 업로드하거나 교육용 예시 데이터를 추가하세요.")

# ---------- 모델 학습 ----------
elif page == "모델 학습":
    st.markdown('<div class="page-title">모델 학습실</div><div class="page-sub">분석 변수와 알고리즘을 선택하고 학습 조건을 설정합니다.</div>', unsafe_allow_html=True)
    df = current_df()
    if df is None:
        st.warning("먼저 데이터셋 메뉴에서 데이터를 등록해 주세요.")
    else:
        st.markdown('<div class="section-label">① 학습 데이터 설정</div>', unsafe_allow_html=True)
        ncols = numeric_columns(df)
        if len(ncols) < 2:
            st.error("회귀 학습을 위해 숫자형 변수가 최소 2개 필요합니다.")
        else:
            c1,c2,c3 = st.columns(3)
            with c1:
                target = st.selectbox("예측할 변수 (목표값)", ncols, index=(ncols.index("평균기온(℃)") if "평균기온(℃)" in ncols else len(ncols)-1))
            with c2:
                features = st.multiselect("학습에 사용할 변수", [x for x in df.columns if x != target],
                                          default=[x for x in ncols if x != target][:3])
            with c3:
                test_size = st.slider("테스트 데이터 비율 (%)", 10, 40, 20, 5)
            st.markdown('<div class="section-label">② 알고리즘 선택</div>', unsafe_allow_html=True)
            algorithm = st.selectbox("학습 알고리즘", ["선형 회귀", "의사결정나무", "랜덤 포레스트", "그래디언트 부스팅", "K-최근접 이웃"])
            c4,c5 = st.columns(2)
            with c4:
                train_ratio = st.slider("학습 데이터 사용 비율 (%)", 10, 100, 100, 10)
            with c5:
                noise_pct = st.slider("학습 데이터 오류 추가 (%)", 0, 30, 0, 5)
            time_order = st.checkbox("시간 순서에 따라 학습/테스트 분리 (연도별 데이터에 권장)", value=("연도" in df.columns))
            model_name = st.text_input("모델 이름", value=f"모델 {len(st.session_state.models)+1:02d}")
            st.caption("주의: 학습 데이터 비율과 테스트 데이터 비율은 서로 다른 설정입니다. 학습 데이터 비율은 분리된 학습 세트 중 실제 사용할 비율입니다.")
            if st.button("모델 학습 시작", type="primary", use_container_width=True):
                if not features:
                    st.error("학습에 사용할 변수를 하나 이상 선택하세요.")
                else:
                    work = df[features + [target]].dropna(subset=[target]).copy()
                    X = work[features]
                    y = work[target]
                    if time_order and "연도" in work.columns:
                        order = np.argsort(work["연도"].values)
                        X, y = X.iloc[order], y.iloc[order]
                        split = max(1, int(len(X) * (1-test_size/100)))
                        X_train, X_test = X.iloc[:split], X.iloc[split:]
                        y_train, y_test = y.iloc[:split], y.iloc[split:]
                    else:
                        X_train, X_test, y_train, y_test = train_test_split(
                            X, y, test_size=test_size/100, random_state=42
                        )
                    take = max(2, int(len(X_train) * train_ratio/100))
                    X_train, y_train = X_train.iloc[:take], y_train.iloc[:take]
                    if len(X_train) < 2 or len(X_test) < 1:
                        st.error("학습 또는 테스트 데이터가 부족합니다.")
                    else:
                        y_fit = y_train.copy()
                        if noise_pct > 0:
                            rng = np.random.default_rng(42)
                            count = max(1, int(len(y_fit)*noise_pct/100))
                            ids = rng.choice(y_fit.index, size=min(count,len(y_fit)), replace=False)
                            spread = float(y_fit.std()) if pd.notna(y_fit.std()) else 1.0
                            y_fit.loc[ids] += rng.normal(0, max(spread*0.5, 0.001), len(ids))
                        try:
                            model = model_pipeline(algorithm, X_train)
                            model.fit(X_train, y_fit)
                            pred = model.predict(X_test)
                            mae = mean_absolute_error(y_test, pred)
                            rmse = np.sqrt(mean_squared_error(y_test, pred))
                            r2 = r2_score(y_test, pred) if len(y_test) > 1 else np.nan
                            record = {"이름": model_name, "알고리즘": algorithm, "예측 변수": target,
                                      "학습 행": len(X_train), "테스트 행": len(X_test),
                                      "학습 데이터 비율(%)": train_ratio, "오류 추가(%)": noise_pct,
                                      "MAE": round(mae,4), "RMSE": round(rmse,4),
                                      "R²": round(r2,4) if pd.notna(r2) else None,
                                      "모델": model, "특징": features, "테스트 X": X_test,
                                      "테스트 y": y_test, "예측값": pred, "생성 시각": datetime.now().strftime("%Y-%m-%d %H:%M")}
                            st.session_state.models[model_name] = record
                            st.session_state.active_model = model_name
                            st.session_state.history.append({k:v for k,v in record.items() if k not in ["모델","특징","테스트 X","테스트 y","예측값"]})
                            st.success(f"{model_name} 학습이 완료되었습니다.")
                        except Exception as e:
                            st.error(f"학습 중 오류가 발생했습니다: {e}")
            if st.session_state.models:
                st.markdown('<div class="section-label">생성된 모델 목록</div>', unsafe_allow_html=True)
                overview = [{k:v for k,v in m.items() if k in ["이름","알고리즘","예측 변수","학습 행","MAE","RMSE","R²"]} for m in st.session_state.models.values()]
                st.dataframe(pd.DataFrame(overview), use_container_width=True, hide_index=True)

# ---------- 테스트 및 시각화 ----------
elif page == "테스트 · 시각화":
    st.markdown('<div class="page-title">테스트 및 시각화</div><div class="page-sub">실제 관측값과 AI 예측값을 비교하고 학습 패턴을 분석합니다.</div>', unsafe_allow_html=True)
    if not st.session_state.models:
        st.info("먼저 모델 학습실에서 모델을 학습시켜 주세요.")
    else:
        names = list(st.session_state.models.keys())
        selected = st.selectbox("분석할 모델 선택", names)
        m = st.session_state.models[selected]
        a,b,c = st.columns(3)
        a.metric("평균 절대 오차 (MAE)", f"{m['MAE']:.4f}")
        b.metric("평균 제곱근 오차 (RMSE)", f"{m['RMSE']:.4f}")
        c.metric("결정계수 (R²)", "계산 불가" if m["R²"] is None else f"{m['R²']:.4f}")
        y_true = np.asarray(m["테스트 y"])
        y_pred = np.asarray(m["예측값"])
        comp = pd.DataFrame({"실제값": y_true, "예측값": y_pred})
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=comp["실제값"], y=comp["예측값"], mode="markers", name="예측 결과",
                                 marker=dict(color="#0f9f9a", size=9, opacity=.75)))
        lo = float(min(comp.min()))
        hi = float(max(comp.max()))
        fig.add_trace(go.Scatter(x=[lo,hi], y=[lo,hi], mode="lines", name="완벽한 예측 기준선",
                                 line=dict(color="#f08a5d", dash="dash")))
        fig.update_layout(title="실제값과 예측값 비교", xaxis_title="실제값", yaxis_title="AI 예측값",
                          template="plotly_white", height=420)
        st.plotly_chart(fig, use_container_width=True)
        residual = y_true - y_pred
        col1,col2 = st.columns(2)
        with col1:
            fig2 = px.histogram(x=residual, nbins=25, title="예측 오차 분포", labels={"x":"오차 (실제값 - 예측값)"})
            fig2.update_layout(template="plotly_white", height=330)
            st.plotly_chart(fig2, use_container_width=True)
        with col2:
            err = pd.DataFrame({"실제값":y_true, "오차":residual})
            fig3 = px.scatter(err, x="실제값", y="오차", title="실제값에 따른 오차 패턴")
            fig3.add_hline(y=0, line_dash="dash", line_color="#f08a5d")
            fig3.update_layout(template="plotly_white", height=330)
            st.plotly_chart(fig3, use_container_width=True)
        st.markdown("**테스트 데이터 상세 결과**")
        st.dataframe(comp.assign(오차=residual), use_container_width=True)

# ---------- 군집 분석 ----------
elif page == "군집 분석":
    st.markdown('<div class="page-title">군집 분석실</div><div class="page-sub">비슷한 기후 특성을 가진 데이터끼리 자동으로 그룹화합니다.</div>', unsafe_allow_html=True)
    df = current_df()
    if df is None:
        st.warning("먼저 데이터셋을 등록해 주세요.")
    else:
        nums = numeric_columns(df)
        if len(nums) < 2:
            st.warning("군집 분석에는 숫자형 변수가 두 개 이상 필요합니다.")
        else:
            selected_cols = st.multiselect("군집 분석에 사용할 변수", nums, default=nums[:min(3,len(nums))])
            k = st.slider("군집 개수", 2, 8, 3)
            if len(selected_cols) >= 2 and st.button("군집 분석 실행", type="primary"):
                data = df[selected_cols].replace([np.inf,-np.inf], np.nan)
                data = data.fillna(data.median(numeric_only=True)).dropna()
                scaler = StandardScaler()
                scaled = scaler.fit_transform(data)
                km = KMeans(n_clusters=k, random_state=42, n_init=10)
                labels = km.fit_predict(scaled)
                result = data.copy()
                result["군집"] = labels.astype(str)
                st.session_state.cluster_result = result
                st.session_state.cluster_cols = selected_cols
            if "cluster_result" in st.session_state:
                result = st.session_state.cluster_result
                cols = st.session_state.cluster_cols
                fig = px.scatter(result, x=cols[0], y=cols[1], color="군집",
                                 title="기후 데이터 군집 분포", color_discrete_sequence=px.colors.qualitative.Set2)
                fig.update_layout(template="plotly_white", height=450)
                st.plotly_chart(fig, use_container_width=True)
                st.dataframe(result, use_container_width=True)

# ---------- 보고서 ----------
elif page == "실험 보고서":
    st.markdown('<div class="page-title">실험 보고서</div><div class="page-sub">학습 조건과 평가 결과를 모아 실험 과정을 정리합니다.</div>', unsafe_allow_html=True)
    if not st.session_state.history:
        st.info("저장된 실험이 없습니다. 모델 학습을 먼저 진행하세요.")
    else:
        hist = pd.DataFrame(st.session_state.history)
        st.markdown('<div class="section-label">실험 기록</div>', unsafe_allow_html=True)
        st.dataframe(hist, use_container_width=True, hide_index=True)
        st.markdown('<div class="section-label">모델별 성능 비교</div>', unsafe_allow_html=True)
        metric = st.selectbox("비교 지표", ["MAE","RMSE","R²"])
        plot_df = hist.dropna(subset=[metric])
        if not plot_df.empty:
            fig = px.bar(plot_df, x="이름", y=metric, color="알고리즘", text_auto=".3f",
                         title=f"실험별 {metric} 비교")
            fig.update_layout(template="plotly_white", height=400)
            st.plotly_chart(fig, use_container_width=True)
        csv = hist.to_csv(index=False).encode("utf-8-sig")
        st.download_button("실험 기록 CSV 다운로드", csv, "기후연구소_실험기록.csv", "text/csv")
        st.caption("MAE와 RMSE는 낮을수록, R²는 일반적으로 1에 가까울수록 예측 성능이 좋습니다. 단, 서로 다른 테스트 데이터의 점수는 직접 비교할 때 주의해야 합니다.")

st.markdown("---")
st.caption("기후 연구소 · 교육 및 탐구 목적의 분석 도구 | 예측 결과는 데이터의 품질과 범위에 따라 달라집니다.")
