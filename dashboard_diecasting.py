"""
다이캐스팅 품질 관리 대시보드
공정 운영 기준 설계 프로세스 기반
- 실시간 판정 (4단계 로직)
- SHAP 해석 (변수 기여도)
- KPI 모니터링 (4종)
- Risk Zone 분류
- Product Type별 분리 운영

실행: streamlit run dashboard_diecasting.py
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR.parent / "data"

MODEL_NAME = "RandomForest"

SHAP_RESULT_PATH = DATA_DIR / "diecasting_rf_shap_top10.csv"

# 페이지 설정

st.set_page_config(
    page_title="다이캐스팅 품질 관리 대시보드",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 커스텀 CSS
st.markdown("""
<style>
    .main .block-container { padding-top: 1rem; max-width: 1400px; }
    .metric-card {
        background: #f8f9fa; border-radius: 10px; padding: 1rem;
        border-left: 4px solid #4A7FB5; margin-bottom: 0.5rem;
    }
    .risk-high { background: #fff0f0; border-left-color: #D94040; }
    .risk-caution { background: #fdf5e6; border-left-color: #E09F30; }
    .risk-stable { background: #e8f8f0; border-left-color: #3A9E6E; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        background-color: #f0f2f6; border-radius: 6px;
        padding: 8px 20px; font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background-color: #4A7FB5; color: white;
    }
    [data-testid="stFileUploaderDropzone"] {
        display: flex;
        flex-direction: column;
        align-items: flex-start;
    }
    [data-testid="stFileUploaderDropzone"] button {
        order: 1;
        margin-bottom: 0.75rem;
    }
    [data-testid="stFileUploaderDropzone"] button,
    [data-testid="stFileUploaderDropzone"] button * {
        color: transparent !important;
        font-size: 0 !important;
    }
    [data-testid="stFileUploaderDropzone"] button::after {
        content: "Upload";
        color: #31333f !important;
        font-size: 1rem !important;
    }
    [data-testid="stFileUploaderDropzoneInstructions"] {
        order: 2;
    }
    [data-testid="stFileUploaderDropzoneInstructions"],
    [data-testid="stFileUploaderDropzoneInstructions"] * {
        color: transparent !important;
        font-size: 0 !important;
        line-height: 0 !important;
        margin: 0 !important;
        min-height: 0 !important;
    }
    [data-testid="stFileUploaderDropzoneInstructions"]::after {
        content: "200MB per file • CSV";
        color: #8a8f98 !important;
        display: block;
        font-size: 0.875rem !important;
        line-height: 1.4 !important;
    }
</style>
""", unsafe_allow_html=True)



# 데이터 로드 & 전처리

@st.cache_data
def load_data(path):
    df = pd.read_csv(path, header=[0, 1])

    # 멀티인덱스 컬럼 공백 정리
    df.columns = pd.MultiIndex.from_tuples(
        [(c[0].strip(), c[1].strip()) for c in df.columns]
    )

    # 독립변수
    proc_cols = [c for c in df.columns if c[0] == 'Process']
    sens_cols = [c for c in df.columns if c[0] == 'Sensor']
    def_cols  = [c for c in df.columns if c[0] == 'Defects']

    X = df[proc_cols + sens_cols].copy()
    X.columns = [f"{c[0].strip()}|{c[1].strip()}" for c in X.columns]

    # 종속변수 그룹화
    group_map = {
        '충전불량': ['Short_Shot'],
        '기포/내부': ['Blow_Hole', 'Bubble'],
        '표면손상': ['Burning_Mark', 'Dent', 'Exfoliation', 'Scratch', 'Stain'],
        '기타': ['Crack', 'Deformation', 'Contamination', 'Impurity', 'Inclusions'],
    }

    Y = pd.DataFrame(index=df.index)
    for grp, keywords in group_map.items():
        matched = [c for c in def_cols if any(k in c[1] for k in keywords)]
        Y[grp] = df[matched].max(axis=1).fillna(0).astype(int)

    # 메타정보
    product_type = df[('Process', 'Product_Type')].astype(int)

    # id, Shot 제거 + 상수 컬럼 제거
    drop_cols = [c for c in X.columns if any(k in c for k in ['|id', '|Shot'])]
    X_clean = X.drop(columns=drop_cols, errors='ignore')

    # 상수 컬럼 제거 (std == 0)
    const_cols = X_clean.columns[X_clean.std() == 0].tolist()
    X_clean = X_clean.drop(columns=const_cols, errors='ignore')

    return df, X, X_clean, Y, product_type


# RandomForest SHAP 상위 10 변수
@st.cache_data
def load_shap_top10(path=SHAP_RESULT_PATH):
    try:
        shap_df = pd.read_csv(path)
        shap_df = shap_df[shap_df["model"] == MODEL_NAME].copy()
        shap_df = shap_df.sort_values("importance", ascending=False).head(10)
        return list(shap_df[["feature", "importance"]].itertuples(index=False, name=None))
    except Exception:
        return [
            ('Process|Clamping_Force', 0.034746),
            ('Process|Rapid_Rise_Time', 0.034430),
            ('Sensor|Melting_Furnace_Temp', 0.031692),
            ('Process|Pressure_Rise_Time', 0.031053),
            ('Process|Cycle_Time', 0.027996),
            ('Process|Spray_Time', 0.027119),
            ('Process|Biscuit_Thickness', 0.026888),
            ('Process|Velocity_1', 0.026823),
            ('Process|High_Velocity', 0.025695),
            ('Process|Casting_Pressure', 0.025125),
        ]


SHAP_TOP10 = load_shap_top10()
SHAP_FEATURES = [f for f, _ in SHAP_TOP10]


# 사이드바

with st.sidebar:
    st.title("🏭 다이캐스팅 품질 관리")
    st.caption("공정 운영 기준 설계 기반 대시보드")

    uploaded = st.file_uploader("CSV 파일 업로드", type=['csv'])

    if uploaded:
        path = uploaded
    else:
        path = DATA_DIR / "DieCasting_Quality_Raw_Data.csv"

    st.divider()
    st.markdown("**불량 그룹 선택**")
    target_group = st.selectbox(
        "분석 대상", ['충전불량', '기포/내부', '표면손상'],
        label_visibility='collapsed'
    )

    st.divider()
    st.markdown("**Risk Zone 기준**")
    th_high = st.slider("High Risk (불량확률 ≥)", 0.3, 0.9, 0.6, 0.05)
    th_caution = st.slider("Caution (불량확률 ≥)", 0.1, 0.5, 0.3, 0.05)

    st.divider()
    st.markdown("**KPI 경고 기준**")
    cv_threshold = st.number_input("공정 변수 CV 경고", value=0.10, step=0.01, format="%.2f")
    defect_target = st.number_input("목표 불량률 (%)", value=15.0, step=1.0, format="%.1f")
    escape_threshold = st.number_input("이탈률 경고 (%)", value=15.0, step=1.0, format="%.1f")


# 데이터 로드
try:
    df_raw, X_all, X_clean, Y_group, product_type = load_data(path)
except Exception as e:
    st.error(f"CSV 로드 실패: {e}")
    st.info("사이드바에서 DieCasting_Quality_Raw_Data.csv를 업로드하세요.")
    st.stop()


# 정상 기준값 산출 (Type별)
@st.cache_data
def compute_normal_stats(_X, _Y, _pt, group, features):
    """Type별 정상 샘플의 중앙값/IQR 산출"""
    stats = {}
    for t in [1, 2]:
        mask = (_pt == t) & (_Y[group] == 0)
        sub = _X.loc[mask, features]
        stats[t] = {
            'median': sub.median(),
            'q1': sub.quantile(0.25),
            'q3': sub.quantile(0.75),
            'std': sub.std(),
            'mean': sub.mean(),
            'n': mask.sum()
        }
    # 전체
    mask_all = _Y[group] == 0
    sub_all = _X.loc[mask_all, features]
    stats['all'] = {
        'median': sub_all.median(),
        'q1': sub_all.quantile(0.25),
        'q3': sub_all.quantile(0.75),
        'std': sub_all.std(),
        'mean': sub_all.mean(),
        'n': mask_all.sum()
    }
    return stats


available_features = [f for f in SHAP_FEATURES if f in X_all.columns]
normal_stats = compute_normal_stats(X_all, Y_group, product_type, target_group, available_features)


# 4단계 판정 함수 (RandomForest SHAP 상위 변수 기반)
def diagnose_sample(sample_row, pt, stats, features, top_n=5):
    """
    RandomForest SHAP 상위 변수와 통계 기준을 함께 사용하는 판정 로직
    - 정상 중앙값 대비 편차 기반으로 위험도 산출
    """
    ref = stats[pt]
    deviations = []

    for feat in features:
        val = sample_row.get(feat, np.nan)
        if pd.isna(val):
            continue
        med = ref['median'][feat]
        q1 = ref['q1'][feat]
        q3 = ref['q3'][feat]
        iqr = q3 - q1

        if med != 0:
            pct_dev = (val - med) / abs(med) * 100
        else:
            pct_dev = 0.0

        # IQR 기반 이탈 점수
        if iqr > 0:
            if val < q1:
                iqr_score = (q1 - val) / iqr
            elif val > q3:
                iqr_score = (val - q3) / iqr
            else:
                iqr_score = 0.0
        else:
            iqr_score = abs(pct_dev) / 10

        deviations.append({
            'feature': feat,
            'name': feat.replace('Process|', '').replace('Sensor|', ''),
            'current': val,
            'normal_median': med,
            'q1': q1, 'q3': q3,
            'pct_dev': pct_dev,
            'iqr_score': iqr_score,
        })

    dev_df = pd.DataFrame(deviations).sort_values('iqr_score', ascending=False)

    # 불량 확률 근사 (RandomForest SHAP 상위 변수의 IQR 이탈 점수 기반)
    total_score = dev_df['iqr_score'].sum()
    prob = min(1.0, total_score / (len(features) * 1.5))  # 정규화

    # 판정
    top = dev_df.head(top_n).copy()
    top['risk'] = top.apply(lambda r:
        '⛔ 위험' if r['iqr_score'] > 1.0 and abs(r['pct_dev']) > 10 else
        ('⚠️ 주의' if r['iqr_score'] > 0.5 else '✅ 정상'), axis=1)

    return prob, top, dev_df


# 메인 탭 구성
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 공정 품질 종합 판정", "🔍 개별 제품 판정", "📈 KPI 모니터링", "📎 참고: Type별 분리 근거"
])


# TAB 1: 공정 품질 종합 판정 (Main Product)
with tab1:
    st.header("공정 품질 종합 판정")
    st.caption("이 화면 하나로 전체 공정 상황을 파악할 수 있도록 설계했습니다")

    # 상단 KPI 카드
    total = len(X_all)
    defect_count = Y_group[target_group].sum()
    defect_rate = defect_count / total * 100
    type1_count = (product_type == 1).sum()
    type2_count = (product_type == 2).sum()

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("총 생산 이력", f"{total:,}건")
    c2.metric("불량 건수", f"{defect_count:,}건", f"{defect_rate:.1f}%")
    c3.metric("Type 1", f"{type1_count:,}건", f"{(product_type==1).sum()/total*100:.1f}%")
    c4.metric("Type 2", f"{type2_count:,}건", f"{(product_type==2).sum()/total*100:.1f}%")

    # Type별 불량률
    t1_def = Y_group.loc[product_type == 1, target_group].mean() * 100
    t2_def = Y_group.loc[product_type == 2, target_group].mean() * 100
    c5.metric("Type별 불량률", f"T1:{t1_def:.1f}% / T2:{t2_def:.1f}%")

    st.divider()

    # SHAP 변수 중요도 + Risk Zone 분포
    col_left, col_right = st.columns([1.2, 1])

    with col_left:
        st.subheader("SHAP 변수 중요도 (mean |SHAP|)")

        shap_df = pd.DataFrame(SHAP_TOP10, columns=['feature', 'importance'])
        shap_df['name'] = shap_df['feature'].str.replace('Process|', '', regex=False).str.replace('Sensor|', '', regex=False)
        shap_df = shap_df.sort_values('importance', ascending=True)

        fig_shap = go.Figure(go.Bar(
            x=shap_df['importance'],
            y=shap_df['name'],
            orientation='h',
            marker_color=['#D94040' if v > 0.5 else '#4A7FB5' for v in shap_df['importance']],
            text=[f'{v:.3f}' for v in shap_df['importance']],
            textposition='outside'
        ))
        fig_shap.update_layout(
            height=400, margin=dict(l=0, r=40, t=10, b=0),
            xaxis_title='mean |SHAP|', yaxis_title='',
            plot_bgcolor='rgba(0,0,0,0)',
        )
        st.plotly_chart(fig_shap, use_container_width=True)

    with col_right:
        st.subheader("Risk Zone 분포")

        # 각 샘플의 위험도 근사 계산 (벡터화)
        @st.cache_data
        def compute_all_risks(_X, _pt, _stats, features):
            scores = pd.Series(0.0, index=_X.index)
            for t in [1, 2]:
                mask_t = _pt == t
                ref = _stats[t]
                for feat in features:
                    if feat not in _X.columns:
                        continue
                    vals = _X.loc[mask_t, feat]
                    q1, q3 = ref['q1'][feat], ref['q3'][feat]
                    iqr = q3 - q1
                    if iqr > 0:
                        below = (q1 - vals).clip(lower=0) / iqr
                        above = (vals - q3).clip(lower=0) / iqr
                        scores.loc[mask_t] += below + above
                    else:
                        med = ref['median'][feat]
                        if med != 0:
                            scores.loc[mask_t] += (abs(vals - med) / abs(med) * 100) / 10
            return (scores / (len(features) * 1.5)).clip(upper=1.0)

        risks = compute_all_risks(X_all, product_type, normal_stats, available_features)
        risk_sr = risks
        n_high = (risk_sr >= th_high).sum()
        n_caution = ((risk_sr >= th_caution) & (risk_sr < th_high)).sum()
        n_stable = (risk_sr < th_caution).sum()

        fig_zone = go.Figure(go.Pie(
            labels=['Stable', 'Caution', 'High Risk'],
            values=[n_stable, n_caution, n_high],
            marker_colors=['#3A9E6E', '#E09F30', '#D94040'],
            hole=0.45,
            textinfo='label+percent',
            textfont_size=14,
        ))
        fig_zone.update_layout(height=400, margin=dict(l=20, r=20, t=10, b=10))
        st.plotly_chart(fig_zone, use_container_width=True)

        st.caption(f"Stable: 불량확률 < {th_caution:.0%} | Caution: {th_caution:.0%}~{th_high:.0%} | High Risk: ≥ {th_high:.0%}")

    # 그룹별 불량 현황
    st.divider()
    st.subheader("불량 그룹별 현황")
    grp_stats = []
    for g in ['충전불량', '기포/내부', '표면손상', '기타']:
        cnt = Y_group[g].sum()
        rate = cnt / total * 100
        grp_stats.append({'그룹': g, '불량 건수': cnt, '불량률(%)': round(rate, 1),
                          '정상 건수': total - cnt})

    grp_df = pd.DataFrame(grp_stats)
    fig_grp = go.Figure()
    fig_grp.add_trace(go.Bar(name='불량', x=grp_df['그룹'], y=grp_df['불량 건수'],
                              marker_color='#D94040'))
    fig_grp.add_trace(go.Bar(name='정상', x=grp_df['그룹'], y=grp_df['정상 건수'],
                              marker_color='#4A7FB5'))
    fig_grp.update_layout(barmode='stack', height=300,
                           margin=dict(l=0, r=0, t=30, b=0),
                           legend=dict(orientation='h', y=1.1))
    st.plotly_chart(fig_grp, use_container_width=True)



# TAB 2: 개별 제품 판정
with tab2:
    st.header("개별 제품 판정 — 4단계 로직")
    st.caption("제품 하나하나에 대해 어떤 값이 판정에 영향을 줬는지 분해해서 확인합니다")

    col_sel, col_info = st.columns([1, 2])

    with col_sel:
        sample_idx = st.number_input("샘플 인덱스", 0, len(X_all)-1, 0, step=1)
        pt = product_type.iloc[sample_idx]
        actual_label = Y_group.iloc[sample_idx][target_group]

    sample = X_all.iloc[sample_idx]
    prob, top_feats, all_devs = diagnose_sample(sample, pt, normal_stats, available_features)

    with col_info:
        # 판정 결과 표시
        if prob >= th_high:
            zone = "⛔ High Risk"
            zone_color = "#D94040"
        elif prob >= th_caution:
            zone = "⚠️ Caution"
            zone_color = "#E09F30"
        else:
            zone = "✅ Stable"
            zone_color = "#3A9E6E"

        st.markdown(f"""
        <div style="background: linear-gradient(135deg, {zone_color}15, {zone_color}05);
                    border: 2px solid {zone_color}; border-radius: 12px; padding: 1.2rem; margin-bottom: 1rem;">
            <h3 style="margin:0; color:{zone_color};">{zone}</h3>
            <p style="margin:0.5rem 0 0 0; font-size:1.1rem;">
                불량 확률: <b>{prob:.1%}</b> &nbsp;|&nbsp;
                실제 라벨: <b>{'불량' if actual_label == 1 else '정상'}</b> &nbsp;|&nbsp;
                Product Type: <b>{pt}</b>
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    # 정상 범위 이탈 변수 테이블 + 시각화
    col_table, col_chart = st.columns([1.2, 1])

    with col_table:
        st.subheader("정상 범위 이탈 변수 Top 5")
        display_df = top_feats[['name', 'normal_median', 'current', 'pct_dev', 'risk']].copy()
        display_df.columns = ['변수', '정상 기준', '현재값', '편차(%)', '판정']
        display_df['정상 기준'] = display_df['정상 기준'].apply(lambda x: f'{x:.3f}' if abs(x) < 100 else f'{x:.0f}')
        display_df['현재값'] = display_df['현재값'].apply(lambda x: f'{x:.3f}' if abs(x) < 100 else f'{x:.0f}')
        display_df['편차(%)'] = display_df['편차(%)'].apply(lambda x: f'{x:+.1f}%')
        display_df = display_df.reset_index(drop=True)
        display_df.index = display_df.index + 1
        display_df.index.name = '순위'
        st.dataframe(display_df, use_container_width=True)

    with col_chart:
        st.subheader("편차 시각화")
        chart_df = top_feats.head(5).copy()
        chart_df['color'] = chart_df['risk'].apply(
            lambda r: '#D94040' if '위험' in r else ('#E09F30' if '주의' in r else '#3A9E6E'))

        fig_dev = go.Figure(go.Bar(
            x=chart_df['pct_dev'],
            y=chart_df['name'],
            orientation='h',
            marker_color=chart_df['color'],
            text=[f'{v:+.1f}%' for v in chart_df['pct_dev']],
            textposition='outside'
        ))
        fig_dev.update_layout(
            height=300, margin=dict(l=0, r=60, t=10, b=0),
            xaxis_title='정상 기준 대비 편차 (%)',
            plot_bgcolor='rgba(0,0,0,0)',
        )
        fig_dev.add_vline(x=0, line_dash='dash', line_color='gray')
        fig_dev.add_vrect(x0=-10, x1=10, fillcolor='green', opacity=0.05,
                          annotation_text='정상 범위 ±10%', annotation_position='top left')
        st.plotly_chart(fig_dev, use_container_width=True)

    # 전체 변수 상세
    with st.expander("📋 전체 변수 정상 범위 이탈 상세"):
        full_df = all_devs[['name', 'normal_median', 'current', 'pct_dev', 'iqr_score']].copy()
        full_df.columns = ['변수', '정상 기준', '현재값', '편차(%)', 'IQR 이탈점수']
        full_df['편차(%)'] = full_df['편차(%)'].apply(lambda x: f'{x:+.1f}%')
        full_df['IQR 이탈점수'] = full_df['IQR 이탈점수'].apply(lambda x: f'{x:.2f}')
        st.dataframe(full_df.reset_index(drop=True), use_container_width=True)


# TAB 3: KPI 모니터링
with tab3:
    st.header("모니터링 KPI 4종")
    st.caption("실시간 모니터링으로 지속적 성과 관리")

    # Rolling 윈도우
    window = st.slider("Rolling 윈도우 (건수)", 50, 500, 100, 50)

    # KPI 1: 공정 변수 CV
    st.subheader("1. 공정 변수 CV (변동계수)")
    cv_feat = st.selectbox("변수 선택", available_features,
                            format_func=lambda x: x.replace('Process|','').replace('Sensor|',''))

    series = X_all[cv_feat].dropna()
    rolling_mean = series.rolling(window).mean()
    rolling_std = series.rolling(window).std()
    rolling_cv = (rolling_std / rolling_mean).dropna()

    fig_cv = go.Figure()
    fig_cv.add_trace(go.Scatter(y=rolling_cv, mode='lines', name='CV',
                                 line=dict(color='#4A7FB5', width=2)))
    fig_cv.add_hline(y=cv_threshold, line_dash='dash', line_color='#D94040',
                      annotation_text=f'경고: CV > {cv_threshold}')
    fig_cv.update_layout(height=300, margin=dict(l=0, r=0, t=30, b=0),
                          yaxis_title='CV (std/mean)', xaxis_title='생산 순서',
                          plot_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig_cv, use_container_width=True)

    current_cv = rolling_cv.iloc[-1] if len(rolling_cv) > 0 else 0
    if current_cv > cv_threshold:
        st.error(f"⚠️ 현재 CV = {current_cv:.4f} → 경고 기준 {cv_threshold} 초과")
    else:
        st.success(f"✅ 현재 CV = {current_cv:.4f} → 안정")

    st.divider()

    # KPI 2-4: 불량률, 이탈률, Risk Zone
    c1, c2, c3 = st.columns(3)

    with c1:
        st.subheader("2. 불량률 추이")
        defect_series = Y_group[target_group]
        rolling_defect = defect_series.rolling(window).mean() * 100

        fig_def = go.Figure()
        fig_def.add_trace(go.Scatter(y=rolling_defect, mode='lines', name='불량률',
                                      line=dict(color='#D94040', width=2)))
        fig_def.add_hline(y=defect_target, line_dash='dash', line_color='gray',
                           annotation_text=f'목표: {defect_target}%')
        fig_def.update_layout(height=280, margin=dict(l=0, r=0, t=10, b=0),
                               yaxis_title='불량률 (%)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_def, use_container_width=True)

    with c2:
        st.subheader("3. 안정 구간 이탈률")
        # IQR 기반 이탈 계산
        escape_counts = []
        ref = normal_stats['all']
        for feat in available_features:
            q1, q3 = ref['q1'][feat], ref['q3'][feat]
            escaped = ((X_all[feat] < q1) | (X_all[feat] > q3)).sum()
            total_valid = X_all[feat].notna().sum()
            escape_counts.append({
                'variable': feat.replace('Process|','').replace('Sensor|',''),
                'escape_rate': escaped / total_valid * 100 if total_valid > 0 else 0
            })

        esc_df = pd.DataFrame(escape_counts).sort_values('escape_rate', ascending=True)
        colors_esc = ['#D94040' if v > escape_threshold else '#4A7FB5' for v in esc_df['escape_rate']]

        fig_esc = go.Figure(go.Bar(
            x=esc_df['escape_rate'], y=esc_df['variable'],
            orientation='h', marker_color=colors_esc,
            text=[f'{v:.1f}%' for v in esc_df['escape_rate']],
            textposition='outside'
        ))
        fig_esc.add_vline(x=escape_threshold, line_dash='dash', line_color='#D94040')
        fig_esc.update_layout(height=280, margin=dict(l=0, r=40, t=10, b=0),
                               xaxis_title='이탈률 (%)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_esc, use_container_width=True)

    with c3:
        st.subheader("4. Risk Zone 추이")
        # Rolling Risk Zone
        chunk = 200
        zone_data = []
        for i in range(0, len(X_all), chunk):
            chunk_risks = risks.iloc[i:i+chunk]
            n = len(chunk_risks)
            if n == 0: continue
            chunk_sr = chunk_risks
            zone_data.append({
                'batch': f'{i}-{i+n}',
                'Stable': (chunk_sr < th_caution).sum() / n * 100,
                'Caution': ((chunk_sr >= th_caution) & (chunk_sr < th_high)).sum() / n * 100,
                'High Risk': (chunk_sr >= th_high).sum() / n * 100,
            })

        zone_df = pd.DataFrame(zone_data)
        fig_rz = go.Figure()
        fig_rz.add_trace(go.Bar(name='High Risk', x=zone_df['batch'], y=zone_df['High Risk'],
                                 marker_color='#D94040'))
        fig_rz.add_trace(go.Bar(name='Caution', x=zone_df['batch'], y=zone_df['Caution'],
                                 marker_color='#E09F30'))
        fig_rz.add_trace(go.Bar(name='Stable', x=zone_df['batch'], y=zone_df['Stable'],
                                 marker_color='#3A9E6E'))
        fig_rz.update_layout(barmode='stack', height=280,
                              margin=dict(l=0, r=0, t=10, b=0),
                              yaxis_title='비율 (%)', plot_bgcolor='rgba(0,0,0,0)',
                              legend=dict(orientation='h', y=1.15))
        st.plotly_chart(fig_rz, use_container_width=True)



# TAB 4: 참고 — Type별 분리 근거 (분석 증거 자료)
with tab4:
    st.header("📎 Type별 분리 운영 근거")
    st.info("본 탭은 분리 운영 전략의 분석 근거를 제공합니다. "
            "학습은 통합으로 진행하되, 운영 기준만 Type별로 분리 산출한 배경 데이터입니다.")

    # Type별 정상 운영 기준 테이블
    st.subheader("Type별 정상 운영 기준 비교")

    compare_rows = []
    for feat in available_features:
        name = feat.replace('Process|', '').replace('Sensor|', '')
        m1 = normal_stats[1]['median'][feat]
        m2 = normal_stats[2]['median'][feat]
        q1_1, q3_1 = normal_stats[1]['q1'][feat], normal_stats[1]['q3'][feat]
        q1_2, q3_2 = normal_stats[2]['q1'][feat], normal_stats[2]['q3'][feat]

        diff_pct = abs(m1 - m2) / max(abs(m1), abs(m2), 1e-10) * 100

        compare_rows.append({
            '변수': name,
            'Type1 중앙값': round(m1, 3),
            'Type1 IQR': f'[{q1_1:.3f} ~ {q3_1:.3f}]' if abs(q1_1) < 100 else f'[{q1_1:.0f} ~ {q3_1:.0f}]',
            'Type2 중앙값': round(m2, 3),
            'Type2 IQR': f'[{q1_2:.3f} ~ {q3_2:.3f}]' if abs(q1_2) < 100 else f'[{q1_2:.0f} ~ {q3_2:.0f}]',
            'Type간 차이(%)': round(diff_pct, 1),
        })

    compare_df = pd.DataFrame(compare_rows)
    st.dataframe(
        compare_df.style.background_gradient(subset=['Type간 차이(%)'], cmap='YlOrRd'),
        use_container_width=True, hide_index=True
    )

    st.divider()

    # 히스토그램 비교
    st.subheader("Type별 정상 분포 비교")

    sel_features = st.multiselect(
        "변수 선택 (복수 가능)",
        available_features,
        default=available_features[:4],
        format_func=lambda x: x.replace('Process|','').replace('Sensor|','')
    )

    if sel_features:
        n_cols = min(len(sel_features), 3)
        n_rows = (len(sel_features) + n_cols - 1) // n_cols

        fig_hist = make_subplots(rows=n_rows, cols=n_cols,
                                  subplot_titles=[f.replace('Process|','').replace('Sensor|','') for f in sel_features])

        for i, feat in enumerate(sel_features):
            row = i // n_cols + 1
            col = i % n_cols + 1

            for pt_val, color, name in [(1, '#4C72B0', 'Type 1'), (2, '#DD8452', 'Type 2')]:
                mask = (product_type == pt_val) & (Y_group[target_group] == 0)
                data = X_all.loc[mask, feat].dropna()

                fig_hist.add_trace(go.Histogram(
                    x=data, name=name, marker_color=color, opacity=0.6,
                    histnorm='probability density',
                    showlegend=(i == 0)
                ), row=row, col=col)

        fig_hist.update_layout(
            height=300 * n_rows,
            barmode='overlay',
            margin=dict(l=0, r=0, t=40, b=0),
            plot_bgcolor='rgba(0,0,0,0)',
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    # 핵심 인사이트
    st.divider()
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("""
        #### 🔵 Type 1 — 사이클·압력 안정성 기반 정밀 제어형
        - Rapid_Rise_Time / Pressure_Rise_Time 변동폭이 매우 작음
        - Coolant_Pressure 및 Cycle_Time이 정상 범위에 안정적으로 유지
        - High_Velocity 역시 좁은 범위 내에서 제어됨
        - → 급격한 압력 변화보다 공정 반복 안정성과 정밀 사이클 유지가 핵심
        """)

    with col_b:
        st.markdown("""
        #### 🟠 Type 2 — 냉각·압력 밸런스 기반 생산 안정형
        - Coolant_Pressure 영향도가 가장 높게 나타남
        - Coolant_Temp와 Cycle_Time 관리 중요
        - Pressure_Rise_Time 편차 민감도가 상대적으로 큼
        - → 고속·고압 자체보다 냉각 및 압력 균형 유지가 품질 안정성 핵심
        """)


# footer
st.divider()
st.caption("🏭 다이캐스팅 품질 관리 대시보드 | KAMP 데이터셋 기반 | 공정 운영 기준 설계 프로세스")
