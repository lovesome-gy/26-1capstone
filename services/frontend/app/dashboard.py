"""
frontend/app/dashboard.py
여주보 수위 예측 AI 시스템 - Streamlit 대시보드

[통합]
- 기존 dashboard.py 구조 유지 (Tab1 실시간/Tab2 보고서/Tab3 이력)
- 모델 비교 탭 추가 (멘토 피드백: Hydro-MAST vs LSTM/XGB 좌우 분할)
- 의사결정 탭 독립 분리 (멘토 피드백: 보고서와 결이 다름)
- 보고서 통계 입력 추가 (멘토 피드백: DB 통계 → JSON → LLM)
"""

import os
import time
from datetime import datetime, timedelta, timezone

import httpx
import plotly.graph_objects as go
import streamlit as st

KST = timezone(timedelta(hours=9))

LLM_URL       = os.getenv("LLM_SERVICE_URL", "http://llm_service:8002")
PREDICTOR_URL = os.getenv("PREDICTOR_URL",   "http://predictor:8001")

ALERT_COLORS = {0:"🟢", 1:"🔵", 2:"🟡", 3:"🟠", 4:"🔴"}
ALERT_LABELS = {0:"정상", 1:"관심", 2:"주의", 3:"경계", 4:"심각"}
ALERT_CONFIG = {
    0: {"color":"#155724","bg":"#d1e7dd"},
    1: {"color":"#0c5460","bg":"#d1ecf1"},
    2: {"color":"#856404","bg":"#fff3cd"},
    3: {"color":"#984c0c","bg":"#ffe5d0"},
    4: {"color":"#842029","bg":"#f8d7da"},
}
CHART_LINES = [
    (6.0,  "관심 6.0m",  "#17a2b8"),
    (7.5,  "주의 7.5m",  "#f0c040"),
    (9.0,  "경계 9.0m",  "#fd7e14"),
    (10.5, "심각 10.5m", "#dc3545"),
]


def level_to_alert(m: float) -> int:
    if m >= 10.5: return 4
    elif m >= 9.0: return 3
    elif m >= 7.5: return 2
    elif m >= 6.0: return 1
    return 0


# ── API 헬퍼 ─────────────────────────────────────────────────────────────────
def get_prediction(model: str = None) -> dict | None:
    params = {}
    if model:
        params["model"] = model
    try:
        r = httpx.post(f"{PREDICTOR_URL}/api/v1/predict", params=params, timeout=15)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"error": str(e)}


def generate_report(payload: dict) -> dict | None:
    try:
        r = httpx.post(f"{LLM_URL}/api/v1/reports/generate", json=payload, timeout=350)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"error": str(e)}


def generate_decision(payload: dict) -> list[dict]:
    try:
        r = httpx.post(f"{LLM_URL}/api/v1/decisions/generate", json=payload, timeout=350)
        r.raise_for_status()
        return r.json().get("items", [])
    except Exception as e:
        st.error(f"의사결정 생성 실패: {e}")
        return []


def get_recent_reports(limit: int = 10) -> list[dict]:
    try:
        r = httpx.get(f"{LLM_URL}/api/v1/reports/",
                      params={"station_id":"3008680","limit":limit}, timeout=10)
        r.raise_for_status()
        return r.json().get("items", [])
    except Exception:
        return []


def get_unack_decisions(limit: int = 20) -> list[dict]:
    try:
        r = httpx.get(f"{LLM_URL}/api/v1/decisions/",
                      params={"station_id":"3008680","limit":limit,"unacknowledged_only":"true"}, timeout=10)
        r.raise_for_status()
        return r.json().get("items", [])
    except Exception:
        return []


# ── 페이지 설정 ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="여주보 수위 예측 AI 시스템",
    page_icon="💧",
    layout="wide",
)

# ── 사이드바 ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ 설정")

    # 멘토 피드백: 모델 선택 또는 분할 비교
    model_mode = st.radio("모델 표시", ["단일 선택", "좌우 분할 비교"])
    if model_mode == "단일 선택":
        selected_model = st.selectbox(
            "예측 모델",
            ["Hydro-MAST", "LSTM/XGB"],
        )

    report_type = st.selectbox(
        "보고서 유형",
        ["hourly", "daily", "weekly", "monthly", "alert"],
        format_func=lambda x: {
            "hourly":"시간별","daily":"일간","weekly":"주간",
            "monthly":"월간","alert":"긴급 경보",
        }[x],
    )
    auto_refresh = st.checkbox("자동 새로고침 (60초)", value=False)

    st.divider()
    st.markdown("### 📋 경보 기준 (여주보)")
    thrs = ["6.0m 미만","6.0~7.5m","7.5~9.0m","9.0~10.5m","10.5m 이상"]
    for lvl, cfg in ALERT_CONFIG.items():
        st.markdown(
            f'<div style="background:{cfg["bg"]};color:{cfg["color"]};'
            f'padding:4px 10px;border-radius:4px;margin-bottom:3px;font-size:12px;">'
            f'<b>{ALERT_COLORS[lvl]} {lvl}단계 ({ALERT_LABELS[lvl]})</b> {thrs[lvl]}</div>',
            unsafe_allow_html=True,
        )
    st.divider()
    st.caption("LLM: Ollama / Qwen3-8b")
    st.caption("DB: PostgreSQL + TimescaleDB")

# ── 헤더 ─────────────────────────────────────────────────────────────────────
st.title("💧 여주보 수위 예측 AI 시스템")
st.caption("실시간 데이터 기반 지능형 수위 예측 및 AI 의사결정 지원 | 물결탐사대")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 실시간 현황",
    "🤖 모델 비교",
    "📄 보고서 생성",
    "🧭 의사결정 지원",
    "📋 보고서 이력",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1: 실시간 현황 (기존 코드 유지 + Plotly 차트 추가)
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    if st.button("🔄 현황 조회", key="t1_refresh"):
        with st.spinner("예측 중..."):
            pred = get_prediction()

        if pred and "error" not in pred:
            level = pred["predicted_level"]
            al    = level_to_alert(level)
            cfg   = ALERT_CONFIG[al]
            all_h = pred.get("all_horizons", {})
            cur   = level - 0.3

            st.markdown(
                f'<div style="background:{cfg["bg"]};border-left:7px solid {cfg["color"]};'
                f'padding:14px 20px;border-radius:8px;margin:10px 0 16px;">'
                f'<div style="color:{cfg["color"]};font-size:20px;font-weight:700;">'
                f'{ALERT_COLORS[al]} {al}단계 ({ALERT_LABELS[al]}) — 예측 수위 {level:.2f} m</div>'
                f'<div style="color:{cfg["color"]};font-size:13px;margin-top:4px;">'
                f'모델: {pred.get("model_name","-")} | 대상: {str(pred.get("target_time","-"))[:16]}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("현재 수위 (추정)", f"{cur:.2f} m")
            c2.metric("10분 후",  f"{all_h.get('h1_pred_m', level):.2f} m")
            c3.metric("1시간 후", f"{all_h.get('h6_pred_m', level):.2f} m")
            c4.metric("6시간 후", f"{all_h.get('h36_pred_m', level):.2f} m")

            vals = [cur,
                    all_h.get("h1_pred_m",  level),
                    all_h.get("h6_pred_m",  level),
                    all_h.get("h18_pred_m", level),
                    all_h.get("h36_pred_m", level)]
            fig = go.Figure(go.Bar(
                x=["현재","10분 후","1시간 후","3시간 후","6시간 후"], y=vals,
                marker_color=["#42a5f5","#66bb6a","#ab47bc","#ef5350","#ff7043"],
                text=[f"{v:.2f}m" for v in vals], textposition="outside", width=0.5,
            ))
            for y_val, label, color in CHART_LINES:
                fig.add_hline(y=y_val, line_dash="dot", line_color=color,
                              annotation_text=label, annotation_font_size=11)
            fig.update_layout(
                title="다지평 예측 수위 비교 (m)", yaxis_title="수위 (m)",
                height=350, showlegend=False,
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            err = pred.get("error","") if pred else ""
            st.warning(f"예측 서비스 연결 실패: {err}\nDocker 서비스를 확인해주세요.")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2: 모델 비교 (멘토 피드백: 좌우 분할)
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("🤖 모델 예측 결과 비교")
    st.caption("멘토 피드백: 하나의 화면에서 두 모델 비교 — 어떤 상황에서 어떤 모델이 적합한지 확인")

    if st.button("🔄 두 모델 동시 조회", key="t2_compare"):
        with st.spinner("두 모델 예측 중..."):
            pred_hydro = get_prediction("hydro_mast")
            pred_lstm  = get_prediction("lstm_xgb")

        col_l, col_r = st.columns(2)

        # ── Hydro-MAST (왼쪽) ─────────────────────────────────────────────
        with col_l:
            st.markdown("### 🌊 Hydro-MAST")
            st.caption("Graph-GRU + Advective Delay | 4지평 동시 예측 | 평균 NSE 0.908")
            if pred_hydro and "error" not in pred_hydro:
                all_h = pred_hydro.get("all_horizons", {})
                level = pred_hydro.get("predicted_level", 0)
                al    = level_to_alert(level)
                cfg   = ALERT_CONFIG[al]
                st.markdown(
                    f'<div style="background:{cfg["bg"]};border-left:5px solid {cfg["color"]};'
                    f'padding:8px 14px;border-radius:6px;">'
                    f'<b style="color:{cfg["color"]};">'
                    f'{ALERT_COLORS[al]} {al}단계 ({ALERT_LABELS[al]}) — {level:.2f} m</b></div>',
                    unsafe_allow_html=True,
                )
                m1,m2,m3,m4 = st.columns(4)
                m1.metric("NSE h1","0.939"); m2.metric("NSE h6","0.924")
                m3.metric("NSE h18","0.902"); m4.metric("NSE h36","0.868")
                h_vals = [all_h.get("h1_pred_m",level), all_h.get("h6_pred_m",level),
                          all_h.get("h18_pred_m",level), all_h.get("h36_pred_m",level)]
                fig_l = go.Figure(go.Bar(
                    x=["10분","1시간","3시간","6시간"], y=h_vals,
                    marker_color=["#42a5f5","#66bb6a","#ab47bc","#ef5350"],
                    text=[f"{v:.2f}m" for v in h_vals], textposition="outside",
                ))
                for y_val, label, color in CHART_LINES[:3]:
                    fig_l.add_hline(y=y_val, line_dash="dot", line_color=color,
                                    annotation_text=label, annotation_font_size=10)
                fig_l.update_layout(title="Hydro-MAST 4지평", height=300, showlegend=False,
                                    plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig_l, use_container_width=True)
            else:
                err = pred_hydro.get("error","") if pred_hydro else "연결 실패"
                st.error(f"Hydro-MAST 오류: {err}")

        # ── LSTM/XGB (오른쪽) ──────────────────────────────────────────────
        with col_r:
            st.markdown("### 🧠 LSTM/XGB")
            st.caption("LSTM + XGBoost 앙상블 | 시계열 딥러닝 | 10분/1시간/3시간")
            if pred_lstm and "error" not in pred_lstm:
                all_h = pred_lstm.get("all_horizons", {})
                level = pred_lstm.get("predicted_level", 0)
                al    = level_to_alert(level)
                cfg   = ALERT_CONFIG[al]
                st.markdown(
                    f'<div style="background:{cfg["bg"]};border-left:5px solid {cfg["color"]};'
                    f'padding:8px 14px;border-radius:6px;">'
                    f'<b style="color:{cfg["color"]};">'
                    f'{ALERT_COLORS[al]} {al}단계 ({ALERT_LABELS[al]}) — {level:.2f} m</b></div>',
                    unsafe_allow_html=True,
                )
                h_vals = [all_h.get("h1_pred_m",level), all_h.get("h6_pred_m",level),
                          all_h.get("h18_pred_m",level), all_h.get("h36_pred_m",level)]
                fig_r = go.Figure(go.Bar(
                    x=["10분","1시간","3시간","6시간(근사)"], y=h_vals,
                    marker_color=["#42a5f5","#66bb6a","#ab47bc","#ef5350"],
                    text=[f"{v:.2f}m" for v in h_vals], textposition="outside",
                ))
                for y_val, label, color in CHART_LINES[:3]:
                    fig_r.add_hline(y=y_val, line_dash="dot", line_color=color,
                                    annotation_text=label, annotation_font_size=10)
                fig_r.update_layout(title="LSTM/XGB 앙상블", height=300, showlegend=False,
                                    plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig_r, use_container_width=True)
                st.caption("※ 6시간 예측은 3시간 모델 근사값")
            else:
                err = pred_lstm.get("error","") if pred_lstm else ""
                if "503" in str(err) or "연결" in str(err):
                    st.info("⏳ LSTM/XGB 서버 연결 대기 중\n\n`services/lstm_xgb/models/`에 모델 파일 복사 후 `docker compose up lstm_xgb -d --build`")
                else:
                    st.error(f"LSTM/XGB 오류: {err}")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3: 보고서 생성 (기존 코드 + 멘토 피드백 반영)
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("📄 AI 자연어 보고서 생성")
    st.caption("멘토 피드백: DB 통계 → JSON → LLM | 월간은 통계 자료 활용")

    with st.expander("수위 입력", expanded=True):
        c1, c2 = st.columns(2)
        cur_level  = c1.number_input("현재 수위 (m)", 0.0, 30.0, 5.23, 0.01, key="r_cur")
        pred_level = c2.number_input("예측 수위 (m)", 0.0, 30.0, 5.87, 0.01, key="r_pred")

    # 멘토 피드백: 일간/주간/월간은 통계 입력 추가
    if report_type in ["daily","weekly","monthly"]:
        st.markdown("**기간 통계** (멘토 피드백: DB 집계 후 LLM 전달)")
        s1,s2,s3,s4 = st.columns(4)
        avg_l  = s1.number_input("평균 수위 (m)", value=5.35, step=0.01)
        max_l  = s2.number_input("최고 수위 (m)", value=6.10, step=0.01)
        min_l  = s3.number_input("최저 수위 (m)", value=4.90, step=0.01)
        alrt_n = s4.number_input("경보 횟수", value=0, min_value=0, step=1)
    else:
        avg_l = max_l = min_l = alrt_n = None

    if st.button("📝 보고서 생성", key="t3_gen"):
        now  = datetime.now(KST)
        dmap = {"hourly":timedelta(hours=1),"daily":timedelta(days=1),
                "weekly":timedelta(weeks=1),"monthly":timedelta(days=30),"alert":timedelta(hours=1)}
        payload = {
            "station_id":      "3008680",
            "report_type":     report_type,
            "water_level_cur":  cur_level,
            "water_level_pred": pred_level,
            "period_start":    (now - dmap[report_type]).isoformat(),
            "period_end":       now.isoformat(),
        }
        if avg_l is not None:
            payload.update({"avg_level":avg_l,"max_level":max_l,"min_level":min_l,"alert_count":int(alrt_n)})

        with st.spinner(f"LLM 보고서 생성 중 ({report_type}) — 최대 5분"):
            t0     = time.time()
            report = generate_report(payload)
            elapsed = time.time() - t0

        if report and "error" not in report:
            al  = report.get("alert_level", 0)
            cfg = ALERT_CONFIG[al]
            st.success(f"보고서 생성 완료 ({elapsed:.1f}초)")
            col_a, col_b, col_c = st.columns(3)
            col_a.metric("경보 단계", f"{ALERT_COLORS[al]} {ALERT_LABELS[al]}")
            col_b.metric("추세",      report.get("trend", "-"))
            col_c.metric("생성 시간", f"{report.get('generation_ms',0)/1000:.1f}초")
            st.markdown("#### 📌 요약")
            st.info(report.get("report_summary",""))
            st.markdown("#### 📄 보고서 본문")
            st.text_area("", value=report.get("report_body",""), height=200, disabled=True, key="t3_body")

            if st.button("🧭 의사결정 바로 생성", key="t3_dec"):
                dp = {"station_id":"3008680","alert_level":al,
                      "water_level_cur":cur_level,"water_level_pred":pred_level,
                      "trend":report.get("trend","stable"),"report_id":report.get("report_id")}
                with st.spinner("의사결정 생성 중..."):
                    items = generate_decision(dp)
                for item in items:
                    p = {1:"🔴 긴급",2:"🟡 일반",3:"🔵 참고"}.get(item.get("priority",2),"")
                    with st.expander(f"{p} {item['decision_title']}"):
                        st.write(item["decision_body"])
                        if item.get("rationale"):
                            st.caption(f"판단 근거: {item['rationale']}")
        else:
            err = report.get("error","LLM 서비스 미연결") if report else "LLM 서비스 미연결"
            st.error(f"보고서 생성 실패: {err}")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4: 의사결정 지원 (멘토 피드백: 보고서와 독립)
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.subheader("🧭 AI 의사결정 지원")
    st.caption("멘토 피드백: 예측 수위 기반 대응 방법 안내 — 보고서와 독립적 기능")

    d1, d2 = st.tabs(["의사결정 생성","미확인 항목"])

    with d1:
        da, db = st.columns(2)
        d_cur  = da.number_input("현재 수위 (m)", value=7.8, step=0.01, key="d_cur")
        d_pred = db.number_input("예측 수위 (m)", value=8.5, step=0.01, key="d_pred")
        d_trend = st.select_slider("수위 추세", ["falling","stable","rising"], value="rising",
                                   format_func=lambda x: {"falling":"↓ 하강","stable":"→ 안정","rising":"↑ 상승"}[x])
        d_alert = level_to_alert(max(d_cur, d_pred))
        d_cfg   = ALERT_CONFIG[d_alert]
        st.markdown(
            f'<div style="background:{d_cfg["bg"]};color:{d_cfg["color"]};'
            f'padding:8px 14px;border-radius:6px;">'
            f'<b>{ALERT_COLORS[d_alert]} 자동 계산: {d_alert}단계 ({ALERT_LABELS[d_alert]})</b></div>',
            unsafe_allow_html=True,
        )

        if st.button("🧭 의사결정 생성", key="t4_gen"):
            payload = {"station_id":"3008680","alert_level":d_alert,
                       "water_level_cur":d_cur,"water_level_pred":d_pred,"trend":d_trend}
            with st.spinner("LLM 의사결정 생성 중..."):
                items = generate_decision(payload)
            if items:
                st.success(f"✅ {len(items)}개 항목 생성")
                for item in items:
                    p   = {1:"🔴 긴급",2:"🟡 일반",3:"🔵 참고"}.get(item.get("priority",2),"")
                    cat = {"gate_control":"🚪 수문 제어","evacuation":"🚨 대피",
                           "monitoring":"👁 모니터링","standby":"⏳ 대기"}.get(item.get("action_category",""),"")
                    with st.expander(f"{p} {cat} — {item['decision_title']}"):
                        st.write(item["decision_body"])
                        if item.get("rationale"):
                            st.caption(f"판단 근거: {item['rationale']}")
                        if st.button("✅ 확인 처리", key=f"ack_{item['decision_id']}"):
                            try:
                                httpx.patch(f"{LLM_URL}/api/v1/decisions/acknowledge",
                                           json={"decision_id":item["decision_id"]}, timeout=5)
                                st.success("확인됨"); st.rerun()
                            except Exception:
                                pass

    with d2:
        if st.button("🔄 미확인 항목 조회", key="t4_unack"):
            items = get_unack_decisions(20)
            if items:
                st.warning(f"⚠️ 미확인 {len(items)}건")
                for item in items:
                    al  = item.get("alert_level", 0)
                    cfg = ALERT_CONFIG[al]
                    p   = {1:"🔴 긴급",2:"🟡 일반",3:"🔵 참고"}.get(item.get("priority",2),"")
                    with st.expander(f"{p} {ALERT_COLORS[al]} {item['decision_title']} ({item['created_at'][:16]})"):
                        st.write(item["decision_body"])
                        if st.button("✅ 확인", key=f"ack2_{item['decision_id']}"):
                            try:
                                httpx.patch(f"{LLM_URL}/api/v1/decisions/acknowledge",
                                           json={"decision_id":item["decision_id"]}, timeout=5)
                                st.rerun()
                            except Exception:
                                pass
            else:
                st.success("✅ 미확인 항목 없음")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 5: 보고서 이력 (기존 Tab3 유지)
# ══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.subheader("📋 최근 보고서 이력")
    if st.button("🔄 이력 불러오기", key="t5_hist"):
        reports = get_recent_reports(10)
        if reports:
            for rpt in reports:
                al = rpt.get("alert_level", 0)
                with st.expander(
                    f"{ALERT_COLORS[al]} [{rpt['report_type']}] "
                    f"{rpt['created_at'][:16]} — {rpt['report_summary'][:50]}..."
                ):
                    st.write(rpt["report_body"])
                    st.caption(
                        f"report_id={rpt['report_id']} | "
                        f"model={rpt['llm_model']} | "
                        f"{rpt.get('generation_ms',0)}ms"
                    )
        else:
            st.info("보고서 이력이 없습니다.")

if auto_refresh:
    time.sleep(60)
    st.rerun()
