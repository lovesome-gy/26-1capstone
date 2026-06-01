# -*- coding: utf-8 -*-
r"""
실시간 예측 시각 확인용 간단 대시보드 생성기.

실행:
  .venv\Scripts\python realtime_visual_check.py
  .venv\Scripts\python realtime_visual_check.py --skip-api
"""
from __future__ import annotations

import argparse
import json
import webbrowser
from pathlib import Path

import pandas as pd

import realtime_pipeline_v2 as rp

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
JSON_PATH = DATA_DIR / "realtime_latest_prediction.json"
HTML_PATH = DATA_DIR / "realtime_visual_check.html"

HORIZON_LABELS = {
    "h1_pred_m": "10분 후",
    "h6_pred_m": "1시간 후",
    "h18_pred_m": "3시간 후",
    "h36_pred_m": "6시간 후",
}
HORIZON_MINUTES = {
    "h1_pred_m": 10,
    "h6_pred_m": 60,
    "h18_pred_m": 180,
    "h36_pred_m": 360,
}


def build_html(payload: dict) -> str:
    base = pd.Timestamp(payload["bucket_start_kst"])
    preds: dict[str, float] = payload["predictions_m"]
    meta: dict = payload.get("meta", {})
    current = meta.get("current_level_m")
    warns = meta.get("warnings", []) or []
    source_kw = meta.get("kwater_source", "-")
    source_hr = meta.get("hrfc_source", "-")

    points = []
    if current is not None:
        points.append({"t": base.strftime("%m-%d %H:%M"), "v": float(current), "name": "현재"})
    horizon_order = ("h1_pred_m", "h6_pred_m", "h18_pred_m", "h36_pred_m")
    for k in horizon_order:
        if k in preds:
            t = (base + pd.Timedelta(minutes=HORIZON_MINUTES[k])).strftime("%m-%d %H:%M")
            points.append({"t": t, "v": float(preds[k]), "name": HORIZON_LABELS[k]})

    labels = [p["t"] for p in points]
    values = [p["v"] for p in points]
    point_names = [p["name"] for p in points]
    marker_colors = ["#111827" if n == "현재" else "#2563eb" for n in point_names]

    summary_cards = []
    table_rows = []
    deltas = []
    delta_labels = []
    for k in horizon_order:
        if k not in preds:
            continue
        pred_v = float(preds[k])
        pred_t = base + pd.Timedelta(minutes=HORIZON_MINUTES[k])
        delta = pred_v - float(current) if current is not None else None
        delta_txt = "-" if delta is None else f"{delta:+.4f} m"
        delta_cls = "flat"
        delta_icon = "→"
        if delta is not None:
            if delta > 0.01:
                delta_cls, delta_icon = "up", "↑"
            elif delta < -0.01:
                delta_cls, delta_icon = "down", "↓"
        summary_cards.append(
            f"""
            <div class="mini-card {delta_cls}">
              <div class="mini-title">{HORIZON_LABELS[k]}</div>
              <div class="mini-value">{pred_v:.4f} m</div>
              <div class="mini-delta">{delta_icon} {delta_txt}</div>
            </div>
            """
        )
        table_rows.append(
            f"<tr><td>{HORIZON_LABELS[k]}</td><td>{pred_t:%Y-%m-%d %H:%M}</td><td>{pred_v:.4f}</td><td>{delta_txt}</td></tr>"
        )
        deltas.append(0.0 if delta is None else float(delta))
        delta_labels.append(HORIZON_LABELS[k])

    rows = []
    if current is not None:
        rows.append(f"<tr><td>현재</td><td>{base:%Y-%m-%d %H:%M}</td><td>{float(current):.4f}</td><td>-</td></tr>")
    rows.extend(table_rows)

    if current is not None and "h36_pred_m" in preds:
        long_delta = float(preds["h36_pred_m"]) - float(current)
        if long_delta > 0.01:
            trend_chip = '<span class="chip up">6시간 기준 상승 가능성</span>'
        elif long_delta < -0.01:
            trend_chip = '<span class="chip down">6시간 기준 하락 가능성</span>'
        else:
            trend_chip = '<span class="chip flat">6시간 기준 큰 변화 없음</span>'
    else:
        trend_chip = '<span class="chip flat">추세 판단 정보 부족</span>'

    warn_html = "<br/>".join(warns) if warns else "현재 경고 없음"
    warn_box_class = "warn-box warn-ok" if not warns else "warn-box warn-has"
    last_update = pd.Timestamp.now(tz="Asia/Seoul").strftime("%Y-%m-%d %H:%M:%S")

    return f"""<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8" />
  <title>실시간 홍수 예측 확인</title>
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <style>
    :root {{
      --bg:#f3f6fb;
      --card:#ffffff;
      --text:#111827;
      --muted:#4b5563;
      --line:#e5e7eb;
      --blue:#2563eb;
      --up:#dc2626;
      --down:#059669;
      --flat:#6b7280;
    }}
    body {{ font-family: "Segoe UI", Arial, sans-serif; margin: 0; padding: 20px; background:var(--bg); color:var(--text); }}
    .wrap {{ max-width: 1100px; margin: 0 auto; }}
    .card {{ background:var(--card); border-radius:14px; padding:18px; margin-bottom:14px; box-shadow:0 1px 8px rgba(15,23,42,.08); }}
    h1 {{ margin:0 0 10px; font-size:24px; }}
    .sub {{ color:var(--muted); font-size:14px; }}
    .top-grid {{ display:grid; grid-template-columns: 1.2fr 1fr; gap:12px; }}
    .mini-grid {{ display:grid; grid-template-columns: repeat(4, 1fr); gap:10px; }}
    .mini-card {{ border:1px solid var(--line); border-radius:10px; padding:12px; }}
    .mini-card.up {{ border-left:4px solid var(--up); }}
    .mini-card.down {{ border-left:4px solid var(--down); }}
    .mini-card.flat {{ border-left:4px solid var(--flat); }}
    .mini-title {{ font-size:13px; color:var(--muted); margin-bottom:5px; }}
    .mini-value {{ font-size:20px; font-weight:700; }}
    .mini-delta {{ margin-top:4px; font-size:13px; color:var(--muted); }}
    .chip {{ display:inline-block; padding:6px 10px; border-radius:999px; font-size:13px; font-weight:600; }}
    .chip.up {{ color:#fff; background:var(--up); }}
    .chip.down {{ color:#fff; background:var(--down); }}
    .chip.flat {{ color:#fff; background:var(--flat); }}
    .warn-box {{ border-radius:10px; padding:10px 12px; margin-top:10px; font-size:14px; line-height:1.5; }}
    .warn-ok {{ background:#ecfdf5; color:#065f46; border:1px solid #a7f3d0; }}
    .warn-has {{ background:#fffbeb; color:#92400e; border:1px solid #fde68a; }}
    .charts {{ display:grid; grid-template-columns: 2fr 1fr; gap:12px; }}
    table {{ border-collapse: collapse; width:100%; }}
    th, td {{ border-bottom:1px solid var(--line); padding:10px; text-align:left; font-size:14px; }}
    th {{ background:#f8fafc; }}
    .meta-line {{ font-size:14px; color:var(--muted); line-height:1.7; }}
    @media (max-width: 980px) {{
      .top-grid, .charts, .mini-grid {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <div class="card">
      <h1>실시간 홍수 예측 모니터</h1>
      <div class="meta-line">
        기준시각(KST): <b>{base:%Y-%m-%d %H:%M}</b> | 화면생성시각: {last_update}<br/>
        데이터 소스: K-water={source_kw}, HRFC={source_hr}
      </div>
      <div style="margin-top:10px;">{trend_chip}</div>
      <div class="{warn_box_class}"><b>알림</b><br/>{warn_html}</div>
    </div>
    <div class="card">
      <div class="mini-grid">
        {''.join(summary_cards)}
      </div>
    </div>
    <div class="card charts">
      <div id="chart_level" style="height:390px;"></div>
      <div id="chart_delta" style="height:390px;"></div>
    </div>
    <div class="card">
      <table>
        <thead><tr><th>구분</th><th>시각(KST)</th><th>수위(m)</th><th>현재 대비 변화</th></tr></thead>
        <tbody>
          {''.join(rows)}
        </tbody>
      </table>
    </div>
  </div>
  <script>
    const labels = {json.dumps(labels, ensure_ascii=False)};
    const values = {json.dumps(values, ensure_ascii=False)};
    const names = {json.dumps(point_names, ensure_ascii=False)};
    const markerColors = {json.dumps(marker_colors, ensure_ascii=False)};
    const deltaLabels = {json.dumps(delta_labels, ensure_ascii=False)};
    const deltas = {json.dumps(deltas, ensure_ascii=False)};

    const traceLevel = {{
      x: labels,
      y: values,
      mode: "lines+markers+text",
      text: names,
      textposition: "top center",
      line: {{ width: 3, color: "#2563eb" }},
      marker: {{ size: 10, color: markerColors }},
      name: "수위"
    }};

    const layoutLevel = {{
      title: "수위 예측 경로 (현재 -> 6시간 후)",
      xaxis: {{ title: "시각(KST)" }},
      yaxis: {{ title: "수위(m)" }},
      margin: {{ l: 60, r: 20, t: 50, b: 60 }},
      paper_bgcolor: "white",
      plot_bgcolor: "white"
    }};

    const traceDelta = {{
      x: deltaLabels,
      y: deltas,
      type: "bar",
      marker: {{
        color: deltas.map(v => v > 0.01 ? "#dc2626" : (v < -0.01 ? "#059669" : "#6b7280"))
      }},
      text: deltas.map(v => (v >= 0 ? "+" : "") + v.toFixed(4) + " m"),
      textposition: "auto",
      name: "변화량"
    }};

    const layoutDelta = {{
      title: "현재 대비 변화량",
      xaxis: {{ title: "예측 시점" }},
      yaxis: {{ title: "변화량(m)", zeroline: true, zerolinecolor: "#9ca3af" }},
      margin: {{ l: 60, r: 20, t: 50, b: 60 }},
      paper_bgcolor: "white",
      plot_bgcolor: "white"
    }};

    Plotly.newPlot("chart_level", [traceLevel], layoutLevel, {{responsive:true, displaylogo:false}});
    Plotly.newPlot("chart_delta", [traceDelta], layoutDelta, {{responsive:true, displaylogo:false}});
  </script>
</body>
</html>
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-api", action="store_true", help="API 없이 테스트 모드")
    ap.add_argument("--no-open", action="store_true", help="브라우저 자동 오픈 안 함")
    args = ap.parse_args()

    rp.run(skip_api=args.skip_api)
    payload = json.loads(JSON_PATH.read_text(encoding="utf-8"))
    html = build_html(payload)
    HTML_PATH.write_text(html, encoding="utf-8")

    print("=" * 60)
    print("Visual check dashboard generated")
    print(f"[saved] {HTML_PATH}")
    if not args.no_open:
        webbrowser.open(HTML_PATH.as_uri())
        print("[opened] browser")


if __name__ == "__main__":
    main()
