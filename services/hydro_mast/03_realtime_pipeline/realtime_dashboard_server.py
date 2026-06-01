# -*- coding: utf-8 -*-
r"""
실시간 예측 대시보드 서버 (갱신 버튼 지원)

실행:
  .venv\Scripts\python realtime_dashboard_server.py
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
import time
import webbrowser
from functools import lru_cache
from pathlib import Path
from urllib.request import urlopen

from flask import Flask, jsonify, request

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
for p in (PROJECT / "01_data_pipeline", PROJECT / "02_model_development"):
    if str(p) not in sys.path:
        sys.path.append(str(p))

import realtime_pipeline_v2 as rp
from config_v2 import HRFC_STATIONS
from hydro_mast_data import HRFC_TOPO_ORDER

DATA_DIR = PROJECT / "04_artifacts" / "data"
JSON_PATH = DATA_DIR / "realtime_latest_prediction.json"

app = Flask(__name__)


def _run_refresh(skip_api: bool = False) -> dict:
    rp.run(skip_api=skip_api)
    return json.loads(JSON_PATH.read_text(encoding="utf-8"))


def _get_latest_or_refresh() -> dict:
    if JSON_PATH.exists():
        return json.loads(JSON_PATH.read_text(encoding="utf-8"))
    return _run_refresh(skip_api=False)


def _load_env_local() -> dict[str, str]:
    env_path = ROOT / ".env"
    out: dict[str, str] = {}
    if not env_path.exists():
        return out
    for line in env_path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#") or "=" not in s:
            continue
        k, v = s.split("=", 1)
        out[k.strip()] = v.strip()
    return out


def _parse_dms_or_float(v: str | None) -> float | None:
    if not v:
        return None
    s = str(v).strip()
    try:
        return float(s)
    except Exception:
        pass
    # e.g. 127-30-00 / 127:30:00
    for sep in ("-", ":"):
        if sep in s:
            parts = s.split(sep)
            if len(parts) >= 3:
                try:
                    d = float(parts[0])
                    m = float(parts[1])
                    sec = float(parts[2])
                    sign = -1.0 if d < 0 else 1.0
                    return sign * (abs(d) + m / 60.0 + sec / 3600.0)
                except Exception:
                    return None
    return None


@lru_cache(maxsize=1)
def _fetch_hrfc_station_info() -> dict[str, dict]:
    env = _load_env_local()
    key = env.get("HRFCO_SERVICE_KEY", "").strip()
    if not key:
        return {}
    urls = [
        f"https://api.hrfco.go.kr/{key}/waterlevel/info.json",
        f"http://api.hrfco.go.kr/{key}/waterlevel/info.json",
    ]
    for u in urls:
        try:
            raw = urlopen(u, timeout=12).read().decode("utf-8", errors="ignore")
            data = json.loads(raw)
            content = data.get("content", [])
            out: dict[str, dict] = {}
            for row in content:
                if not isinstance(row, dict):
                    continue
                code = str(row.get("wlobscd", "")).strip()
                if not code:
                    continue
                lat = _parse_dms_or_float(row.get("lat"))
                lon = _parse_dms_or_float(row.get("lon"))
                out[code] = {
                    "code": code,
                    "name": str(row.get("obsnm") or HRFC_STATIONS.get(code) or code),
                    "lat": lat,
                    "lon": lon,
                }
            if out:
                return out
        except Exception:
            continue
    return {}


def _build_station_payload() -> list[dict]:
    # fallback approximate points around Yeoju when lat/lon API fails
    fallback = {
        "1007662": (37.426, 127.536),
        "1007664": (37.417, 127.545),
        "1007641": (37.299, 127.638),
        "1007639": (37.300, 127.636),
        "1007635": (37.545, 127.490),
        "1007633": (37.615, 127.360),
    }
    info = _fetch_hrfc_station_info()
    payload: list[dict] = []
    for code in HRFC_TOPO_ORDER:
        row = info.get(code, {})
        lat = row.get("lat")
        lon = row.get("lon")
        if (lat is None or lon is None) and code in fallback:
            lat, lon = fallback[code]
        payload.append(
            {
                "code": code,
                "name": HRFC_STATIONS.get(code, code),
                "lat": lat,
                "lon": lon,
                "is_target": code == "1007639",
            }
        )
    return payload


@app.get("/api/latest")
def api_latest():
    try:
        payload = _get_latest_or_refresh()
        return jsonify({"ok": True, "payload": payload})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.get("/api/health")
def api_health():
    required = {
        "model": (rp.MODEL_DIR / "hydro_mast_v2.pt").exists(),
        "feature_scaler": (rp.MODEL_DIR / "feature_scaler_v2.pkl").exists(),
        "target_scaler": (rp.MODEL_DIR / "target_scaler_v2.pkl").exists(),
        "train_csv": (rp.DATA_DIR / "features_v2_train.csv").exists(),
        "test_csv": (rp.DATA_DIR / "features_v2_test.csv").exists(),
    }
    env_exists = (PROJECT / ".env").exists()
    return jsonify({"ok": True, "status": "healthy", "env_exists": env_exists, "required_files": required})


@app.post("/api/predict")
def api_predict():
    """평가자용 단일 추론 API."""
    try:
        body = request.get_json(silent=True) or {}
        skip_api = bool(body.get("skip_api", True))
        payload = _run_refresh(skip_api=skip_api)
        return jsonify({"ok": True, "payload": payload})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.post("/api/refresh")
def api_refresh():
    try:
        body = request.get_json(silent=True) or {}
        skip_api = bool(body.get("skip_api", False))
        payload = _run_refresh(skip_api=skip_api)
        return jsonify({"ok": True, "payload": payload})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.get("/api/stations")
def api_stations():
    try:
        return jsonify({"ok": True, "stations": _build_station_payload()})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e), "stations": []}), 500


@app.get("/")
def index():
    return """<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8" />
  <title>실시간 홍수 예측 모니터</title>
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <link
    rel="stylesheet"
    href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"
    integrity="sha256-p4NxAoJBhIIN+hmNHrzRCf9tD/miZyoHS5obTRR9BMY="
    crossorigin=""
  />
  <script
    src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"
    integrity="sha256-20nQCchB9co0qIjJZRGuk2/Z9VM+kNiyxNV1lvTlZBo="
    crossorigin=""
  ></script>
  <style>
    :root {
      --bg:#f3f6fb; --card:#fff; --text:#111827; --muted:#4b5563; --line:#e5e7eb;
      --blue:#2563eb; --up:#dc2626; --down:#059669; --flat:#6b7280;
    }
    body { font-family:"Segoe UI",Arial,sans-serif; margin:0; padding:20px; background:var(--bg); color:var(--text); }
    .wrap { max-width:1100px; margin:0 auto; }
    .card { background:var(--card); border-radius:14px; padding:18px; margin-bottom:14px; box-shadow:0 1px 8px rgba(15,23,42,.08); }
    h1 { margin:0 0 10px; font-size:24px; }
    .meta-line { font-size:14px; color:var(--muted); line-height:1.7; }
    .top-actions { display:flex; gap:8px; margin-top:10px; flex-wrap:wrap; }
    button { border:0; border-radius:10px; padding:10px 14px; cursor:pointer; font-weight:600; }
    .btn-primary { background:var(--blue); color:#fff; }
    .btn-light { background:#e5e7eb; color:#111827; }
    .btn-primary:disabled, .btn-light:disabled { opacity:.6; cursor:not-allowed; }
    .status { font-size:13px; color:var(--muted); margin-top:8px; }
    .mini-grid { display:grid; grid-template-columns:repeat(4,1fr); gap:10px; }
    .mini-card { border:1px solid var(--line); border-radius:10px; padding:12px; }
    .mini-card.up { border-left:4px solid var(--up); }
    .mini-card.down { border-left:4px solid var(--down); }
    .mini-card.flat { border-left:4px solid var(--flat); }
    .mini-title { font-size:13px; color:var(--muted); margin-bottom:5px; }
    .mini-value { font-size:20px; font-weight:700; }
    .mini-delta { margin-top:4px; font-size:13px; color:var(--muted); }
    .chip { display:inline-block; padding:6px 10px; border-radius:999px; font-size:13px; font-weight:600; }
    .chip.up { color:#fff; background:var(--up); } .chip.down { color:#fff; background:var(--down); } .chip.flat { color:#fff; background:var(--flat); }
    .warn-box { border-radius:10px; padding:10px 12px; margin-top:10px; font-size:14px; line-height:1.5; }
    .warn-ok { background:#ecfdf5; color:#065f46; border:1px solid #a7f3d0; }
    .warn-has { background:#fffbeb; color:#92400e; border:1px solid #fde68a; }
    .charts { display:grid; grid-template-columns:2fr 1fr; gap:12px; }
    .loc-grid { display:grid; grid-template-columns:1.3fr 1fr; gap:12px; align-items:stretch; }
    .map-panel { border:1px solid var(--line); border-radius:12px; padding:10px; background:#f8fafc; }
    .loc-panel { border:1px solid var(--line); border-radius:12px; padding:12px; background:#fff; }
    .loc-title { font-size:15px; font-weight:700; margin-bottom:8px; }
    .loc-item { font-size:14px; color:var(--muted); margin:4px 0; }
    .loc-item b { color:var(--text); }
    .river-label { font-size:12px; fill:#334155; font-weight:600; }
    .node-label { font-size:11px; fill:#111827; }
    #mapView { width:100%; height:280px; border-radius:10px; overflow:hidden; border:1px solid var(--line); }
    .map-note { font-size:12px; color:var(--muted); margin-top:8px; }
    .map-legend { display:flex; gap:8px; flex-wrap:wrap; margin-top:8px; }
    .legend-chip { font-size:12px; padding:4px 8px; border-radius:999px; background:#eef2ff; color:#1e3a8a; border:1px solid #c7d2fe; }
    .target-dot {
      width:14px; height:14px; border-radius:50%;
      background:#ef4444; border:2px solid #fff; box-shadow:0 0 0 2px #ef4444;
    }
    .ctx-dot {
      width:10px; height:10px; border-radius:50%;
      background:#0284c7; border:2px solid #fff; box-shadow:0 0 0 1px #0284c7;
    }
    table { border-collapse:collapse; width:100%; }
    th,td { border-bottom:1px solid var(--line); padding:10px; text-align:left; font-size:14px; }
    th { background:#f8fafc; }
    @media (max-width:980px) { .mini-grid,.charts,.loc-grid { grid-template-columns:1fr; } }
  </style>
</head>
<body>
  <div class="wrap">
    <div class="card">
      <h1>실시간 홍수 예측 모니터</h1>
      <div id="meta" class="meta-line">데이터 로딩 중...</div>
      <div class="top-actions">
        <button id="btnRefresh" class="btn-primary">갱신 (실시간 재추론)</button>
      </div>
      <div id="status" class="status"></div>
      <div id="trend" style="margin-top:10px;"></div>
      <div id="warnBox" class="warn-box warn-ok">현재 경고 없음</div>
    </div>
    <div class="card"><div id="summaryCards" class="mini-grid"></div></div>
    <div class="card loc-grid">
      <div class="map-panel">
        <div class="loc-title">예측 위치 지도 (여주 권역)</div>
        <div id="mapView" aria-label="예측 위치 지도"></div>
        <div class="map-legend">
          <span class="legend-chip">🔴 예측 타깃 지점</span>
          <span class="legend-chip">🔵 인접 참조 지점</span>
        </div>
        <div class="map-note" style="display:flex;align-items:center;gap:8px;">
          <input type="checkbox" id="toggleStations" />
          <label for="toggleStations">전체 관측소 레이어 보기</label>
        </div>
        <div class="map-note">지도는 OpenStreetMap 기준이며, 보조 지점은 가시화를 위한 참조용입니다.</div>
      </div>
      <div class="loc-panel">
        <div class="loc-title">예측 대상 설명</div>
        <div id="locTarget" class="loc-item">로딩 중...</div>
        <div id="locScope" class="loc-item"></div>
        <div id="locDamUsage" class="loc-item"></div>
        <div class="loc-item" style="margin-top:10px;">
          <b>해석 가이드</b><br/>
          - 화면 값은 <b>특정 지점(여주보 상류)</b> 수위 예측값입니다.<br/>
          - 권역 전체 평균 수위를 직접 예측하는 화면이 아닙니다.
        </div>
      </div>
    </div>
    <div class="card charts">
      <div id="chart_level" style="height:390px;"></div>
      <div id="chart_delta" style="height:390px;"></div>
    </div>
    <div class="card">
      <table>
        <thead><tr><th>구분</th><th>시각(KST)</th><th>수위(m)</th><th>현재 대비 변화</th></tr></thead>
        <tbody id="tb"></tbody>
      </table>
    </div>
  </div>
  <script>
    const H_LABELS = {h1_pred_m:"10분 후", h6_pred_m:"1시간 후", h18_pred_m:"3시간 후", h36_pred_m:"6시간 후"};
    const H_MINS = {h1_pred_m:10, h6_pred_m:60, h18_pred_m:180, h36_pred_m:360};
    const ORDER = ["h1_pred_m","h6_pred_m","h18_pred_m","h36_pred_m"];
    let mapObj = null;
    let targetMarker = null;
    let rangeCircle = null;
    let stationLayer = null;
    let stationMarkers = [];

    const el = (id) => document.getElementById(id);
    const fmt = (v) => Number(v).toFixed(4);
    const nowStr = () => new Date().toLocaleString("ko-KR", {hour12:false});

    function setLoading(on, msg="") {
      el("btnRefresh").disabled = on;
      el("status").textContent = msg;
    }

    function makeDotIcon(kind){
      const cls = kind === "target" ? "target-dot" : "ctx-dot";
      return L.divIcon({
        className: "",
        html: `<div class="${cls}"></div>`,
        iconSize: kind === "target" ? [18,18] : [14,14],
        iconAnchor: kind === "target" ? [9,9] : [7,7]
      });
    }

    async function loadStations(){
      try {
        const r = await fetch("/api/stations");
        const j = await r.json();
        if (!j.ok) return [];
        return (j.stations || []).filter(s => s.lat != null && s.lon != null);
      } catch {
        return [];
      }
    }

    function clearStationLayer(){
      if (!stationLayer) return;
      stationLayer.clearLayers();
      stationMarkers = [];
    }

    async function drawStationLayer(){
      if (!mapObj || !stationLayer) return;
      clearStationLayer();
      const stations = await loadStations();
      stations.forEach(s => {
        // 타깃 지점(1007639)은 별도 강조 마커가 이미 있으므로 레이어 중복 표시 제외
        if (s.is_target) return;
        const kind = s.is_target ? "target" : "ctx";
        const mk = L.marker([s.lat, s.lon], { icon: makeDotIcon(kind) });
        mk.bindPopup(`<b>${s.name}</b><br/>관측코드: ${s.code}`);
        stationLayer.addLayer(mk);
        stationMarkers.push(mk);
      });
    }

    async function initMap(){
      if (mapObj) return;
      mapObj = L.map("mapView", { zoomControl: true, attributionControl: true }).setView([37.30, 127.64], 11);
      L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
        maxZoom: 18,
        attribution: "&copy; OpenStreetMap contributors"
      }).addTo(mapObj);
      stationLayer = L.layerGroup().addTo(mapObj);
      await drawStationLayer();
      const toggle = el("toggleStations");
      toggle.checked = false;
      stationLayer.remove();
      toggle.addEventListener("change", async () => {
        if (toggle.checked) {
          if (!mapObj.hasLayer(stationLayer)) mapObj.addLayer(stationLayer);
          if (stationMarkers.length === 0) await drawStationLayer();
        } else {
          if (mapObj.hasLayer(stationLayer)) mapObj.removeLayer(stationLayer);
        }
      });
    }

    async function render(payload){
      await initMap();
      const base = new Date(payload.bucket_start_kst.replace(" ", "T"));
      const preds = payload.predictions_m || {};
      const meta = payload.meta || {};
      const current = meta.current_level_m;
      const warns = meta.warnings || [];
      const targetCode = meta.target_station_code || "1007639";
      const targetName = meta.target_station_name || "여주보 상류";
      const targetVar = meta.target_variable || "수위(m)";
      const targetScope = meta.prediction_scope || "지점 수위 예측";
      const damUsage = meta.dam_level_usage || "댐 수위는 입력 피처";

      // 타깃 지점(개략 좌표)
      const targetLat = 37.300;
      const targetLng = 127.636;
      if (targetMarker) mapObj.removeLayer(targetMarker);
      if (rangeCircle) mapObj.removeLayer(rangeCircle);
      targetMarker = L.marker([targetLat, targetLng], { icon: makeDotIcon("target") }).addTo(mapObj);
      const currTxt = (current == null) ? "-" : `${fmt(current)} m`;
      targetMarker.bindPopup(
        `<b>${targetName}</b><br/>관측코드: ${targetCode}<br/>현재 수위: ${currTxt}<br/>예측 변수: ${targetVar}`
      );
      rangeCircle = L.circle([targetLat, targetLng], {
        radius: 1200, color: "#ef4444", weight: 2, fillColor: "#ef4444", fillOpacity: 0.08
      }).addTo(mapObj);
      mapObj.setView([targetLat, targetLng], 12);

      const requested = meta.requested_bucket_kst || payload.bucket_start_kst;
      const generated = meta.generated_at_kst || "-";
      const kwObs = meta.kwater_latest_obs_kst || "-";
      const hrObs = meta.hrfc_latest_obs_kst || "-";
      el("meta").innerHTML =
        `요청시각(KST): <b>${requested}</b> | 기준버킷(KST): <b>${payload.bucket_start_kst}</b><br/>` +
        `생성시각: ${generated} | 화면시각: ${nowStr()}<br/>` +
        `최신 실측시각: K-water=${kwObs}, HRFC=${hrObs}<br/>` +
        `데이터 소스: K-water=${meta.kwater_source || "-"}, HRFC=${meta.hrfc_source || "-"}`;

      if (current != null && preds.h36_pred_m != null) {
        const d = preds.h36_pred_m - current;
        if (d > 0.01) el("trend").innerHTML = '<span class="chip up">6시간 기준 상승 가능성</span>';
        else if (d < -0.01) el("trend").innerHTML = '<span class="chip down">6시간 기준 하락 가능성</span>';
        else el("trend").innerHTML = '<span class="chip flat">6시간 기준 큰 변화 없음</span>';
      } else {
        el("trend").innerHTML = '<span class="chip flat">추세 판단 정보 부족</span>';
      }

      const warnBox = el("warnBox");
      if (warns.length === 0) {
        warnBox.className = "warn-box warn-ok";
        warnBox.innerHTML = "<b>알림</b><br/>현재 경고 없음";
      } else {
        warnBox.className = "warn-box warn-has";
        warnBox.innerHTML = "<b>알림</b><br/>" + warns.join("<br/>");
      }

      el("locTarget").innerHTML = `<b>타깃 지점:</b> ${targetName} (${targetCode})<br/><b>예측 변수:</b> ${targetVar}`;
      el("locScope").innerHTML = `<b>범위:</b> ${targetScope}`;
      el("locDamUsage").innerHTML = `<b>댐수위 관계:</b> ${damUsage}`;

      const cards = [];
      const rows = [];
      const labels = [], values = [], names = [], deltas = [], dlabels = [];
      if (current != null) {
        labels.push(payload.bucket_start_kst.slice(5,16));
        values.push(current);
        names.push("현재");
        rows.push(`<tr><td>현재</td><td>${payload.bucket_start_kst}</td><td>${fmt(current)}</td><td>-</td></tr>`);
      }
      for (const k of ORDER) {
        if (preds[k] == null) continue;
        const pred = preds[k];
        const t = new Date(base.getTime() + H_MINS[k]*60000);
        const tK = `${t.getFullYear()}-${String(t.getMonth()+1).padStart(2,"0")}-${String(t.getDate()).padStart(2,"0")} ${String(t.getHours()).padStart(2,"0")}:${String(t.getMinutes()).padStart(2,"0")}`;
        const d = (current == null) ? null : (pred - current);
        const dTxt = (d == null) ? "-" : `${d >= 0 ? "+" : ""}${fmt(d)} m`;
        let cls = "flat", icon = "→";
        if (d != null && d > 0.01) { cls = "up"; icon = "↑"; }
        else if (d != null && d < -0.01) { cls = "down"; icon = "↓"; }
        cards.push(`<div class="mini-card ${cls}"><div class="mini-title">${H_LABELS[k]}</div><div class="mini-value">${fmt(pred)} m</div><div class="mini-delta">${icon} ${dTxt}</div></div>`);
        rows.push(`<tr><td>${H_LABELS[k]}</td><td>${tK}</td><td>${fmt(pred)}</td><td>${dTxt}</td></tr>`);
        labels.push(`${String(t.getMonth()+1).padStart(2,"0")}-${String(t.getDate()).padStart(2,"0")} ${String(t.getHours()).padStart(2,"0")}:${String(t.getMinutes()).padStart(2,"0")}`);
        values.push(pred);
        names.push(H_LABELS[k]);
        dlabels.push(H_LABELS[k]);
        deltas.push(d == null ? 0 : d);
      }
      el("summaryCards").innerHTML = cards.join("");
      el("tb").innerHTML = rows.join("");

      const levelX = names.slice();
      const levelHover = labels.map((ts, i) => `${names[i]}<br>${ts}<br>${fmt(values[i])} m`);
      Plotly.newPlot("chart_level", [{
        x: levelX, y: values, mode: "lines+markers+text", text: names, textposition: "top center",
        line: {width:3, color:"#2563eb"},
        marker: {size:10, color:names.map(n => n === "현재" ? "#111827" : "#2563eb")},
        hovertemplate: "%{customdata}<extra></extra>",
        customdata: levelHover,
        name: "수위"
      }], {
        title: "수위 예측 경로 (현재 -> 6시간 후)",
        xaxis: {title: "예측 시점", type: "category", categoryorder: "array", categoryarray: levelX},
        yaxis: {title: "수위(m)", dtick: 0.01, tickformat: ".3f"},
        margin: {l:60,r:20,t:50,b:60}, paper_bgcolor:"#fff", plot_bgcolor:"#fff"
      }, {responsive:true, displaylogo:false});

      Plotly.newPlot("chart_delta", [{
        x: dlabels, y: deltas, type: "bar",
        marker: {color: deltas.map(v => v > 0.01 ? "#dc2626" : (v < -0.01 ? "#059669" : "#6b7280"))},
        text: deltas.map(v => `${v >= 0 ? "+" : ""}${fmt(v)} m`), textposition: "auto", name: "변화량"
      }], {
        title: "현재 대비 변화량",
        xaxis: {title: "예측 시점"},
        yaxis: {title: "변화량(m)", zeroline:true, zerolinecolor:"#9ca3af"},
        margin: {l:60,r:20,t:50,b:60}, paper_bgcolor:"#fff", plot_bgcolor:"#fff"
      }, {responsive:true, displaylogo:false});
    }

    async function loadLatest() {
      setLoading(true, "최신 데이터 불러오는 중...");
      try {
        const r = await fetch("/api/latest");
        const j = await r.json();
        if (!j.ok) throw new Error(j.error || "latest failed");
        await render(j.payload);
        setLoading(false, "최신 표시 완료");
      } catch (e) {
        setLoading(false, `오류: ${e.message}`);
      }
    }

    async function doRefresh() {
      setLoading(true, "실시간 API 재추론 중...");
      try {
        const r = await fetch("/api/refresh", {
          method: "POST",
          headers: {"Content-Type":"application/json"},
          body: JSON.stringify({skip_api: false})
        });
        const j = await r.json();
        if (!j.ok) throw new Error(j.error || "refresh failed");
        await render(j.payload);
        setLoading(false, "실시간 갱신 완료");
      } catch (e) {
        setLoading(false, `오류: ${e.message}`);
      }
    }

    el("btnRefresh").addEventListener("click", () => doRefresh());
    loadLatest();
  </script>
</body>
</html>"""


def _open_browser(port: int) -> None:
    time.sleep(1.0)
    webbrowser.open(f"http://127.0.0.1:{port}")


def _prewarm_assets() -> None:
    try:
        rp.get_spec()
        rp.get_feature_scaler()
        rp.get_target_scaler()
        rp.get_train_hist_sorted()
        rp.get_model_bundle()
        print("[prewarm] inference assets loaded")
    except Exception as e:
        print(f"[prewarm] skipped: {e}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8787)
    ap.add_argument("--no-open", action="store_true")
    args = ap.parse_args()

    if not args.no_open:
        t = threading.Thread(target=_open_browser, args=(args.port,), daemon=True)
        t.start()
    _prewarm_assets()
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
