# -*- coding: utf-8 -*-
"""v2 전처리·모델 공통 설정 (논문 기반)"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent  # 완성ai모델 루트
HANDOFF = PROJECT / "yeoju_ai_handoff_2024_2025_KST"

DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
DOCS_DIR = ROOT / "docs"

# 예측 대상: HRFC 여주보(상류) 1007639
TARGET_COL = "hrfc_1007639_수위_m"
TARGET_FALLBACK = "kwater_댐수위_m"

# 10분 간격 lag (CNN-GRU·기존 파이프라인: 1h/3h/6h/12h)
LAG_STEPS = [6, 18, 36, 72]
# 선행 예측: 6시간 후 (10분×36) — 권혜수(2023) CNN-GRU 논문
HORIZON_STEPS = 36

# 통합 모델: 한 번에 예측할 시점 (10분 스텝) — 단기+장기
# 1=10분, 6=1시간, 18=3시간, 36=6시간 (다변량 3h lead + CNN-GRU 6h)
MULTI_HORIZONS = [1, 6, 18, 36]
MULTI_HORIZON_LABELS = {1: "10분", 6: "1시간", 18: "3시간", 36: "6시간"}
# 시계열 윈도우 (다변량 2024: 과거 36시간 → 10분이면 216, 단 계산 부담으로 72=12h)
SEQ_LOOKBACK = 72

# 학습 데이터 구간 (2년 전체)
DATA_START = "2024-01-01 00:00:00"
TRAIN_END = "2025-12-31 23:50:00"
# 성능 지표용: 학습 구간 말미 10% (모델 학습 CSV에는 미포함)
VAL_HOLDOUT_FRACTION = 0.1
# 결측·이상치 (yeoju handoff + 수문결측 LSTM 2022 BoxPlot)
IQR_K = 1.5
INTERP_LIMIT = 6  # 10분×6 = 1시간
FFILL_LIMIT = 18  # 3시간

# lag·차분·시간부호화 대상
KEY_SERIES = [
    TARGET_COL,
    "kwater_댐수위_m",
    "kwater_강우량_mm",
    "kwater_유입량_m3s",
    "kwater_총방류량_m3s",
    "hrfc_1007641_수위_m",
    "hrfc_수위_m_평균",
]

HRFC_STATIONS = {
    "1007639": "여주보_상류",
    "1007641": "여주보_하류",
    "1007664": "이포보_하류",
    "1007662": "이포보_상류",
    "1007660": "강천",
    "1007656": "세종대교",
    "1007655": "세종교",
    "1007650": "도덕",
    "1007640": "강천2",
    "1007637": "원덕",
    "1007635": "양평",
    "1007633": "운길",
    "1007626": "복하천",
    "1007625": "복하천2",
    "1007620": "흥천",
    "1007617": "흥천2",
    "1007615": "흥천3",
}
