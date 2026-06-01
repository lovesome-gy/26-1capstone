# -*- coding: utf-8 -*-
"""API/실시간 실행용 공통 설정 (배포 패키지 기준 경로)"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent

# 배포 패키지 표준 경로
DATA_DIR = PROJECT / "04_artifacts" / "data"
MODEL_DIR = PROJECT / "04_artifacts" / "models"
DOCS_DIR = PROJECT / "04_artifacts" / "docs"

TARGET_COL = "hrfc_1007639_수위_m"
TARGET_FALLBACK = "kwater_댐수위_m"

LAG_STEPS = [6, 18, 36, 72]
HORIZON_STEPS = 36
MULTI_HORIZONS = [1, 6, 18, 36]
MULTI_HORIZON_LABELS = {1: "10분", 6: "1시간", 18: "3시간", 36: "6시간"}
SEQ_LOOKBACK = 72

DATA_START = "2024-01-01 00:00:00"
TRAIN_END = "2025-12-31 23:50:00"
VAL_HOLDOUT_FRACTION = 0.1
IQR_K = 1.5
INTERP_LIMIT = 6
FFILL_LIMIT = 18

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
