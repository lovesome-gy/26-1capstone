# -*- coding: utf-8 -*-
r"""
키/환경 검증 1회 스크립트.

실행:
  .venv\Scripts\python test_realtime_env.py
"""
from __future__ import annotations

from pathlib import Path
from urllib.parse import quote_plus
from urllib.parse import unquote
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parent
ENV_PATH = ROOT / ".env"


def load_env(path: Path) -> dict[str, str]:
    if not path.exists():
        raise FileNotFoundError(f".env not found: {path}")
    out: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#") or "=" not in s:
            continue
        k, v = s.split("=", 1)
        out[k.strip()] = v.strip()
    return out


def validate(ping: bool = False) -> None:
    env = load_env(ENV_PATH)
    required = [
        "DATA_GO_KR_SERVICE_KEY",
        "HRFCO_SERVICE_KEY",
        "KWATER_DAM_CODE_YEOJU",
    ]
    missing = [k for k in required if not env.get(k)]
    if missing:
        raise RuntimeError(f"필수 키 누락: {missing}")

    data_go = env["DATA_GO_KR_SERVICE_KEY"]
    data_go_dec = unquote(data_go) if "%" in data_go else data_go
    hrfco = env["HRFCO_SERVICE_KEY"]
    dam_code = env["KWATER_DAM_CODE_YEOJU"]

    print("=" * 60)
    print("ENV validation OK")
    print(f"DATA_GO_KR_SERVICE_KEY: len(raw)={len(data_go)}, len(decoded)={len(data_go_dec)}")
    print(f"HRFCO_SERVICE_KEY: len={len(hrfco)}")
    print(f"KWATER_DAM_CODE_YEOJU: {dam_code}")
    print("KMA_SERVICE_KEY:", "set" if env.get("KMA_SERVICE_KEY") else "not set (optional)")
    print("HRFCO_WLOBSCD_LIST:", env.get("HRFCO_WLOBSCD_LIST", "(auto)") or "(auto)")
    print("=" * 60)

    if ping:
        day = "2025-12-31"
        key = env["DATA_GO_KR_SERVICE_KEY"]
        dam = env["KWATER_DAM_CODE_YEOJU"]
        urls = [
            "http://apis.data.go.kr/B500001/dam/sluicePresentCondition/hourlist",
            "https://apis.data.go.kr/B500001/dam/sluicePresentCondition/hourlist",
            "https://apis.data.go.kr/B500001/dam/sluicePresentCondition/minlist",
            "http://apis.data.go.kr/B500001/dam/sluicePresentCondition/minlist",
        ]
        print("[ping] K-water endpoints")
        for u in urls:
            q = (
                f"serviceKey={key}&pageNo=1&numOfRows=5&_type=json"
                f"&damcode={quote_plus(str(dam))}&stdt={day}&eddt={day}"
            )
            full = f"{u}?{q}"
            try:
                with urlopen(full, timeout=10) as r:
                    body = r.read().decode("utf-8", errors="ignore")
                print(f"  OK  {u}  bytes={len(body)}")
            except Exception as e:
                print(f"  FAIL {u}  err={e}")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--ping", action="store_true", help="K-water 기본 엔드포인트 1회 연결 테스트")
    args = ap.parse_args()
    validate(ping=args.ping)
