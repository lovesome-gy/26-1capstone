# 여주보 실시간 실측 → v2(72열) → 모델 추론 파이프라인

## 목적

- 최신 실측(API)을 학습 기준(v2 72열)으로 맞춰 단건 추론
- 학습과 같은 윈도우 길이(72 스텝, 12시간)로 모델 입력

## 파일

- `.env`
  - API 키/댐코드/옵션 설정
- `test_realtime_env.py`
  - 키/환경 변수 형식 검증(1회)
- `realtime_pipeline_v2.py`
  - API 수집 → 72열 변환 → 추론 1회

## 1회 검증 순서

```powershell
cd <project-root>/03_realtime_pipeline
python test_realtime_env.py
```

정상 출력 확인 후:

```powershell
python realtime_pipeline_v2.py
```

성공 시 아래 파일이 생성됩니다:

- `data/realtime_latest_prediction.json`

## 시각 확인(간단 대시보드)

한 번 실행하면 실시간 추론 후 그래프 HTML이 생성되고 자동으로 열립니다.

```powershell
.venv\Scripts\python realtime_visual_check.py
```

또는 배치 파일:

```powershell
.\open_realtime_visual_check.bat
```

생성 파일:

- `data/realtime_visual_check.html`

## 갱신 버튼 포함 대시보드(권장)

버튼으로 실시간 재추론을 수행하려면 로컬 서버 버전을 사용하세요.

```powershell
.venv\Scripts\python realtime_dashboard_server.py
```

또는:

```powershell
.\open_realtime_dashboard_server.bat
```

접속 주소:

- `http://127.0.0.1:8787`

버튼 기능:

- `갱신 (실시간 재추론)` : API 호출 포함 최신화
- `빠른점검 (API 생략)` : 모델 경로/화면만 빠르게 점검

## API 연결 점검이 아직 필요할 때

API 호출 없이 파이프라인 구조/모델 추론만 먼저 점검:

```powershell
.venv\Scripts\python realtime_pipeline_v2.py --skip-api
```

## .env 주요 키

- `DATA_GO_KR_SERVICE_KEY` (K-water 공공데이터포털 키)
- `HRFCO_SERVICE_KEY` (한강홍수통제소 키)
- `KWATER_DAM_CODE_YEOJU` (여주보 댐코드)

선택:

- `HRFCO_WLOBSCD_LIST` (콤마 구분 관측소 코드 목록)
- `KWATER_API_URL`, `HRFCO_API_URL` (기본 경로 실패 시 오버라이드)

HRFCO 기본 형식:

- 최신 전체: `https://api.hrfco.go.kr/{ServiceKey}/waterlevel/list/10M.json`
- 최신 단일 관측소: `https://api.hrfco.go.kr/{ServiceKey}/waterlevel/list/10M/{WLOBSCD}.json`

`HRFCO_API_URL` 오버라이드 사용 시:

- 템플릿 방식 지원: `https://api.hrfco.go.kr/{key}/waterlevel/list/10M/{sid}.json`

## 구현 기준

- 입력 열수: `docs/preprocess_v2_spec.json` 의 `feature_columns` 72열
- 윈도우: `config_v2.py` 의 `SEQ_LOOKBACK=72`
- 모델: `models/hydro_mast_v2.pt`

## 주의

- `.env` 는 로컬 전용(커밋 금지)
- API 응답 필드명이 환경마다 다를 수 있어 기본 경로가 실패하면 `.env` 에 URL 오버라이드 필요
