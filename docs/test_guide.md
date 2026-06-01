# 🎬 여주보 수위 예측 AI 시스템 — 시연 영상 제작 가이드

> 이 가이드는 팀원이 로컬 환경에서 시스템을 실행하고 시연 영상을 촬영할 수 있도록 작성되었습니다.
> **작성: 신가연**

---

## ✅ 사전 준비 (최초 1회)

### 1. 필수 프로그램 설치

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) 설치 및 실행
- [Git](https://git-scm.com/downloads) 설치
- Windows PowerShell (기본 설치됨)

### 2. 레포지토리 클론

```bash
git clone https://github.com/lovesome-gy/26-1capstone.git
cd 26-1capstone
```

### 3. 환경변수 파일 생성

`.env.example` 파일을 복사해서 `.env` 만들기:

```bash
copy .env.example .env
```

`.env` 파일을 메모장으로 열어서 아래 값 입력:

```
POSTGRES_PASSWORD=thisisfuckingcapstoneproject
DATABASE_URL=postgresql+asyncpg://yeoju_admin:thisisfuckingcapstoneproject@postgres:5432/yeoju_water
HRFCO_SERVICE_KEY=B7E9353B-67B5-4213-A636-07E78443FC80
DATA_GO_KR_SERVICE_KEY=DOu9PdLU0IElgUcOuDgZrmMozs1V9ry7%2Fws2bmnni%2F%2B%2BjPdWedQ1zu8WO%2FwdbZZYTvPb0JyAsVUiO0n66WqBlQ%3D%3D
OLLAMA_TIMEOUT=300
```

> ⚠️ `.env` 파일은 깃허브에 올리면 안 됩니다. `.gitignore`에 이미 등록되어 있어 자동으로 제외됩니다.

### 4. Hydro-MAST 코드 배치 (정휘수)

```bash
cd services\hydro_mast
git clone https://github.com/hwisu-jung/flood-forecast-project .
cd ..\..
```

그 다음 `services\hydro_mast\.env` 파일 생성:

```bash
copy services\hydro_mast\.env.example services\hydro_mast\.env
```

`services\hydro_mast\.env` 메모장으로 열어서:

```
DATA_GO_KR_SERVICE_KEY=DOu9PdLU0IElgUcOuDgZrmMozs1V9ry7%2Fws2bmnni%2F%2B%2BjPdWedQ1zu8WO%2FwdbZZYTvPb0JyAsVUiO0n66WqBlQ%3D%3D
HRFCO_SERVICE_KEY=B7E9353B-67B5-4213-A636-07E78443FC80
KWATER_DAM_CODE_YEOJU=1007602
```

### 5. LSTM/XGB 모델 파일 배치 (김민준)

[AI_LSTM_XGB](https://github.com/bird539/AI_LSTM_XGB) 레포의 `Server/` 폴더에서 아래 파일들을 `services\lstm_xgb\models\` 에 복사:

```
total_scaler.pkl
target_scaler.pkl
yeoju_lstm_model.keras
yeoju_lstm_model_1h.keras
yeoju_lstm_model_3h.keras
yeoju_xgb_model.pkl
yeoju_xgb_model_1h.pkl
yeoju_xgb_model_3h.pkl
```

또는 터미널에서 자동 클론:

```bash
cd services\lstm_xgb
git clone --no-checkout https://github.com/bird539/AI_LSTM_XGB temp
cd temp
git sparse-checkout init --cone
git sparse-checkout set Server
git checkout main
cd ..
xcopy /Y "temp\Server\*.pkl" "models\"
xcopy /Y "temp\Server\*.keras" "models\"
Remove-Item -Recurse -Force temp
cd ..\..
```

---

## ▶️ 매번 시연 시 실행 순서

### Step 1. Docker Desktop 실행

Docker Desktop 아이콘을 더블클릭해서 실행하고 **고래 아이콘**이 작업표시줄에 뜰 때까지 기다립니다.

### Step 2. 터미널에서 프로젝트 폴더 이동

```bash
cd "C:\Users\사용자이름\Desktop\26-1capstone"
```

### Step 3. 서비스 실행

아래 명령어를 **순서대로** 입력하세요:

```bash
docker compose up postgres ollama llm_service -d
```

✅ `yeoju_postgres Healthy`, `yeoju_ollama Healthy` 확인 후 (약 30초 대기):

```bash
docker compose up hydro_mast -d
```

✅ hydro_mast가 뜨면 (약 10초 대기):

```bash
docker compose up predictor frontend data_collector lstm_xgb -d --build
```

### Step 4. Hydro-MAST 초기 설정 (최초 1회 또는 재시작 후)

```bash
docker exec yeoju_hydro_mast cp /app/.env.example /app/.env
docker exec yeoju_hydro_mast sh -c "echo 'DATA_GO_KR_SERVICE_KEY=DOu9PdLU0IElgUcOuDgZrmMozs1V9ry7%2Fws2bmnni%2F%2B%2BjPdWedQ1zu8WO%2FwdbZZYTvPb0JyAsVUiO0n66WqBlQ%3D%3D' > /app/.env && echo 'HRFCO_SERVICE_KEY=B7E9353B-67B5-4213-A636-07E78443FC80' >> /app/.env && echo 'KWATER_DAM_CODE_YEOJU=1007602' >> /app/.env"
docker exec yeoju_hydro_mast mkdir -p /app/04_artifacts/data
```

features_v2_train.csv 파일을 컨테이너에 복사 (파일이 `services\hydro_mast\04_artifacts\data\` 에 있어야 함):

```bash
docker cp "services\hydro_mast\04_artifacts\data\features_v2_train.csv" yeoju_hydro_mast:/app/04_artifacts/data/features_v2_train.csv
docker exec yeoju_hydro_mast cp /app/04_artifacts/data/features_v2_train.csv /app/04_artifacts/data/features_v2_test.csv
```

### Step 5. LSTM/XGB 초기 데이터 수집 (최초 1회)

```bash
docker exec -e HRFCO_SERVICE_KEY=B7E9353B-67B5-4213-A636-07E78443FC80 yeoju_lstm_xgb python -m app.hrfco_collector --init
```

### Step 6. LLM 준비 완료 확인

```bash
docker compose logs llm_service --tail=5
```

`LLM Service 준비 완료` 메시지가 보이면 완료입니다.

### Step 7. 브라우저 접속

```
http://localhost:8501
```

---

## 🎥 시연 순서 (영상 촬영 가이드)

### 화면 1: 실시간 현황 탭
1. `📊 실시간 현황` 탭 클릭
2. **현황 조회** 버튼 클릭
3. 경보 단계 배너 + 수위 지표 카드 + 다지평 예측 차트 확인

### 화면 2: 모델 비교 탭
1. `🤖 모델 비교` 탭 클릭
2. **두 모델 동시 조회** 버튼 클릭
3. Hydro-MAST(왼쪽) vs LSTM/XGB(오른쪽) 비교 화면 확인

### 화면 3: AI 보고서 생성 탭
1. `📄 보고서 생성` 탭 클릭
2. 현재 수위 / 예측 수위 입력 (예: 5.23 / 5.87)
3. **보고서 생성** 버튼 클릭
4. LLM이 생성한 한국어 보고서 확인 (1~5분 소요)

### 화면 4: 의사결정 지원 탭
1. `🧭 의사결정 지원` 탭 클릭
2. 수위를 경계 단계로 설정 (예: 현재 9.5m / 예측 10.2m, 추세: 상승)
3. **의사결정 생성** 버튼 클릭
4. 수문 제어 / 대피 조치 등 항목 확인

### 화면 5: 보고서 이력 탭
1. `📋 보고서 이력` 탭 클릭
2. **이력 불러오기** 버튼 클릭
3. 생성된 보고서 목록 확인

---

## 🔴 종료 방법

```bash
docker compose down
```

---

## ❓ 자주 발생하는 문제

| 증상 | 해결 방법 |
|---|---|
| `localhost:8501` 접속 안 됨 | `docker ps` 로 yeoju_frontend 실행 여부 확인 |
| 예측 서비스 연결 실패 | Step 4 Hydro-MAST 초기 설정 재실행 |
| 보고서 생성 500 에러 | `docker compose logs llm_service --tail=10` 확인 |
| LLM 타임아웃 | CPU 실행 시 5분 이상 소요 정상, 기다리면 됨 |
| ollama unhealthy | 실제 동작 중이므로 무시하고 진행 |

---

## 📞 문의

시연 중 문제 발생 시 팀장(신가연)에게 연락하세요.