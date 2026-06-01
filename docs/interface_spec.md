# LLM Service 인터페이스 명세서
**Part④ 담당: 신가연 | 버전: v1.0 | 최종 수정: 2026-05**

---

## 1. 서비스 개요

| 항목 | 내용 |
|------|------|
| 서비스명 | LLM Service (여주보 수위 예측 AI 보고서·의사결정 지원) |
| 포트 | 8002 |
| LLM | Ollama / Qwen3-8b (온프레미스) |
| DB | PostgreSQL (TimescaleDB) |
| 주요 테이블 | `make_report_tb`, `decision_support_tb` |

---

## 2. 보고서 생성 API

### POST `/api/v1/reports/generate`

수위 예측 데이터를 입력받아 LLM 자연어 보고서를 생성하고 DB에 저장한다.

**요청 본문 (JSON)**

| 필드 | 타입 | 필수 | 설명 |
|------|------|------|------|
| `station_id` | string | 선택 | 관측소 ID (기본: "3008680") |
| `report_type` | string | 선택 | `hourly`\|`daily`\|`weekly`\|`monthly`\|`alert` |
| `water_level_cur` | float | **필수** | 현재 수위 (m) |
| `water_level_pred` | float | **필수** | 예측 수위 (m) |
| `period_start` | datetime | **필수** | 보고 기간 시작 (ISO8601+KST) |
| `period_end` | datetime | **필수** | 보고 기간 종료 (ISO8601+KST) |
| `prediction_id` | int | 선택 | 연계 예측 레코드 ID |
| `avg_level` | float | 선택 | 기간 평균 수위 (daily/weekly/monthly용) |
| `max_level` | float | 선택 | 기간 최고 수위 |
| `min_level` | float | 선택 | 기간 최저 수위 |
| `alert_count` | int | 선택 | 기간 경보 발생 횟수 |

**요청 예시**
```json
{
  "station_id": "3008680",
  "report_type": "hourly",
  "water_level_cur": 5.23,
  "water_level_pred": 5.87,
  "period_start": "2025-07-15T14:00:00+09:00",
  "period_end": "2025-07-15T15:00:00+09:00",
  "prediction_id": 42
}
```

**응답 본문 (JSON)**

| 필드 | 타입 | 설명 |
|------|------|------|
| `report_id` | int | 생성된 보고서 ID |
| `created_at` | datetime | 생성 시각 |
| `alert_level` | int | 0:정상 1:관심 2:주의 3:경계 4:심각 |
| `trend` | string | `rising`\|`falling`\|`stable` |
| `report_summary` | string | 1~2문장 핵심 요약 |
| `report_body` | string | 전체 보고서 본문 |
| `generation_ms` | int | LLM 응답 소요 시간(ms) |

---

### GET `/api/v1/reports/`

최근 보고서 목록 조회.

**쿼리 파라미터**

| 파라미터 | 기본값 | 설명 |
|---------|--------|------|
| `station_id` | "3008680" | 관측소 ID |
| `limit` | 10 | 최대 조회 건수 (1~100) |
| `alert_level` | (없음) | 특정 경보 단계 필터 |

---

### GET `/api/v1/reports/{report_id}`

보고서 단건 조회.

---

## 3. 의사결정 지원 API

### POST `/api/v1/decisions/generate`

경보 단계와 수위 데이터를 입력받아 LLM 의사결정 지원 항목(1~3개)을 생성한다.

**요청 본문 (JSON)**

| 필드 | 타입 | 필수 | 설명 |
|------|------|------|------|
| `station_id` | string | 선택 | 관측소 ID |
| `alert_level` | int | **필수** | 0~4 경보 단계 |
| `water_level_cur` | float | **필수** | 현재 수위 (m) |
| `water_level_pred` | float | **필수** | 예측 수위 (m) |
| `trend` | string | 선택 | `rising`\|`falling`\|`stable` |
| `report_id` | int | 선택 | 연계 보고서 ID |
| `prediction_id` | int | 선택 | 연계 예측 ID |

**응답 본문 (JSON)**

```json
{
  "total": 2,
  "items": [
    {
      "decision_id": 1,
      "action_category": "gate_control",
      "priority": 1,
      "decision_title": "수문 개방도 조정 검토",
      "decision_body": "현재 수위 7.8m로 주의 단계...",
      "rationale": "상승 추세 지속 시 경계 단계 진입 우려",
      "is_acknowledged": false
    }
  ]
}
```

---

### PATCH `/api/v1/decisions/acknowledge`

담당자 확인 처리.

**요청**: `{ "decision_id": 1 }`

---

## 4. 경보 단계 기준 (여주보)

| 단계 | 라벨 | 수위 기준 |
|------|------|----------|
| 0 | 정상 | 6.0m 미만 |
| 1 | 관심 | 6.0 ~ 7.5m |
| 2 | 주의 | 7.5 ~ 9.0m |
| 3 | 경계 | 9.0 ~ 10.5m |
| 4 | 심각 | 10.5m 이상 |

---

## 5. 보고서 유형별 입력 가이드

| 유형 | 용도 | 추가 필드 |
|------|------|----------|
| `hourly` | 1시간 실시간 현황 | 없음 |
| `daily` | 일간 통계 요약 | `avg_level`, `max_level`, `min_level`, `alert_count` |
| `weekly` | 주간 추세 분석 | `avg_level`, `max_level`, `min_level`, `alert_count` |
| `monthly` | 월간 종합 보고 | `avg_level`, `max_level`, `min_level`, `alert_count` |
| `alert` | 경보 발생 즉시 | 없음 (긴급) |

---

## 6. 헬스체크

### GET `/health`

```json
{
  "status": "ok",
  "service": "llm_service",
  "ollama_available": true,
  "ollama_model": "qwen3:8b"
}
```

---

## 7. 테스트 스크립트

| 스크립트 | 목적 | WBS |
|---------|------|-----|
| `scripts/test_ollama.py` | Ollama 통신 테스트 | Task 1.3 |
| `scripts/test_api.py` | FastAPI E2E 테스트 | Task 5.3 |
| `scripts/check_performance.py` | 추론 속도·품질 검증 | Task 6.1 |
| `scripts/validate_pipeline.py` | 전체 파이프라인 통합 | Task 6.2 |

```bash
# 실행 순서
python scripts/test_ollama.py
python scripts/test_api.py
python scripts/check_performance.py --runs 5
python scripts/validate_pipeline.py
```
