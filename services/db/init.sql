-- ============================================================
-- 여주보 수위 예측 AI 시스템 - DB 초기화 스크립트
-- TimescaleDB 확장 + 전체 테이블 정의
-- ============================================================

-- TimescaleDB 확장 활성화
CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;

-- ============================================================
-- [Part①/②] 원시 수위 관측 데이터 (10분 단위 시계열)
-- 출처: 한강홍수통제소(HRFC), K-water
-- ============================================================
CREATE TABLE IF NOT EXISTS water_level_raw (
    time            TIMESTAMPTZ     NOT NULL,               -- KST 기준 10분 슬롯
    station_id      VARCHAR(20)     NOT NULL,               -- 관측소 ID
    source          VARCHAR(10)     NOT NULL,               -- 'hrfc' | 'kwater'
    water_level_m   DOUBLE PRECISION,                       -- 수위 (m)
    flow_rate_cms   DOUBLE PRECISION,                       -- 유량 (m³/s)
    rainfall_mm     DOUBLE PRECISION,                       -- 강우량 (mm)
    gate_open_pct   DOUBLE PRECISION,                       -- 수문 개방도 (%)
    created_at      TIMESTAMPTZ     DEFAULT NOW()
);

-- TimescaleDB 하이퍼테이블 변환 (시계열 최적화)
SELECT create_hypertable('water_level_raw', 'time', if_not_exists => TRUE);

CREATE INDEX IF NOT EXISTS idx_water_level_raw_station
    ON water_level_raw (station_id, time DESC);

-- ============================================================
-- [Part②/③] 모델 학습/평가 메타데이터
-- ============================================================
CREATE TABLE IF NOT EXISTS model_registry (
    model_id        SERIAL          PRIMARY KEY,
    model_name      VARCHAR(50)     NOT NULL,               -- 'LSTM' | 'GRU' | 'Transformer'
    version         VARCHAR(20)     NOT NULL,               -- 'v1.0.0'
    architecture    JSONB           NOT NULL,               -- 하이퍼파라미터 전체
    train_period    TSTZRANGE,                              -- 학습 기간
    rmse            DOUBLE PRECISION,                       -- 평가 지표
    mae             DOUBLE PRECISION,
    r2_score        DOUBLE PRECISION,
    accuracy_pct    DOUBLE PRECISION,                       -- 목표: 90% 이상
    artifact_path   TEXT,                                   -- 모델 파일 경로
    is_active       BOOLEAN         DEFAULT FALSE,          -- 현재 서빙 중인 모델
    created_at      TIMESTAMPTZ     DEFAULT NOW(),
    UNIQUE (model_name, version)
);

-- ============================================================
-- [Part③] 실시간 수위 예측 결과
-- ============================================================
CREATE TABLE IF NOT EXISTS prediction_result (
    prediction_id   BIGSERIAL       PRIMARY KEY,
    predicted_at    TIMESTAMPTZ     NOT NULL DEFAULT NOW(),-- 예측 실행 시각
    target_time     TIMESTAMPTZ     NOT NULL,               -- 예측 대상 시각
    station_id      VARCHAR(20)     NOT NULL,
    model_id        INTEGER         REFERENCES model_registry(model_id),
    predicted_level DOUBLE PRECISION NOT NULL,              -- 예측 수위 (m)
    actual_level    DOUBLE PRECISION,                       -- 실제 수위 (사후 기입)
    confidence_low  DOUBLE PRECISION,                       -- 신뢰구간 하한
    confidence_high DOUBLE PRECISION,                       -- 신뢰구간 상한
    horizon_minutes INTEGER         NOT NULL DEFAULT 60,    -- 예측 선행 시간 (분)
    input_window    JSONB                                   -- 입력으로 사용한 과거 데이터 요약
);

CREATE INDEX IF NOT EXISTS idx_prediction_result_time
    ON prediction_result (target_time DESC, station_id);

-- ============================================================
-- [Part④ - 신가연] 자연어 보고서 생성 테이블
-- LLM이 생성한 수위 현황 보고서를 저장한다.
-- ============================================================
CREATE TABLE IF NOT EXISTS make_report_tb (
    report_id       BIGSERIAL       PRIMARY KEY,
    created_at      TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    station_id      VARCHAR(20)     NOT NULL DEFAULT '3008680',  -- 여주보
    report_period   TSTZRANGE       NOT NULL,               -- 보고서가 다루는 기간
    report_type     VARCHAR(20)     NOT NULL,               -- 'hourly'|'daily'|'alert'
    water_level_cur DOUBLE PRECISION,                       -- 현재 수위 (m)
    water_level_pred DOUBLE PRECISION,                      -- 예측 수위 (m)
    trend           VARCHAR(10),                            -- 'rising'|'falling'|'stable'
    alert_level     SMALLINT        DEFAULT 0,              -- 0:정상 1:관심 2:주의 3:경계 4:심각
    -- LLM 생성 본문
    report_summary  TEXT            NOT NULL,               -- 1~2줄 요약
    report_body     TEXT            NOT NULL,               -- 전체 보고서 본문
    -- 메타
    llm_model       VARCHAR(50)     DEFAULT 'qwen3:8b',
    prompt_version  VARCHAR(20),                            -- 프롬프트 버전 관리
    generation_ms   INTEGER,                                -- LLM 응답 소요 시간(ms)
    prediction_id   BIGINT          REFERENCES prediction_result(prediction_id)
);

CREATE INDEX IF NOT EXISTS idx_make_report_tb_created
    ON make_report_tb (created_at DESC);
CREATE INDEX IF NOT EXISTS idx_make_report_tb_alert
    ON make_report_tb (alert_level, created_at DESC);

COMMENT ON TABLE make_report_tb IS
    '[Part④] LLM 기반 수위 현황 자연어 보고서 - 신가연';

-- ============================================================
-- [Part④ - 신가연] AI 의사결정 지원 테이블
-- LLM이 생성한 수문 조작/대피 등 의사결정 가이드를 저장한다.
-- ============================================================
CREATE TABLE IF NOT EXISTS decision_support_tb (
    decision_id     BIGSERIAL       PRIMARY KEY,
    created_at      TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    station_id      VARCHAR(20)     NOT NULL DEFAULT '3008680',
    alert_level     SMALLINT        NOT NULL,               -- make_report_tb와 동기화
    -- 의사결정 항목
    action_category VARCHAR(30)     NOT NULL,               -- 'gate_control'|'evacuation'|'monitoring'
    priority        SMALLINT        NOT NULL DEFAULT 2,     -- 1:긴급 2:일반 3:참고
    -- LLM 생성 본문
    decision_title  VARCHAR(200)    NOT NULL,               -- 조치 제목
    decision_body   TEXT            NOT NULL,               -- 상세 조치 내용
    rationale       TEXT,                                   -- 판단 근거 (LLM 설명)
    -- 연계 정보
    report_id       BIGINT          REFERENCES make_report_tb(report_id),
    prediction_id   BIGINT          REFERENCES prediction_result(prediction_id),
    -- 메타
    llm_model       VARCHAR(50)     DEFAULT 'qwen3:8b',
    prompt_version  VARCHAR(20),
    generation_ms   INTEGER,
    is_acknowledged BOOLEAN         DEFAULT FALSE,          -- 담당자 확인 여부
    acknowledged_at TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_decision_support_tb_created
    ON decision_support_tb (created_at DESC);
CREATE INDEX IF NOT EXISTS idx_decision_support_tb_priority
    ON decision_support_tb (priority, alert_level, created_at DESC);

COMMENT ON TABLE decision_support_tb IS
    '[Part④] LLM 기반 AI 의사결정 지원 - 신가연';

-- ============================================================
-- 초기 데이터: 여주보 관측소 정보
-- ============================================================
CREATE TABLE IF NOT EXISTS station_info (
    station_id      VARCHAR(20)     PRIMARY KEY,
    station_name    VARCHAR(100)    NOT NULL,
    river_name      VARCHAR(50),
    source          VARCHAR(10),
    latitude        DOUBLE PRECISION,
    longitude       DOUBLE PRECISION,
    alert_level_1   DOUBLE PRECISION,                       -- 관심 수위 (m)
    alert_level_2   DOUBLE PRECISION,                       -- 주의 수위 (m)
    alert_level_3   DOUBLE PRECISION,                       -- 경계 수위 (m)
    alert_level_4   DOUBLE PRECISION                        -- 심각 수위 (m)
);

INSERT INTO station_info VALUES
    ('3008680', '여주보', '남한강', 'hrfc', 37.2985, 127.6378, 6.0, 7.5, 9.0, 10.5)
ON CONFLICT (station_id) DO NOTHING;
