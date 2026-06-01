"""
scripts/test_ollama.py
Task 1.3 산출물: Ollama REST API 기본 통신 테스트

실행:
    python scripts/test_ollama.py
    python scripts/test_ollama.py --url http://localhost:11434 --model qwen3:8b
"""

import argparse
import json
import sys
import time

import httpx


def check_server(base_url: str) -> bool:
    """Ollama 서버 가용성 확인."""
    print(f"[1/4] Ollama 서버 연결 확인: {base_url}")
    try:
        r = httpx.get(f"{base_url}/api/tags", timeout=5)
        r.raise_for_status()
        models = r.json().get("models", [])
        print(f"      ✅ 연결 성공 | 로드된 모델 수: {len(models)}")
        for m in models:
            print(f"         - {m.get('name', '?')}")
        return True
    except Exception as e:
        print(f"      ❌ 연결 실패: {e}")
        return False


def check_model(base_url: str, model: str) -> bool:
    """지정 모델이 로드되어 있는지 확인."""
    print(f"\n[2/4] 모델 확인: {model}")
    try:
        r = httpx.get(f"{base_url}/api/tags", timeout=5)
        models = [m["name"] for m in r.json().get("models", [])]
        # qwen3:8b 와 qwen3:8b-q4_K_M 등 변형 모두 허용
        found = any(model.split(":")[0] in m for m in models)
        if found:
            print(f"      ✅ 모델 확인됨")
            return True
        else:
            print(f"      ⚠️  모델 없음. Pull 시작...")
            r2 = httpx.post(
                f"{base_url}/api/pull",
                json={"name": model, "stream": False},
                timeout=300,
            )
            r2.raise_for_status()
            print(f"      ✅ Pull 완료")
            return True
    except Exception as e:
        print(f"      ❌ 모델 확인 실패: {e}")
        return False


def test_generate(base_url: str, model: str) -> bool:
    """/api/generate 기본 호출 테스트."""
    print(f"\n[3/4] /api/generate 호출 테스트")
    payload = {
        "model": model,
        "prompt": "안녕하세요. 여주보 수위 예측 시스템입니다. '테스트 완료'라고만 답하세요.",
        "stream": False,
        "options": {"temperature": 0.1, "num_predict": 30},
    }
    try:
        start = time.time()
        r = httpx.post(f"{base_url}/api/generate", json=payload, timeout=60)
        r.raise_for_status()
        elapsed = time.time() - start
        resp_text = r.json().get("response", "").strip()
        print(f"      ✅ 응답 수신 | {elapsed:.1f}초")
        print(f"         응답: {resp_text[:80]}")
        return True
    except Exception as e:
        print(f"      ❌ /api/generate 실패: {e}")
        return False


def test_chat(base_url: str, model: str) -> bool:
    """/api/chat 수위 도메인 호출 테스트 (실제 사용 엔드포인트)."""
    print(f"\n[4/4] /api/chat 수위 도메인 테스트")
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "당신은 수자원 관리 전문 AI입니다. 간결하게 답변하십시오.",
            },
            {
                "role": "user",
                "content": (
                    "여주보 현재 수위: 5.23m, 예측 수위: 5.87m, 경보 단계: 0(정상).\n"
                    "한 문장으로 현황을 요약하십시오."
                ),
            },
        ],
        "stream": False,
        "options": {"temperature": 0.3, "num_predict": 100},
    }
    try:
        start = time.time()
        r = httpx.post(f"{base_url}/api/chat", json=payload, timeout=60)
        r.raise_for_status()
        elapsed = time.time() - start
        content = r.json().get("message", {}).get("content", "").strip()
        print(f"      ✅ 응답 수신 | {elapsed:.1f}초")
        print(f"         응답: {content[:120]}")
        if elapsed > 5:
            print(f"      ⚠️  응답 시간 {elapsed:.1f}초 — WBS 목표(5초) 초과")
        return True
    except Exception as e:
        print(f"      ❌ /api/chat 실패: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Ollama 통신 테스트")
    parser.add_argument("--url", default="http://localhost:11434")
    parser.add_argument("--model", default="qwen3:8b")
    args = parser.parse_args()

    print("=" * 55)
    print("  Ollama REST API 통신 테스트  (Task 1.3)")
    print("=" * 55)

    results = [
        check_server(args.url),
        check_model(args.url, args.model),
        test_generate(args.url, args.model),
        test_chat(args.url, args.model),
    ]

    print("\n" + "=" * 55)
    passed = sum(results)
    total = len(results)
    status = "✅ 전체 통과" if passed == total else f"⚠️  {passed}/{total} 통과"
    print(f"  결과: {status}")
    print("=" * 55)
    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()
