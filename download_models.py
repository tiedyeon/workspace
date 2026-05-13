# -*- coding: utf-8 -*-
"""
download_models.py — 시계열 사전학습 모델 다운로드

대회 규정:
  공식 가중치 공개, 상업적/비상업적 이용 허용 라이선스만 사용 가능
  (MIT, Apache 2.0, CC BY, CC BY-NC 등)

다운로드 대상 (모두 Apache 2.0):
  1. Chronos-T5 small  (Amazon)
     - https://huggingface.co/amazon/chronos-t5-small
  2. TimesFM-200m       (Google)
     - https://huggingface.co/google/timesfm-1.0-200m
  3. Lag-Llama          (ServiceNow)
     - https://huggingface.co/time-series-foundation-models/Lag-Llama

⚠ 주의:
  - 이 대회는 sensor-local 3D 좌표 (multivariate). 위 모델들은 대부분 univariate.
  - 직접 적용보다 잔차 학습 / 앙상블 다양성 용으로 시도 가치.
  - 다운로드 안 해도 custom LSTM/Transformer 만으로 충분히 경쟁 가능.

실행:
  python download_models.py
  소요: 약 5~15분 (네트워크 속도 따라)
  용량: 약 2~3 GB
"""

from __future__ import annotations

import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


def download_chronos():
    print("\n[1/3] Chronos-T5 small (Amazon, Apache 2.0)...")
    try:
        from transformers import AutoConfig, AutoTokenizer, AutoModelForSeq2SeqLM
        model_name = "amazon/chronos-t5-small"
        print(f"  다운로드 시작: {model_name}")
        AutoConfig.from_pretrained(model_name)
        AutoTokenizer.from_pretrained(model_name)
        AutoModelForSeq2SeqLM.from_pretrained(model_name)
        print("  ✓ Chronos-T5 small 완료")
        return True
    except Exception as e:
        print(f"  ✗ 실패: {e}")
        return False


def download_timesfm():
    print("\n[2/3] TimesFM-1.0-200m (Google, Apache 2.0)...")
    try:
        from huggingface_hub import snapshot_download
        snapshot_download(repo_id="google/timesfm-1.0-200m", repo_type="model")
        print("  ✓ TimesFM 완료")
        return True
    except Exception as e:
        print(f"  ✗ 실패: {e}")
        print(f"    참고: TimesFM 은 별도 라이브러리 (timesfm) 필요할 수 있음")
        return False


def download_lag_llama():
    print("\n[3/3] Lag-Llama (ServiceNow, Apache 2.0)...")
    try:
        from huggingface_hub import snapshot_download
        snapshot_download(
            repo_id="time-series-foundation-models/Lag-Llama",
            repo_type="model",
        )
        print("  ✓ Lag-Llama 완료")
        return True
    except Exception as e:
        print(f"  ✗ 실패: {e}")
        return False


def main():
    print("=" * 64)
    print(" 시계열 사전학습 모델 다운로드 시작")
    print("=" * 64)
    print()
    print("⚠ 대회 규정: 라이선스 OK (모두 Apache 2.0)")
    print("⚠ 디스크 사용: 약 2~3GB")
    print("⚠ Hugging Face 토큰 필요할 수 있음 (huggingface-cli login)")
    print()

    results = {}
    results["Chronos"] = download_chronos()
    results["TimesFM"] = download_timesfm()
    results["Lag-Llama"] = download_lag_llama()

    print()
    print("=" * 64)
    print(" 결과 요약")
    print("=" * 64)
    for name, ok in results.items():
        status = "✓ 성공" if ok else "✗ 실패"
        print(f"  {name:15s}: {status}")

    cache_dir = Path.home() / ".cache" / "huggingface"
    print(f"\nHuggingFace 캐시 위치: {cache_dir}")
    print(f"  ※ 다른 위치 원하면 HF_HOME 환경변수 설정")

    print("\n다음 단계: python eda_phase1.py")


if __name__ == "__main__":
    main()
