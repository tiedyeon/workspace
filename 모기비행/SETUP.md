# 환경 셋업 가이드 — 모기 비행 궤적 예측

> Windows + **VS Code 내장 cmd** + GPU 권장 기준.
> 이 대회는 시계열 모델 (LSTM/Transformer) 학습이 핵심이라 **GPU 사용 강력 권장**.
>
> ⚠ 본 문서의 명령어는 **cmd 기반**. PowerShell 사용 시 차이점은 부록 C 참고.

---

## 0. 사전 조건

```powershell
# Python 3.11 확인
python --version

# GPU 확인 (NVIDIA)
nvidia-smi
```

**검증**:

- `Python 3.11.x` 표시되어야 함
- `nvidia-smi` 가 GPU 정보 출력 → CUDA 사용 가능
- 출력 안 되면 → CPU 모드로 진행 (학습 느림, 가능은 함)

`nvidia-smi` 결과의 CUDA Version 기록해두기 (예: 12.1). PyTorch 설치 시 매칭 필요.

---

## 1. 프로젝트 폴더 진입

```powershell
cd C:\Users\winmo\Desktop\workspace\모기궤적
```

**현재 폴더 상태** (예상):

```
모기궤적\
├─ data\
│  ├─ train\           ← TRAIN_*.csv 10000개
│  ├─ test\            ← TEST_*.csv 10000개
│  ├─ train_labels.csv
│  └─ sample_submission.csv
└─ (이 문서)
```

---

## 2. 가상환경 생성 + 활성화

```cmd
:: 2-1. .smart 생성 (한 번만)
python -m venv .smart

:: 2-2. 활성화 (새 터미널마다 매번 필요) — cmd 는 .bat 사용
.smart\Scripts\activate.bat
```

**검증**: 프롬프트가 `(.smart) C:\...\모기궤적>` 로 변함.

비활성화하려면:

```cmd
deactivate
```

---

## 3. PyTorch GPU 설치 (CUDA 12.1 예시)

⚠ PyTorch 는 CUDA 버전별로 패키지가 다름. `nvidia-smi` 결과의 CUDA Version 에 맞춰 설치.

```cmd
:: CUDA 12.1 의 경우
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

:: CUDA 11.8 의 경우 (위 명령 대신)
:: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

:: GPU 없는 경우 (CPU only — 학습 매우 느림)
:: pip install torch torchvision torchaudio
```

공식 가이드: https://pytorch.org/get-started/locally/

**검증**:

```cmd
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

`CUDA: True` 가 나와야 GPU 모드.

---

## 4. 나머지 패키지 설치

```cmd
pip install -r requirements.txt
```

**스모크 테스트**:

```cmd
python -c "import numpy, pandas, sklearn, lightgbm, xgboost, scipy, matplotlib, seaborn, tqdm; print('OK')"
```

`OK` 출력 → 정상.

---

## 5. (선택) 시계열 사전학습 모델 다운로드

이 대회에서 사용해볼 수 있는 오픈소스 시계열 foundation 모델들. 한 번 다운로드 받으면 재학습 없이 zero-shot 또는 fine-tune 가능.

```cmd
:: Hugging Face 캐시 위치 (선택 — 기본은 ~/.cache/huggingface)
:: set HF_HOME=C:\Users\winmo\Desktop\workspace\모기궤적\hf_cache

:: 모델 다운로드 스크립트 실행
python download_models.py
```

다운로드되는 모델 (라이선스 OK):

- **Chronos** (Amazon, Apache 2.0) — univariate 시계열 forecasting
- **TimesFM** (Google, Apache 2.0) — Google's time series foundation model
- **Lag-Llama** (ServiceNow, Apache 2.0) — small probabilistic forecaster

⚠ 이 대회는 sensor-local 3D 좌표라 위 모델들은 직접 적용보다는 **잔차 학습** 또는 **앙상블 다양성** 용으로 시도할 가능성이 큼. 데이터가 작아 (10k 샘플) custom LSTM/Transformer 가 더 효과적일 수 있음.

---

## 6. 데이터 검증

```cmd
python -c "import pandas as pd; from pathlib import Path; train_files = list(Path('data/train').glob('TRAIN_*.csv')); test_files = list(Path('data/test').glob('TEST_*.csv')); labels = pd.read_csv('data/train_labels.csv'); print(f'train: {len(train_files)} files'); print(f'test:  {len(test_files)} files'); print(f'labels: {labels.shape}'); print(labels.head())"
```

**기대 출력**:

```
train: 10000 files
test:  10000 files
labels: (10000, 4)
   id     x     y     z
0  TRAIN_00001  3.099  0.504  -0.157
...
```

---

## 7. 산출물 폴더 준비

```cmd
mkdir outputs eda_outputs cache
```

- `outputs/`: 모델 산출물 (submission\_\*.csv, OOF 예측 등)
- `eda_outputs/`: EDA figure / CSV
- `cache/`: 로딩 캐시 (parquet 등)

---

## 8. .gitignore 권장

```gitignore
.smart/
data/
hf_cache/
outputs/
eda_outputs/
cache/
__pycache__/
*.pyc
.ipynb_checkpoints/
.vscode/
```

---

## 9. 셋업 완료 체크리스트

```
□ python --version → 3.11.x
□ nvidia-smi → GPU 인식 (GPU 모드 사용 시)
□ .smart 활성화 → 프롬프트에 (.smart)
□ torch.cuda.is_available() → True (GPU) 또는 CPU 확인
□ pip install -r requirements.txt 무에러 완료
□ data/ 폴더에 train/ test/ train_labels.csv sample_submission.csv
□ train 10000, test 10000 파일 확인
□ outputs/, eda_outputs/, cache/ 폴더 생성
□ (선택) python download_models.py — 사전학습 모델 다운로드
```

---

## 10. 다음 단계 — EDA Phase 1 부터

```cmd
python eda_phase1.py
```

EDA 출력은 `eda_outputs/phase1_*.png` 와 CSV 로 저장됨. 결과 보고 Phase 2 (운동 패턴 분석) 진행.

---

## 부록 A — 트러블슈팅

| 증상                                | 원인                     | 해결                                                |
| ----------------------------------- | ------------------------ | --------------------------------------------------- |
| `activate.bat 인식 안 됨`           | 경로 오타                | `.smart\Scripts\activate.bat` (`.\.smart\...` 아님) |
| `torch.cuda.is_available() = False` | CUDA 미감지              | `nvidia-smi` 확인, PyTorch CUDA 버전 매칭 재설치    |
| GPU OOM (Out of Memory)             | batch_size 너무 큼       | batch_size 줄임 (예: 256 → 128)                     |
| 학습 매우 느림                      | CPU 모드 또는 GPU 미활용 | `model.to('cuda')` 확인                             |
| HuggingFace 다운로드 실패           | 네트워크/방화벽          | VPN 또는 mirror 사용                                |

---

## 부록 B — 한 번에 셋업 (cmd 한 줄 한 줄 복붙)

```cmd
cd C:\Users\winmo\Desktop\workspace\모기궤적
python -m venv .smart
.smart\Scripts\activate.bat
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
mkdir outputs eda_outputs cache
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
python eda_phase1.py
```

---

## 부록 C — PowerShell 사용자라면

| 항목            | cmd (위 사용)                 | PowerShell                                                                   |
| --------------- | ----------------------------- | ---------------------------------------------------------------------------- |
| 활성화          | `.smart\Scripts\activate.bat` | `.\.smart\Scripts\Activate.ps1`                                              |
| 주석            | `:: 주석` 또는 `REM`          | `# 주석`                                                                     |
| 환경변수        | `set VAR=val`                 | `$env:VAR = "val"`                                                           |
| 폴더 생성       | `mkdir a b c`                 | `New-Item -ItemType Directory ...`                                           |
| ExecutionPolicy | 불필요                        | `Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned` 필요 가능 |
