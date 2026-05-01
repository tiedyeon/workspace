# 환경 셋업 가이드라인 — 스마트 창고 출고 지연 예측

> 새 머신에서 이 프로젝트 환경을 처음부터 구성할 때 따라가는 단계별 절차.
> Windows + PowerShell 기준. 실패 시 각 단계의 "검증" 항목으로 원인 좁히기.
>
> 이 문서를 처음 본다면 먼저 `..\CLAUDE.md` 를 읽어 프로젝트 맥락(대회·검증 전략·진행 이력)을 파악할 것.

---

## 0. 사전 조건 확인

```powershell
# Python 3.11 설치 확인 (3.11.x 가 표시되어야 함)
python --version

# 없다면 https://www.python.org/downloads/ 에서 3.11 설치
# 설치 시 "Add python.exe to PATH" 체크 필수
```

**검증**: `Python 3.11.x` 출력. 3.10 이하 / 3.13 이상이면 일부 패키지 빌드 실패 가능 → 3.11로 맞출 것.

---

## 1. 프로젝트 폴더 진입

```powershell
cd C:\Users\winmo\Desktop\workspace\스마트물류
```

**현재 폴더의 예상 상태**:
```
스마트물류\
├─ dataset.zip            ← 4개 csv 압축본
├─ layout_info.csv        ← (zip 외부에도 존재)
├─ sample_submission.csv  ← (zip 외부에도 존재)
├─ requirements.txt       ← 이번 단계에서 추가됨
└─ SETUP.md               ← 이 문서
```

`..\CLAUDE.md` (workspace 루트)는 핸드오프 문서, 그대로 둔다.

---

## 2. 데이터 배치 (`data/` 폴더 구성)

CLAUDE.md 의 "2. 데이터셋" 섹션과 모든 스크립트는 **`data/` 폴더 안의 4개 csv** 를 가정한다.

```powershell
# 2-1. data 폴더 생성
New-Item -ItemType Directory -Path .\data -Force | Out-Null

# 2-2. zip 해제 (4개 csv 가 .\data\ 로 풀림)
Expand-Archive -Path .\dataset.zip -DestinationPath .\data -Force

# 2-3. zip 외부 중복본 정리 (선택, data\ 안에 동일 파일 존재)
#   → 안 지우고 그대로 둬도 학습에는 영향 없음
# Remove-Item .\layout_info.csv, .\sample_submission.csv

# 2-4. (선택) 디스크 절약하려면 dataset.zip 도 제거 가능
#   → 백업 의미로 남겨두는 것을 권장
# Remove-Item .\dataset.zip
```

**검증**:
```powershell
Get-ChildItem .\data | Format-Table Name, Length
```
다음 4개 파일이 모두 보여야 함:
| 파일 | 대략 크기 |
| --- | --- |
| `train.csv` | 약 121 MB |
| `test.csv` | 약 23 MB |
| `layout_info.csv` | 약 22 KB |
| `sample_submission.csv` | 약 800 KB |

행수 점검:
```powershell
# 헤더 포함 행수 — train: 250001, test: 50001
(Get-Content .\data\train.csv -ReadCount 0).Count
(Get-Content .\data\test.csv  -ReadCount 0).Count
```

---

## 3. 가상환경 생성 및 활성화

프로젝트 의존성을 시스템 Python에 섞이지 않게 분리한다.

> **이 프로젝트는 가상환경 폴더명을 `.smart` 로 통일한다** (관례적 `.venv` 아님). 모든 스크립트·docs·gitignore 가 `.smart` 기준이므로 다른 이름으로 만들지 말 것.

```powershell
# 3-1. .smart 생성 (한 번만, 폴더명 주의: .venv 아닌 .smart)
python -m venv .smart

# 3-2. 활성화 (새 PowerShell 창마다 매번 필요)
.\.smart\Scripts\Activate.ps1
```

**ExecutionPolicy 에러가 뜨면** (`...스크립트 로드를 사용할 수 없으므로...`):
```powershell
# 현재 세션에 한해 일회성 허용 (시스템 정책 안 바꿈)
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
# 그 다음 활성화 재시도
.\.smart\Scripts\Activate.ps1
```

**검증**: 프롬프트 맨 앞에 `(.smart)` 가 붙어 있어야 한다. 예) `(.smart) PS C:\...\스마트물류>`

---

## 4. 패키지 설치

```powershell
# pip 자체 업그레이드 (선택, 권장)
python -m pip install --upgrade pip

# requirements.txt 일괄 설치 (3~5분 소요)
pip install -r requirements.txt
```

**LightGBM / XGBoost / CatBoost 설치 실패 시 대처**:
- LightGBM: Visual C++ 재배포 패키지 필요할 수 있음 → `pip install lightgbm --no-cache-dir` 재시도
- CatBoost: 패키지 크기 큼(약 100MB), 다운로드 자체가 느릴 수 있음 → 기다릴 것
- 모두 실패한다면 conda 사용 옵션 존재 (`conda install -c conda-forge lightgbm xgboost catboost`)

**검증 (스모크 테스트)**:
```powershell
python -c "import numpy, pandas, sklearn, lightgbm, xgboost, catboost; print('versions:', numpy.__version__, pandas.__version__, sklearn.__version__, lightgbm.__version__, xgboost.__version__, catboost.__version__); print('OK')"
```
마지막 줄에 `OK` 가 출력되면 정상.

---

## 5. 데이터 로딩 점검

CLAUDE.md 의 데이터 구조와 일치하는지 확인 (250,000행 × 94열, 50,000행 × 93열).

```powershell
python -c "import pandas as pd; tr=pd.read_csv('data/train.csv'); te=pd.read_csv('data/test.csv'); print('train:', tr.shape, '| test:', te.shape); print('target stats:', tr['avg_delay_minutes_next_30m'].describe()[['mean','50%','max']].to_dict())"
```

**기대 출력**:
```
train: (250000, 94) | test: (50000, 93)
target stats: {'mean': 18.96..., '50%': 9.03..., 'max': 715.86...}
```

수치가 다르면 데이터 파일이 손상되었거나 다른 버전임 → 데이콘에서 다시 다운받을 것.

---

## 6. (선택) Jupyter 커널 등록

EDA를 노트북으로 돌릴 계획이면 이 .smart 를 Jupyter 에서 고를 수 있게 등록한다.

```powershell
python -m ipykernel install --user --name smartlogi-venv --display-name "Python (smartlogi)"
```

VSCode·JupyterLab 에서 커널 선택 시 `Python (smartlogi)` 가 보이면 OK.

---

## 7. 산출물 폴더 준비

스크립트 실행 시 자동 생성되지만, 미리 만들어 두면 .gitignore 설정 시 편함.

```powershell
New-Item -ItemType Directory -Path .\outputs, .\eda_outputs -Force | Out-Null
```

---

## 8. .gitignore 권장 항목

이 프로젝트를 git 으로 관리한다면:

```gitignore
# 가상환경
.smart/

# 데이터 (대용량 + 라이선스)
data/
dataset.zip

# 산출물
outputs/
eda_outputs/

# Python 부산물
__pycache__/
*.pyc
.ipynb_checkpoints/

# IDE
.vscode/
.idea/
```

---

## 9. 셋업 완료 체크리스트

순서대로 다 ✅ 되면 본격 작업 시작 가능.

```
□ python --version → 3.11.x
□ data\ 폴더에 train.csv / test.csv / layout_info.csv / sample_submission.csv 4개
□ train.csv 행수 250001 (헤더 포함), test.csv 50001
□ .smart 생성 + 활성화 (프롬프트에 (.smart))
□ pip install -r requirements.txt 무에러 완료
□ smoke test → "OK" 출력
□ pandas 로 train shape (250000, 94) 확인
```

---

## 10. 다음 단계 — 스크립트 복원

환경이 갖춰지면 다음 작업은 **이전 머신에 있던 스크립트 복원**이다. CLAUDE.md 의 "8. 사용 중인 코드 파일 요약" 표 기준:

| 파일 | 우선순위 | 비고 |
| --- | --- | --- |
| `baseline_lgbm_quantile.py` | 🔴 필수 | submission1 재현 (OOF MAE 9.1768 일치 확인 = 환경 검증 종료) |
| `eda.py` | 🟡 선택 | EDA 그림 재생성. 결과 안 변함 |
| `submission2_lgbm_log1p.py` | ⚪ 보존만 | 폐기된 라인, 참고용 |

`baseline_lgbm_quantile.py` 가 존재하지 않으므로 **새로 작성해야 한다**. 작성 시 합의된 사항(GroupKFold by layout_id, 정리된 111개 피처 셋, NaN 그대로 입력, `outputs/submission1_*` 접두어)을 반드시 반영할 것. 작성·실행이 끝나고 OOF MAE 9.1768 재현이 확인되면 그 다음에 submission3 (Huber 또는 CatBoost MAE) 로 진행.

---

## 부록 A — 흔한 트러블슈팅

| 증상 | 원인 | 해결 |
| --- | --- | --- |
| `Activate.ps1: 스크립트 로드 불가` | PowerShell ExecutionPolicy | `Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned` |
| `pip` 가 시스템 Python 으로 설치 | .smart 미활성화 | 프롬프트에 `(.smart)` 확인 후 재시도 |
| `lightgbm.basic.LightGBMError: Do not support special JSON characters` | 컬럼명에 특수문자 | 학습 직전 `df.columns = df.columns.str.replace(...)` 로 정리 |
| 메모리 부족 (train.csv 로딩 시) | 8GB RAM 한계 | `pd.read_csv(..., dtype={...})` 로 dtype 다운캐스팅, 또는 청크 로딩 |
| Jupyter 에서 패키지 import 실패 | .smart 커널 미등록 | 6장 ipykernel 등록 |

## 부록 B — 다른 OS 에서 셋업

이 가이드는 Windows + PowerShell 전용이지만, macOS/Linux 에서도 같은 흐름.
- `python -m venv .smart` 동일
- 활성화: `source .smart/bin/activate`
- 압축 해제: `unzip dataset.zip -d data/`
- 나머지(pip, 검증) 동일
