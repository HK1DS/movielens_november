# TODO — MovieLens 25M k=30 팀원 작업 (2026-11-13 기준)

팀원 작업은 **GPU 학습·평가(E 단계)** 만이다. LLM/API 호출(intent 추출)은 진욱 PC에서 돌리고,
결과는 Goodreads 때처럼 `data_bundle/`에 분할 압축해서 git으로 전달한다.
구조·실행법은 [`README.md`](README.md), 깨면 안 되는 규칙은 [`AGENTS.md`](AGENTS.md).

## 현재 상태

| 단계 | TO (유저별 80/10/10) | TO_GLOBAL (전체 70/10/20) |
|---|---|---|
| 데이터 준비 (ml-25m, k=30, 이전 레포와 규모 일치) | ✅ 133,684 유저 / 15,785 아이템 / 24,046,865 | ✅ |
| step1: exact intent (API) | ✅ 11/12 완료 (유저 133,200 + 아이템 15,784) | ⬜ 유저 약 99k |
| step2: related intent (API) | 🔄 16,083 / 148,984 (11%) | ⬜ 약 115k |
| D: KG pack (`kg_pack.pt`, `meta_kg_pack.pt`) | ⬜ step2 직후 (CPU, 몇 분) | ⬜ |
| `data_bundle/` 업로드 | ⬜ D 직후 | ⬜ |

API 속도는 Luxia 계정 제한 때문에 시간당 약 4,300~5,800회로 고정이다.
남은 API 시간: TO step2 약 24~31시간, TO_GLOBAL step1+step2 약 39~50시간 (진욱 PC가 돌아가는 시간 기준).

## 0. 준비 (처음 한 번)

```powershell
git clone -b team-mate https://github.com/HK1DS/movielens_november
cd movielens_november
powershell -ExecutionPolicy Bypass -File scripts\setup_venv.ps1        # Windows + NVIDIA GPU (venv: %USERPROFILE%\.venvs\goodreads-ikgr, Goodreads venv 재사용)
& "$env:USERPROFILE\.venvs\goodreads-ikgr\Scripts\Activate.ps1"
python -m pytest -q tests/                                              # (선택) 전부 passed 확인
```
Linux: `pip install torch==2.3.0 --index-url https://download.pytorch.org/whl/cu121 && pip install -r requirements-pipeline.txt`.

**데이터 — 묶음이 올라오기 전 (#1~#3용):** ml-25m을 직접 받아서 무료 준비 단계를 돌린다 (API 호출 없음, 15~30분, 디스크 약 4GB).
```powershell
# https://files.grouplens.org/datasets/movielens/ml-25m.zip 을 받아서 raw\ml-25m\ 에 압축 해제 (ratings.csv, movies.csv)
python run_pipeline.py --k 30 --split TO --steps 0ATP          # 콘솔에 [scale] MATCH 가 떠야 함 (위 표의 숫자와 동일)
python run_pipeline.py --k 30 --split TO_GLOBAL --steps P
python build_meta_kg.py --inter data/kc_k30/ikgr-custom/ikgr-custom.inter --out run_k30_TO/meta_kg_pack.pt
python build_meta_kg.py --inter data/kc_k30/ikgr-custom/ikgr-custom.inter --out run_k30_TO_GLOBAL/meta_kg_pack.pt
```
`[scale] MISMATCH`가 뜨면 멈추고 진욱에게 알린다 (데이터가 다르면 결과를 합칠 수 없음).

**데이터 — 묶음이 올라온 뒤 (#4~#5용):** `git pull` 후 `python scripts/unpack_data_bundle.py` (sha256 확인 후 복원, 스크립트 준비 중).

**매 실행 전에** (새 터미널마다): `$env:PYTHONIOENCODING = "utf-8"` (bash: `export PYTHONIOENCODING=utf-8`)

### 공통 규칙
- 중단: `Ctrl+C` 한 번. 재개: **같은 커맨드 다시 입력** → 끝난 모델×seed는 건너뛰고, 돌던 것만 epoch 0부터 다시.
- 손해 없이 끊는 시점: 콘솔에 새 `===== <모델> seed=<n> (<SPLIT>) =====` 헤더가 뜬 직후.
- 실행 중에는 코드와 `config.yaml`을 수정하지 않는다 (코드/설정 해시가 바뀌면 끝난 결과까지 무효화).
- 한 seed 안에서 비교하는 모델들은 **같은 PC**에서 돌린다 (GPU가 다르면 같은 seed라도 수치가 조금 다를 수 있음).
- split 규칙·epoch·학습률 등 하이퍼파라미터는 바꾸지 않는다 (모든 spec 공통 프로토콜, AGENTS.md 불변 조건 5).
- 끝나면 `run_k30_<SPLIT>/slice_eval_<SPLIT>_result.json`과 `run_k30_<SPLIT>/per_user_<SPLIT>/` 폴더를 압축해서 공유 → 진욱이 합쳐서 3-seed 리포트를 만든다.
- ⚠️ MovieLens는 interaction이 약 2,400만 건으로 Goodreads보다 훨씬 많다. LightGCN 계열은 배치마다 그래프 전체를 전파해서
  에폭 시간이 데이터 크기보다 더 빠르게 늘 수 있다 → **#1 측정 결과를 보고 seed 수·spec·epoch을 확정한다.**

## 1. 작업 목록

| # | 작업 | 필요한 것 | 담당 | 상태 | 예상 시간 |
|---|---|---|---|---|---|
| A | TO step2 → D → TO 묶음 업로드 | — | 진욱 PC | 🔄 step2 11% | API 약 24~31h |
| B | TO_GLOBAL step1·step2 → D → TO_GLOBAL 묶음 업로드 | — | 진욱 PC | ⬜ | API 약 39~50h |
| C | 평가 fingerprint 수정: spec이 안 쓰는 KG 파일은 해시에서 제외 (지금은 `kg_pack.pt`가 생기면 intent 없는 spec 결과까지 무효화됨) | — | 진욱 | ⬜ **#2·#3 시작 전 필수** | — |
| D | 묶음 pack/unpack 스크립트, 팀원별 결과 병합 스크립트 | — | 진욱 | ⬜ | — |
| 1 | **GPU 시간 측정 (최우선)**: TO, seed 2020, 2 epoch — BPR, LightGCN, IKGR_dyn_nointent, IKGR_lgcn_dyn_nointent | §0 데이터 | 팀원 전원 (각자 GPU) | ⬜ 지금 가능 | spec당 2 epoch |
| 2 | intent 없는 5개 spec × 3 seed — TO: BPR, LightGCN, IKGR_kgoff, IKGR_dyn_nointent, IKGR_lgcn_dyn_nointent | §0 데이터 + C | 팀원 (seed별 분담) | ⬜ C 이후 | #1 결과로 산정 |
| 3 | 같은 5개 spec × 3 seed — TO_GLOBAL | §0 데이터 + C | 팀원 (seed별 분담) | ⬜ C 이후 | #1 결과로 산정 |
| 4 | intent spec 2개 × 3 seed — TO: IKGR_dyn, IKGR_lgcn_dyn | TO 묶음 (A) | 팀원 (seed별 분담) | ⬜ A 이후 | #1 결과로 산정 |
| 5 | 같은 2개 spec × 3 seed — TO_GLOBAL | TO_GLOBAL 묶음 (B) | 팀원 (seed별 분담) | ⬜ B 이후 | #1 결과로 산정 |
| 6 | 결과 병합 + paired_compare 리포트 | 2~5 결과 | 진욱 | ⬜ | — |

### seed 분담 (안 — 이름 채워 넣기)
| seed | 담당 | 할 일 |
|---|---|---|
| 2020 | 팀원 A | #2~#5의 seed 2020 전부 (같은 seed는 한 PC에서) |
| 2021 | 팀원 B | #2~#5의 seed 2021 전부 |
| 2022 | 팀원 C | #2~#5의 seed 2022 전부 |

## 2. 커맨드

### #1 GPU 시간 측정 (지금 바로)
```powershell
python run_pipeline.py --k 30 --split TO --steps E --seeds 2020 --epochs 2 --specs BPR
python run_pipeline.py --k 30 --split TO --steps E --seeds 2020 --epochs 2 --specs LightGCN
python run_pipeline.py --k 30 --split TO --steps E --seeds 2020 --epochs 2 --specs IKGR_dyn_nointent
python run_pipeline.py --k 30 --split TO --steps E --seeds 2020 --epochs 2 --specs IKGR_lgcn_dyn_nointent
```
2 epoch인 이유: validation이 2 epoch마다 돌아서(eval_step 2), 1 epoch만 돌리면 저장된 모델이 없어 평가 단계가 실패할 수 있다.
각 spec이 끝나면 `run_k30_TO/slice_eval_TO_result.json`의 `train_sec`(2 epoch + validation 1회 합계, 또는 콘솔의 epoch 시간)과
`nvidia-smi`로 본 최대 VRAM을 아래 표에 적어서 `team-mate` 브랜치에 푸시한다.
이 짧은 실행의 결과는 나중에 정식 실행(40 epoch)에서 자동으로 무효화되니 지우지 않아도 된다.
첫 epoch이 너무 오래 걸리면(예: 2시간 이상) `Ctrl+C`로 끊고 걸린 시간과 진행률만 적는다.

### #2·#3 intent 없는 spec (C 완료 후, 자기 seed로)
```powershell
python run_pipeline.py --k 30 --split TO        --steps E --seeds 2020 --specs BPR,LightGCN,IKGR_kgoff,IKGR_dyn_nointent,IKGR_lgcn_dyn_nointent
python run_pipeline.py --k 30 --split TO_GLOBAL --steps E --seeds 2020 --specs BPR,LightGCN,IKGR_kgoff,IKGR_dyn_nointent,IKGR_lgcn_dyn_nointent
```
spec 단위로 나눠 돌려도 된다 (`--specs LightGCN`만 따로 등).

### #4·#5 intent spec (묶음 복원 후, 자기 seed로)
```powershell
python run_pipeline.py --k 30 --split TO        --steps E --seeds 2020 --specs IKGR_dyn,IKGR_lgcn_dyn
python run_pipeline.py --k 30 --split TO_GLOBAL --steps E --seeds 2020 --specs IKGR_dyn,IKGR_lgcn_dyn
```

### 결과 확인 (자기 결과만 볼 때)
```powershell
python paired_compare.py --run_dir run_k30_TO --split TO --report     # → run_k30_TO/report_TO.md
```

## #1 측정 결과 (팀원 기입)

| 이름 | GPU (VRAM) | spec | 1 epoch 시간 (train_sec ÷ 2) | 최대 VRAM | 비고 |
|---|---|---|---|---|---|
| | | BPR | | | |
| | | LightGCN | | | |
| | | IKGR_dyn_nointent | | | |
| | | IKGR_lgcn_dyn_nointent | | | |

참고 (Goodreads, RTX 3060 Ti 8GB, 64차원): IKGR_lgcn 계열 약 4.2시간/seed, LightGCN 약 5.3시간/seed.

## 참고: 데이터 묶음 (예정, Goodreads와 같은 방식)

- `data_bundle/<이름>.tgz.partNN` (95MB씩 분할) + `bundle.sha256` + `scripts/unpack_data_bundle.py`.
- TO 묶음: `data/kc_k30/ikgr-custom/ikgr-custom.inter`, TO 프로필·provenance, `run_k30_TO/kg_pack.pt`, `meta_kg_pack.pt`.
- TO_GLOBAL 묶음: TO_GLOBAL 프로필·provenance, `run_k30_TO_GLOBAL/kg_pack.pt`, `meta_kg_pack.pt` (`.inter`는 TO 묶음과 공통).
- `interactions_k30.csv`(700MB)는 `.inter`와 같은 내용이고 학습에 안 쓰여서 제외 예정.
- 묶음이 올라오면 §0에서 직접 만든 데이터 대신 묶음을 쓴다 (sha256으로 모두 같은 파일인지 보장).
