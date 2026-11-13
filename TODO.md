# TODO — MovieLens 25M k=30 실험 분담 (김진욱 · 유은선, 2026-11-13 기준)

LLM/API 호출(intent 추출)은 **진욱 PC**에서만 돌리고, GPU 학습·평가(E 단계)는 **seed를 나눠서 두 사람이** 돌린다.
intent 결과는 Goodreads 때처럼 `data_bundle/`에 분할 압축해서 git으로 전달한다.
구조·실행법은 [`README.md`](README.md), 깨면 안 되는 규칙은 [`AGENTS.md`](AGENTS.md).

## 역할 분담

| | 김진욱 (API PC, RTX 3060 Ti 8GB) | 유은선 |
|---|---|---|
| LLM/API | TO step2 → D → TO 묶음 업로드, TO_GLOBAL step1·step2 → D → TO_GLOBAL 묶음 업로드 | — |
| GPU seed | **2020, 2021** (TO + TO_GLOBAL) | **2022** (TO + TO_GLOBAL) |
| 결과 | 은선 결과 병합 + 3-seed 리포트 | 끝난 split마다 결과 zip 전달 |

- **seed 단위로 나누는 이유:** 한 seed 안에서 비교하는 모델들은 같은 PC에서 돌려야 수치 비교가 공정하다(GPU가 다르면 같은 seed라도 값이 조금 다름).
  그래서 "seed 하나 = 한 사람이 7개 spec 전부"로 나눈다.
- **진욱이 2개인 이유:** API 호출은 GPU를 안 써서 진욱 PC의 GPU는 API가 도는 동안 비어 있고, RTX 3060 Ti가 더 빠를 가능성이 크다
  (Goodreads 때 팀원 PC는 GTX 1060으로 같은 작업이 2배 넘게 걸림). **#1 시간 측정 결과를 보고 다시 조정한다.**
- **같은 PC에서 API와 학습을 동시에 돌려도 된다.** 단, TO_GLOBAL step2를 처음 시작할 때 GPU 인코딩(약 15분)이 있으니
  그때는 진욱 PC의 학습을 seed 경계에서 잠깐 멈춘다(8GB에 둘 다 올리면 메모리 부족 위험).

## 현재 상태

| 단계 | TO (유저별 80/10/10) | TO_GLOBAL (전체 70/10/20) |
|---|---|---|
| 데이터 준비 (ml-25m, k=30, 이전 레포와 규모 일치) | ✅ 133,684 유저 / 15,785 아이템 / 24,046,865 | ✅ |
| step1: exact intent (API) | ✅ 11/12 완료 (유저 133,200 + 아이템 15,784) | ⬜ 유저 약 99k |
| step2: related intent (API) | 🔄 11/13 14:03부터 7시간 실행 중 (시작 시 16,083 / 148,984) | ⬜ 약 115k |
| D: KG pack (`kg_pack.pt`, `meta_kg_pack.pt`) | ⬜ step2 직후 (CPU, 몇 분) | ⬜ |
| `data_bundle/` 업로드 | ⬜ D 직후 | ⬜ |
| 평가 서명 수정 · 묶음/병합 스크립트 | ✅ 11/13 | ✅ |

API 속도는 Luxia 계정 제한 때문에 시간당 약 4,300~5,800회로 고정이다.
남은 API 시간: TO step2 약 24~31시간, TO_GLOBAL step1+step2 약 39~50시간 (진욱 PC가 돌아가는 시간 기준).

## 사람별 체크리스트 (위에서부터 순서대로)

### 유은선 (seed 2022)
| # | 할 일 | 시작 조건 | 상태 |
|---|---|---|---|
| 은1 | §0 준비: clone(`team-mate`) → venv → ml-25m 데이터 준비 → 메타 KG (`[scale] MATCH` 확인) | 지금 | ⬜ |
| 은2 | **#1 GPU 시간 측정** → 아래 표에 기입 후 `team-mate`에 푸시 (또는 진욱에게 전달) | 은1 | ⬜ |
| 은3 | #2 TO, intent 없는 5개 spec, seed 2022 | 은2 + 진욱과 범위 확정 | ⬜ |
| 은4 | #3 TO_GLOBAL, intent 없는 5개 spec, seed 2022 | 은3 | ⬜ |
| 은5 | #4 TO, intent spec 2개, seed 2022 | TO 묶음 업로드 → `git pull` + `python scripts/unpack_data_bundle.py --split TO` | ⬜ |
| 은6 | #5 TO_GLOBAL, intent spec 2개, seed 2022 | TO_GLOBAL 묶음 업로드 → `git pull` + `python scripts/unpack_data_bundle.py --split TO_GLOBAL` | ⬜ |
| 은7 | split 하나가 끝날 때마다 결과 zip → 진욱 (아래 "결과 전달") | 은4·은5·은6 | ⬜ |

순서는 바꿔도 된다: TO 묶음이 은3·은4보다 먼저 올라오면 은5를 먼저 해도 된다 (같은 seed 안이라 상관없음).

### 김진욱 (API + seed 2020, 2021)
| # | 할 일 | 시작 조건 | 상태 |
|---|---|---|---|
| 진1 | TO step2 이어서 실행 (`scripts\intent_run_*h.bat`, GPU 안 씀) | 지금 | 🔄 |
| 진2 | 메타 KG 만들기 (§0의 `build_meta_kg.py` 두 줄; 데이터는 이미 준비됨) | 지금 | ⬜ |
| 진3 | **#1 GPU 시간 측정** (API 도는 중에 같이 돌려도 됨) → 아래 표에 기입 | 진2 | ⬜ |
| 진4 | #2·#3 intent 없는 5개 spec, seed 2020·2021, TO → TO_GLOBAL | 진3 + 은선과 범위 확정 | ⬜ |
| 진5 | TO step2 끝 → D 자동 실행 → `python scripts/pack_data_bundle.py --split TO` → `data_bundle/` 커밋·푸시 → 은선에게 알림 | 진1 완료 | ⬜ |
| 진6 | TO_GLOBAL step1 (`scripts\intent_global_step1_7h.bat`, GPU 숨김) | 진5 | ⬜ |
| 진7 | TO_GLOBAL step2 시작은 **낮에**, 학습 잠깐 멈추고 (`scripts\intent_global_run_7h.bat`, 첫 GPU 인코딩 약 15분) → D | 진6 완료 | ⬜ |
| 진8 | `python scripts/pack_data_bundle.py --split TO_GLOBAL` → 커밋·푸시 → 은선에게 알림 | 진7 완료 | ⬜ |
| 진9 | #4·#5 intent spec 2개, seed 2020·2021 (TO는 진5 이후, TO_GLOBAL은 진8 이후) | 진5 / 진8 | ⬜ |
| 진10 | 은선 결과 병합 + 리포트 (아래 "결과 모으기") | 은7 | ⬜ |

## 0. 준비 (처음 한 번, 은선)

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

**데이터 — 묶음이 올라온 뒤 (#4~#5용):** `git pull` 후 `python scripts/unpack_data_bundle.py --split <SPLIT>` (sha256 확인 후 복원).
§0에서 직접 만든 메타 KG·provenance를 묶음 파일로 덮어써도 내용이 같으면 끝난 결과는 그대로 유지된다.

**매 실행 전에** (새 터미널마다): `$env:PYTHONIOENCODING = "utf-8"` (bash: `export PYTHONIOENCODING=utf-8`)

### 공통 규칙
- 중단: `Ctrl+C` 한 번. 재개: **같은 커맨드 다시 입력** → 끝난 모델×seed는 건너뛰고, 돌던 것만 epoch 0부터 다시.
- 손해 없이 끊는 시점: 콘솔에 새 `===== <모델> seed=<n> (<SPLIT>) =====` 헤더가 뜬 직후.
- 실행 중에는 코드와 `config.yaml`을 수정하지 않는다 (코드/설정 해시가 바뀌면 끝난 결과까지 무효화).
- 두 사람 모두 **같은 브랜치 최신 상태**(`git pull`)로 돌린다. 코드가 다르면 결과를 합칠 수 없다 (병합 스크립트가 거부함).
- 자기 seed만 돌린다 (`--seeds 2022` 또는 `--seeds 2020,2021`). 한 seed의 7개 spec은 한 PC에서.
- split 규칙·epoch·학습률 등 하이퍼파라미터는 바꾸지 않는다 (모든 spec 공통 프로토콜, AGENTS.md 불변 조건 5).
- ⚠️ MovieLens는 interaction이 약 2,400만 건으로 Goodreads보다 훨씬 많다. LightGCN 계열은 배치마다 그래프 전체를 전파해서
  에폭 시간이 데이터 크기보다 더 빠르게 늘 수 있다 → **#1 측정 결과를 보고 둘이 seed 수·spec·epoch을 확정한다.**

## 1. 작업 정의

| # | 작업 | 필요한 것 |
|---|---|---|
| 1 | **GPU 시간 측정**: TO, seed 2020, 2 epoch — BPR, LightGCN, IKGR_dyn_nointent, IKGR_lgcn_dyn_nointent (각자 자기 GPU에서) | §0 데이터 |
| 2 | intent 없는 5개 spec — TO: BPR, LightGCN, IKGR_kgoff, IKGR_dyn_nointent, IKGR_lgcn_dyn_nointent | §0 데이터 |
| 3 | 같은 5개 spec — TO_GLOBAL | §0 데이터 |
| 4 | intent spec 2개 — TO: IKGR_dyn, IKGR_lgcn_dyn | TO 묶음 |
| 5 | 같은 2개 spec — TO_GLOBAL | TO_GLOBAL 묶음 |

## 2. 커맨드

### #1 GPU 시간 측정 (둘 다, 지금 바로)
```powershell
python run_pipeline.py --k 30 --split TO --steps E --seeds 2020 --epochs 2 --specs BPR
python run_pipeline.py --k 30 --split TO --steps E --seeds 2020 --epochs 2 --specs LightGCN
python run_pipeline.py --k 30 --split TO --steps E --seeds 2020 --epochs 2 --specs IKGR_dyn_nointent
python run_pipeline.py --k 30 --split TO --steps E --seeds 2020 --epochs 2 --specs IKGR_lgcn_dyn_nointent
```
2 epoch인 이유: validation이 2 epoch마다 돌아서(eval_step 2), 1 epoch만 돌리면 저장된 모델이 없어 평가 단계가 실패할 수 있다.
각 spec이 끝나면 `run_k30_TO/slice_eval_TO_result.json`의 `train_sec`(2 epoch + validation 1회 합계, 또는 콘솔의 epoch 시간)과
`nvidia-smi`로 본 최대 VRAM을 아래 표에 적는다. 측정용이라 seed는 둘 다 2020으로 같게 한다.
이 짧은 실행의 결과는 나중에 정식 실행(40 epoch)에서 자동으로 무효화되니 지우지 않아도 된다.
첫 epoch이 너무 오래 걸리면(예: 2시간 이상) `Ctrl+C`로 끊고 걸린 시간과 진행률만 적는다.

### #2·#3 intent 없는 spec
```powershell
# 은선
python run_pipeline.py --k 30 --split TO        --steps E --seeds 2022 --specs BPR,LightGCN,IKGR_kgoff,IKGR_dyn_nointent,IKGR_lgcn_dyn_nointent
python run_pipeline.py --k 30 --split TO_GLOBAL --steps E --seeds 2022 --specs BPR,LightGCN,IKGR_kgoff,IKGR_dyn_nointent,IKGR_lgcn_dyn_nointent
# 진욱
python run_pipeline.py --k 30 --split TO        --steps E --seeds 2020,2021 --specs BPR,LightGCN,IKGR_kgoff,IKGR_dyn_nointent,IKGR_lgcn_dyn_nointent
python run_pipeline.py --k 30 --split TO_GLOBAL --steps E --seeds 2020,2021 --specs BPR,LightGCN,IKGR_kgoff,IKGR_dyn_nointent,IKGR_lgcn_dyn_nointent
```
spec 단위로 나눠 돌려도 된다 (`--specs LightGCN`만 따로 등).

### #4·#5 intent spec (묶음 복원 후)
```powershell
# 은선 (진욱은 --seeds 2020,2021)
python run_pipeline.py --k 30 --split TO        --steps E --seeds 2022 --specs IKGR_dyn,IKGR_lgcn_dyn
python run_pipeline.py --k 30 --split TO_GLOBAL --steps E --seeds 2022 --specs IKGR_dyn,IKGR_lgcn_dyn
```

### 결과 전달 (은선 → 진욱)
split 하나가 끝나면 `run_k30_<SPLIT>/slice_eval_<SPLIT>_result.json`과 `run_k30_<SPLIT>/per_user_<SPLIT>/` 폴더를
zip 하나로 묶어(예: `eunseon_TO.zip`) 진욱에게 보낸다. 중간 결과를 보내도 되고, 다시 보내면 새 seed·spec만 추가된다.

### 결과 모으기 (진욱)
```powershell
python scripts/merge_results.py --split TO --inputs eunseon_TO.zip          # 진욱의 run_k30_TO/에 합쳐짐 (기존 파일은 .bak)
python paired_compare.py --run_dir run_k30_TO --split TO --report           # → run_k30_TO/report_TO.md
```
코드·데이터·프로토콜이 다른 결과는 합쳐지지 않고 이유(`code`, `data_fingerprint`, `protocol` …)가 출력된다 → 같은 브랜치로 다시 돌린다.

## #1 측정 결과 (기입)

| 이름 | GPU (VRAM) | spec | 1 epoch 시간 (train_sec ÷ 2) | 최대 VRAM | 비고 |
|---|---|---|---|---|---|
| 김진욱 | RTX 3060 Ti (8GB) | BPR | | | |
| 김진욱 | RTX 3060 Ti (8GB) | LightGCN | | | |
| 김진욱 | RTX 3060 Ti (8GB) | IKGR_dyn_nointent | | | |
| 김진욱 | RTX 3060 Ti (8GB) | IKGR_lgcn_dyn_nointent | | | |
| 유은선 | | BPR | | | |
| 유은선 | | LightGCN | | | |
| 유은선 | | IKGR_dyn_nointent | | | |
| 유은선 | | IKGR_lgcn_dyn_nointent | | | |

참고 (Goodreads, RTX 3060 Ti 8GB, 64차원): IKGR_lgcn 계열 약 4.2시간/seed, LightGCN 약 5.3시간/seed.
측정 후 둘이 정할 것: 40 epoch 기준 총 시간이 감당 가능한지, 안 되면 seed 수·TO_GLOBAL 범위·epoch 중 무엇을 줄일지, 분담 비율.

## 참고: 데이터 묶음 (Goodreads와 같은 방식)

- `data_bundle/movielens_k30_<SPLIT>_bundle.tgz.partNN` (95MB씩 분할) + `.sha256` + `.files.json`, 복원은 `scripts/unpack_data_bundle.py`.
- TO 묶음: `data/kc_k30/ikgr-custom/ikgr-custom.inter`, 규모 통계, TO 프로필·provenance, `run_k30_TO/kg_pack.pt`, `meta_kg_pack.pt`.
- TO_GLOBAL 묶음: TO_GLOBAL 프로필·provenance, `run_k30_TO_GLOBAL/kg_pack.pt`, `meta_kg_pack.pt` (`.inter`는 TO 묶음과 공통).
- `interactions_k30.csv`(700MB)는 `.inter`와 같은 내용이고 학습에 안 쓰여서 넣지 않는다.
- 묶음이 올라오면 §0에서 직접 만든 데이터 대신 묶음을 쓴다 (sha256으로 두 사람이 같은 파일인지 보장).
