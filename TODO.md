# TODO — MovieLens 25M k=30 실험 분담 (김진욱 · 유은선, 2026-11-13 밤 기준, 목표: 11/20 리포트)

LLM/API 호출(intent 추출)은 **진욱 PC**에서 돌리고, 그동안 **은선 GPU는 intent가 필요 없는 모델 5개**를 먼저 돌린다.
intent가 필요한 모델 2개는 intent 묶음이 나오는 대로 진욱 GPU가 돌린다 (API 호출은 GPU를 안 써서 동시에 가능).
구조·실행법은 [`README.md`](README.md), 깨면 안 되는 규칙은 [`AGENTS.md`](AGENTS.md).

## 역할 분담

| | 김진욱 (RTX 3060 Ti 8GB) | 유은선 |
|---|---|---|
| LLM/API | TO step2 → D → TO 묶음, TO_GLOBAL step1·step2 → D → TO_GLOBAL 묶음 | — |
| GPU | **intent 모델 2개** × 3 seed × 2 split: `IKGR_dyn`, `IKGR_lgcn_dyn` (묶음 나온 split부터) | **intent 없는 모델 5개** × 3 seed × 2 split: `BPR`, `IKGR_kgoff`, `IKGR_dyn_nointent`, `LightGCN`, `IKGR_lgcn_dyn_nointent` — **지금 시작** |
| 결과 | 은선 결과 병합 + 3-seed 리포트 | split·단계가 끝날 때마다 결과 zip 전달 |

- 은선 작업은 intent 결과가 필요 없어서 API가 끝나길 기다리지 않는다. 데이터는 ml-25m으로 직접 만든다 (§0).
- 같은 seed의 모델이 두 PC에 나뉜다. 비교는 3-seed 평균으로 하고, PC 간 차이가 얼마나 되는지 보려고
  **두 사람 모두 `BPR` seed 2020 (TO)** 을 돌려서 값을 비교한다 (BPR은 10분 정도).

## 학습 설정 (11/13 확정, 모든 모델 동일)

| 항목 | 값 | 이유 |
|---|---|---|
| 배치 | **8192** (`config.yaml` `train_batch_size`) | LightGCN 계열은 배치마다 그래프 전체(엣지 약 3,900만)를 전파해서 시간이 배치 개수에 비례. 8192 × 40 epoch ≈ 이전 레포(2048 × 12 epoch)와 업데이트 횟수가 비슷함 |
| 인접행렬 | **CSR** (`sparse_format`) | COO와 계산 동일(테스트로 확인), GPU에서 2.9배 빠름 |
| epoch | 최대 40, 2 epoch마다 검증, 3번 연속 개선 없으면 정지 | Goodreads와 같은 공통 프로토콜 |
| 차원·학습률 | 64 / 0.001 | 그대로 |

진욱 PC 실측 (RTX 3060 Ti, TO, `scripts\benchmark_speed.bat`): LightGCN 2048·COO 114분/epoch(40 epoch 76시간) → **8192·CSR 9.8분/epoch(6.5시간)**,
IKGR_lgcn_dyn_nointent 6.9시간, BPR·IKGR_dyn_nointent 0.1~0.4시간. TO_GLOBAL은 train이 작아서 약 0.8배.

## 현재 상태

| 단계 | TO (유저별 80/10/10) | TO_GLOBAL (전체 70/10/20) |
|---|---|---|
| 데이터 준비 (ml-25m, k=30, 이전 레포와 규모 일치) | ✅ 133,684 유저 / 15,785 아이템 / 24,046,865 | ✅ |
| step1: exact intent (API) | ✅ 11/12 | ⬜ 유저 약 99k |
| step2: related intent (API) | ✅ 11/14 15:25 (키 5개 중 2개 소진, 예비 키로 마무리) | ⬜ 약 115k |
| D: KG pack | ✅ 11/14 15:26 | ⬜ |
| `data_bundle/` 업로드 | ✅ 11/14 (메타 KG 제외, 아래 ⚠) | ⬜ 11/16~17 (진욱 PC를 계속 돌릴 때) |

## 사람별 체크리스트

### 유은선 — intent 없는 모델 5개 (지금 시작)
| # | 할 일 | 예상 시간 (RTX 3060 Ti 기준) | 상태 |
|---|---|---|---|
| 은1 | §0 준비: clone(`team-mate`) → venv → ml-25m 데이터 → 메타 KG (`[scale] MATCH` 확인) | 30분~1시간 | ⬜ |
| 은2 | (권장) `scripts\benchmark_speed.bat` → 자기 GPU 속도 확인, 결과 표에 기입 | 10~15분 | ⬜ |
| 은3 | 보정용: `BPR` seed 2020 TO (아래 1단계에 포함됨) → 값 진욱과 비교 | — | ⬜ |
| 은4 | **1단계 (빠른 MF 모델 3개)**: TO → TO_GLOBAL, seed 2020·2021·2022 | 약 10~20시간 (대부분 평가·로딩) | ⬜ |
| 은5 | 1단계 결과 zip → 진욱 | — | ⬜ |
| 은6 | **2단계 (그래프 모델 2개)**: `LightGCN`, `IKGR_lgcn_dyn_nointent`, TO | 약 45시간 | ⬜ |
| 은7 | 2단계 TO_GLOBAL | 약 36시간 | ⬜ |
| 은8 | split이 끝날 때마다 결과 zip → 진욱 | — | ⬜ |

합계 약 4일 (GPU가 3060 Ti보다 느리면 더 걸림 → 은2 결과를 보고 진욱과 조정: 예) TO_GLOBAL의 그래프 모델 일부를 진욱 GPU로).
첫 LightGCN 실행에서 2 epoch마다 찍히는 `valid ... ndcg@10` 값이 올라가는지 처음 몇 시간 동안 확인해서 진욱에게 공유한다.

### 김진욱 — API + intent 모델 2개
| # | 할 일 | 시작 조건 | 상태 |
|---|---|---|---|
| 진1 | TO step2 이어서 실행 (`scripts\intent_run_*h.bat`, GPU 안 씀) → 끝나면 D 자동 실행 | 지금 | ✅ 11/14 |
| 진2 | 보정용 `BPR` seed 2020 TO 한 번 (`--seeds 2020 --specs BPR`) → 은선 값과 비교 | 지금 (메타 KG 있음) | ⬜ |
| 진3 | `python scripts/pack_data_bundle.py --split TO` → `data_bundle/` 커밋·푸시 | 진1 완료 | ✅ 11/14 (메타 KG 제외) |
| 진4 | **TO intent 모델**: `IKGR_dyn`, `IKGR_lgcn_dyn` × seed 2020·2021·2022 (API와 동시에) | 진1 완료 + 메타 KG 통일(⚠) | ⬜ 약 27시간 |
| 진5 | TO_GLOBAL step1 (`scripts\intent_global_step1_7h.bat`, GPU 숨김) | 진1 완료 | ⬜ |
| 진6 | TO_GLOBAL step2 시작은 **낮에**, GPU 학습을 seed 경계에서 잠깐 멈추고 (`scripts\intent_global_run_7h.bat`, 첫 GPU 인코딩 약 15분) → D | 진5 완료 | ⬜ |
| 진7 | `python scripts/pack_data_bundle.py --split TO_GLOBAL` → 커밋·푸시 | 진6 완료 | ⬜ |
| 진8 | **TO_GLOBAL intent 모델** × 3 seed | 진6 완료 | ⬜ 약 22시간 |
| 진9 | 은선 결과 병합 + 리포트 (아래 "결과 모으기") | 은5·은8 | ⬜ |

## 0. 준비 (처음 한 번, 은선)

```powershell
git clone -b team-mate https://github.com/HK1DS/movielens_november
cd movielens_november
powershell -ExecutionPolicy Bypass -File scripts\setup_venv.ps1        # Windows + NVIDIA GPU (venv: %USERPROFILE%\.venvs\goodreads-ikgr, Goodreads venv 재사용)
& "$env:USERPROFILE\.venvs\goodreads-ikgr\Scripts\Activate.ps1"
python -m pytest -q tests/                                              # (선택) 전부 passed 확인
```
Linux: `pip install torch==2.3.0 --index-url https://download.pytorch.org/whl/cu121 && pip install -r requirements-pipeline.txt`.

**데이터:** ml-25m을 직접 받아서 무료 준비 단계를 돌린다 (API 호출 없음, 15~30분, 디스크 약 4GB).
```powershell
# https://files.grouplens.org/datasets/movielens/ml-25m.zip 을 받아서 raw\ml-25m\ 에 압축 해제 (ratings.csv, movies.csv)
python run_pipeline.py --k 30 --split TO --steps 0ATP          # 콘솔에 [scale] MATCH 가 떠야 함 (위 표의 숫자와 동일)
python run_pipeline.py --k 30 --split TO_GLOBAL --steps P
python build_meta_kg.py --inter data/kc_k30/ikgr-custom/ikgr-custom.inter --out run_k30_TO/meta_kg_pack.pt
python build_meta_kg.py --inter data/kc_k30/ikgr-custom/ikgr-custom.inter --out run_k30_TO_GLOBAL/meta_kg_pack.pt
```
`[scale] MISMATCH`가 뜨면 멈추고 진욱에게 알린다 (데이터가 다르면 결과를 합칠 수 없음).
나중에 묶음(`python scripts/unpack_data_bundle.py`)으로 메타 KG·provenance를 덮어써도 내용이 같으면 끝난 결과는 그대로 유지된다.

**매 실행 전에** (새 터미널마다): `$env:PYTHONIOENCODING = "utf-8"` (bash: `export PYTHONIOENCODING=utf-8`)

### 공통 규칙
- 중단: `Ctrl+C` 한 번. 재개: **같은 커맨드 다시 입력** → 끝난 모델×seed는 건너뛰고, 돌던 것만 epoch 0부터 다시.
- 손해 없이 끊는 시점: 콘솔에 새 `===== <모델> seed=<n> (<SPLIT>) =====` 헤더가 뜬 직후.
- 실행 중에는 코드와 `config.yaml`을 수정하지 않는다 (코드/설정 해시가 바뀌면 끝난 결과까지 무효화).
- 두 사람 모두 **`team-mate` 브랜치 최신 상태**(`git pull`)로 돌린다. 코드·설정이 다르면 결과를 합칠 수 없다 (병합 스크립트가 거부함).
- 배치·epoch·학습률 등은 커맨드에서 바꾸지 않는다 (`--batch`, `--epochs` 옵션 쓰지 않기). 한 GPU에 학습을 두 개 동시에 돌리지 않는다.

## 커맨드

### 은선 1단계 — 빠른 MF 모델 3개 (BPR 포함 → 보정값도 여기서 나옴)
```powershell
python run_pipeline.py --k 30 --split TO        --steps E --seeds 2020,2021,2022 --specs BPR,IKGR_kgoff,IKGR_dyn_nointent
python run_pipeline.py --k 30 --split TO_GLOBAL --steps E --seeds 2020,2021,2022 --specs BPR,IKGR_kgoff,IKGR_dyn_nointent
```

### 은선 2단계 — 그래프 모델 2개 (오래 걸림, 중간에 끊고 이어 가도 됨)
```powershell
python run_pipeline.py --k 30 --split TO        --steps E --seeds 2020,2021,2022 --specs LightGCN,IKGR_lgcn_dyn_nointent
python run_pipeline.py --k 30 --split TO_GLOBAL --steps E --seeds 2020,2021,2022 --specs LightGCN,IKGR_lgcn_dyn_nointent
```
seed를 하나씩 나눠 돌려도 된다 (`--seeds 2020`만 먼저 등).

### 진욱 — 보정용 BPR, intent 모델 (묶음 나온 split부터)
```powershell
python run_pipeline.py --k 30 --split TO --steps E --seeds 2020 --specs BPR
python run_pipeline.py --k 30 --split TO        --steps E --seeds 2020,2021,2022 --specs IKGR_dyn,IKGR_lgcn_dyn
python run_pipeline.py --k 30 --split TO_GLOBAL --steps E --seeds 2020,2021,2022 --specs IKGR_dyn,IKGR_lgcn_dyn
```

### 결과 전달 (은선 → 진욱)
`run_k30_<SPLIT>/slice_eval_<SPLIT>_result.json`과 `run_k30_<SPLIT>/per_user_<SPLIT>/` 폴더를 zip 하나로 묶어
(예: `eunseon_TO_stage1.zip`) 진욱에게 보낸다. 같은 파일을 다시 보내도 새 seed·모델만 추가된다.

### 결과 모으기 (진욱)
```powershell
python scripts/merge_results.py --split TO --inputs eunseon_TO_stage1.zip   # 진욱의 run_k30_TO/에 합쳐짐 (기존 파일은 .bak)
python paired_compare.py --run_dir run_k30_TO --split TO --report           # → run_k30_TO/report_TO.md
```
코드·데이터·설정이 다른 결과는 합쳐지지 않고 이유(`code`, `data_fingerprint`, `protocol` …)가 출력된다 → 같은 브랜치로 다시 돌린다.

## GPU 속도 (benchmark_speed.bat 결과, 40 epoch 기준 시간)

| 이름 | GPU | LightGCN (8192·CSR) | IKGR_lgcn_dyn_nointent (8192·CSR) | BPR | IKGR_dyn_nointent |
|---|---|---|---|---|---|
| 김진욱 | RTX 3060 Ti 8GB | 6.5시간 (9.8분/epoch) | 6.9시간 | 0.1시간 | 0.4시간 |
| 유은선 | | | | | |

평가·데이터 로딩은 별도로 실행 1회당 대략 30분~1시간 더 걸린다 (추정).

## 보정값 (BPR seed 2020, TO)

| 이름 | NDCG@10 | Recall@10 |
|---|---|---|
| 김진욱 | | |
| 유은선 | | |

## 참고: 데이터 묶음 (Goodreads와 같은 방식)

- `data_bundle/movielens_k30_<SPLIT>_bundle.tgz.partNN` (95MB씩 분할) + `.sha256` + `.files.json`, 복원은 `scripts/unpack_data_bundle.py`.
- TO 묶음 (11/14 업로드): `data/kc_k30/ikgr-custom/ikgr-custom.inter`, 규모 통계, TO 프로필·provenance, `run_k30_TO/kg_pack.pt`.
  `meta_kg_pack.pt`는 **뺐다** (아래 ⚠). 풀어도 각자 만든 메타 KG는 그대로 남는다.
- TO_GLOBAL 묶음: TO_GLOBAL 프로필·provenance, `run_k30_TO_GLOBAL/kg_pack.pt`, `meta_kg_pack.pt` (`.inter`는 TO 묶음과 공통).
- `interactions_k30.csv`(700MB)는 `.inter`와 같은 내용이고 학습에 안 쓰여서 넣지 않는다.
- 은선은 intent 모델을 돌리지 않으므로 묶음이 꼭 필요하지는 않다 (참고·검증용).

### ⚠ 메타 KG (`meta_kg_pack.pt`) — 다시 만들거나 덮어쓰지 말 것 (11/14 확인)

`build_meta_kg.py`는 실행할 때마다 장르 노드 번호 순서가 달라진다 (아이템을 파이썬 set 순서로 읽음). 그래서 두 PC의
`meta_kg_pack.pt` 내용이 서로 다르고, 파일을 바꾸면 그 파일을 읽는 `IKGR_dyn_nointent`·`IKGR_lgcn_dyn_nointent`(진욱은
`IKGR_dyn`·`IKGR_lgcn_dyn`)의 끝난 결과가 서명 불일치로 **처음부터 다시 학습된다**. `BPR`·`LightGCN`·`IKGR_kgoff`는 안 읽어서 무관.
- 은선: 지금 가진 `run_k30_TO/meta_kg_pack.pt`, `run_k30_TO_GLOBAL/meta_kg_pack.pt`를 그대로 둔다.
- intent 모델과 intent 없는 모델을 같은 메타 KG로 비교하려고, 진욱이 intent 모델(진4)을 시작하기 전에 어느 파일로 통일할지 정한다.

