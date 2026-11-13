# IKGR 파이프라인 — MovieLens (아키텍처 뼈대)

한양대 데이터사이언스 졸업프로젝트(지도교수 김현준)의 Goodreads 실험 레포
(`졸작_11월/Goodreads_experiment-main`)에서 검증한 **누수 없는(train-only) 추천 파이프라인 아키텍처**를
데이터셋과 분리해서 옮긴 레포다. 데이터·결과·캐시·레거시 도구는 가져오지 않았고, 데이터셋에 묶여 있던
부분은 전부 `data_adapters/` 하나로 뺐다.

> **데이터셋 버전: MovieLens 25M (`ml-25m`) 고정** — 이전 MovieLens 레포
> [HK1DS/hk1ds_jwkim_movielens](https://github.com/HK1DS/hk1ds_jwkim_movielens)와 같은 버전.
> [`ml-25m.zip`](https://files.grouplens.org/datasets/movielens/ml-25m.zip)을 받아 `raw/ml-25m/`에 풀면 된다
> (`ratings.csv`, `movies.csv`; `raw/`는 git 제외). 다른 버전(ml-32m, ml-latest, ml-1m)으로 바꾸면 이전 결과와 비교할 수 없다.
>
> **상태:** 뼈대만 있다. `data_adapters/movielens.py`의 두 메서드(`iter_interactions`, `load_items`)가
> TODO라서 MovieLens로는 아직 돌지 않는다. 나머지 단계는 합성 데이터로 0→R 전 구간이 테스트로 검증돼 있다(§6).

---

## 1. 아키텍처 한눈에

```
 raw 데이터 ──(adapter)──▶ 0 interactions.csv ─▶ A k-core ─▶ T RecBole .inter  ◀── 평가의 유일한 기준 파일
                                                               │
                         ikgr_core/temporal_split.py ◀─────────┤ (RecBole 1.2.0 시간 split을 행 단위로 재현)
                                                               │
   P train-only 유저 프로필 + provenance ◀─────────────────────┤
     │ (각 split의 train 구간 중 첫 비-train 시점보다 엄격히 이전 행만)
     ▼
   S 비용 추정 ─▶ B step1: LLM exact intents ─▶ C step2: RAG(mpnet) + LLM related intents ─▶ D intent KG pack
                                                                                       D metadata KG pack ◀─ adapter.meta_relations
     ▼
   E eval_slices.py: 누수 가드 → 공통 학습 프로토콜(early stop, 검증 NDCG@10) → 슬라이스 평가
                     → per-user 덤프 → 인기도 페널티 재랭킹 대조군
   R paired_compare.py: 유저 단위 paired bootstrap CI + Wilcoxon + Pareto 표
```

split마다(`TO` = 유저별 80/10/10, `TO_GLOBAL` = 전역 시간 컷 70/10/20) 프로필·intent·KG·결과를 따로 만든다
(`data/kc_k{K}/…_{SPLIT}`, `run_k{K}_{SPLIT}/`). TO용 train-only 프로필은 TO_GLOBAL에선 train-only가 아니기 때문이다.

## 2. 모델 (`ikgr_core/model_ikgr.py`, RecBole `GeneralRecommender`, BPR 학습)

| 구성요소 | 설정 키 | 내용 |
|---|---|---|
| CF 베이스 | `lgcn_layers` | 0이면 MF 임베딩, >0이면 train 그래프 위 LightGCN 전파(RecBole LightGCN과 점수 동일, 테스트로 확인) |
| intent KG | `use_kg`, `kg_layers`, `intent_learnable` | LLM intent 노드(mpnet 임베딩 랜덤투영 초기화) ↔ user/item, 패딩 gather 평균 + 게이트 |
| 메타데이터 KG | `use_meta_kg` | **어댑터가 정의한 relation마다** 노드 임베딩 테이블 1개 + 게이트 1개 (Goodreads: author/publisher/shelf) |
| recency 동적 프로필 | `use_dynamic`, `recency_topn`, `recency_tau_days` | train 구간 최근 N개 아이템의 지수 감쇠 가중 평균 (DynLLM 착안) |
| 융합 | 기본 / `profile_attn` / `use_corona` | 게이트 합(기본) / MHA / 채널별 late fusion(CORONA 착안) |
| 평가 전용 | `corona_cand`, `corona_rerank_grid`, pop grid | 그래프 후보생성(top-M 마스킹) / soft graph prior 재랭킹 / 인기도 페널티 대조군 |

주요 spec (`eval_slices.build_specs()`): `IKGR_kgoff`(=MF), `BPR`, `LightGCN`, `IKGR_dyn_nointent`(메타+recency),
`IKGR_dyn`(+intent), `IKGR_lgcn_dyn_nointent` / `IKGR_lgcn_dyn`(LightGCN 백본 위), 그 외 `IKGR_meta_only`,
`IKGR_full_hetero`, `IKGR_full`, `IKGR_cand_db`, `IKGR_rerank_db_rel` 등. 기본 실행 spec은 Goodreads 마지막
비교(2026-11, emb64)와 같은 7개다(`run_pipeline.py`의 `DEFAULT_SPECS`).

해석 원칙(Goodreads에서 가져옴): KG 계열이 long-tail/coverage를 올렸다고 말하려면, 같은 NDCG 수준에서
**CF + 인기도 페널티 곡선보다 위**에 있어야 한다(`report_*.md`의 Pareto ★/◆).

## 3. 디렉터리

```
config.yaml                    dataset(어댑터) / paths / llm / rag / recbole(공통 학습 프로토콜)
run_pipeline.py                단계 오케스트레이터 (--k, --split, --steps 0ATPSBCDER)
prepare_interactions.py        0  어댑터 → data/interactions.csv (+ dataset.min_rating)
apply_k_core.py                A  (k,k)-core
make_temporal_inter.py         T  RecBole .inter (+ 선택적 timestamp clip 창)
build_trainonly_profiles.py    P  train-only 유저 프로필 + provenance (형식은 어댑터 facet이 결정)
estimate_llm_cost.py           S  LLM 호출/비용 추정 (캐시 반영)
step1.py, step2.py             B, C  LLM intent 추출 / RAG 확장 (캐시·재개·Ctrl+C 안전; step2 RAG 입력은 한 번만 생성)
build_kg.py, build_meta_kg.py  D  intent KG pack / 메타데이터 KG pack(format 2)
eval_slices.py                 E  학습·평가·누수 가드·per-user 덤프·재랭킹 대조군
paired_compare.py              R  paired CI + Pareto 리포트
prompts/                       step1/step2 프롬프트 (step1은 데이터셋에 맞게 문구 수정 권장)
data_adapters/
  base.py                      어댑터 계약 (ItemMeta, ProfileFacet, MetaRelation)
  movielens.py                 ← TODO: MovieLens 어댑터 뼈대
  generic_csv.py               interactions.csv + items.csv 범용 어댑터 (테스트용, 변환 후 바로 쓰기용)
ikgr_core/
  temporal_split.py            RecBole 1.2.0 시간 split 재현 + profile-safe 경계
  model_ikgr.py                IKGR 모델
  corona_retriever.py          그래프 후보생성 / soft prior (relation별 채널)
  llm_client.py, rag.py, jobs.py, utils.py
tests/                         단위 테스트 + 가짜 LLM/인코더 스모크 테스트
scripts/setup_venv.ps1         Windows GPU venv (기존 goodreads-ikgr venv 재사용)
```

## 4. 새 데이터셋 붙이기 — 어댑터 계약

파이프라인에서 raw 파일을 읽는 건 어댑터뿐이다. `config.yaml`의 `dataset.adapter`로 고른다.

| 어댑터가 제공 | 쓰는 곳 | 의미 |
|---|---|---|
| `iter_interactions()` | 0 | `(user_id, item_id, rating, timestamp_epoch)` 스트림. timestamp 없으면 0.0 |
| `load_items(ids)` → `ItemMeta` | P, D | `title`(유저 프로필의 최근 목록에 들어갈 표시명), `profile`(아이템 intent용 정적 텍스트), `relations`(relation → 노드 목록) |
| `recent_label`, `title_order`, `title_n`, `profile_window`, `profile_facets` | P | 유저 프로필 형식: `"<recent_label>: t1; t2 … | <facet.label>: v1, v2 …"`. 윈도 = 최신 `profile_window`개 safe-train 행(None = train 이력 전체), 제목 = 윈도에서 `title_order`(`recent` 최신순 / `rating` 평점 높은 순, 동점은 최신순) 앞 `title_n`개, facet = 윈도 전체 집계. provenance = 윈도 전체 |
| `meta_relations` | D, E | 메타 KG 노드 타입 (relation별 `max_per_item`, `min_df`) |
| `ts_clip` | T | timestamp clamp 창 (`dataset.ts_clip`으로 덮어쓰기 가능) |
| `source_files()` | D | 내용이 바뀌면 KG pack을 다시 만들 원본 파일 목록 |

**어댑터 누수 규칙:** 아이템 메타데이터는 정적 카탈로그 정보여야 한다. 유저 활동을 전 기간에 걸쳐 모은 신호
(유저가 단 태그, 리뷰, 인기 shelf 카운트 등)는 검증/테스트 기간 행동을 포함한다. 쓰려면 train 구간으로
다시 만들거나, 최소한 그 결정을 기록해 둔다. 누수 가드는 **유저 프로필** 쪽만 검사하고 아이템 메타데이터는 검사하지 않는다.

### MovieLens 어댑터 (`data_adapters/movielens.py`) — 결정 사항

버전은 **ml-25m 고정**(평점 25,000,095 · 유저 162,541 · 영화 62,423). 2026-11-10에 이전 레포
(`hk1ds_jwkim_movielens`의 `prepare_movielens.py`)와 **같은 선택**으로 구현했다(비교 가능성 우선):

| 항목 | 선택 | 바꾸려면 |
|---|---|---|
| positive | 모든 평점을 implicit interaction으로 유지 | `dataset.min_rating` |
| `(no genres listed)` 영화 | 평점째 제외 → 평점 24,973,468 · 유저 162,541 · 영화 54,479 | `options.drop_no_genres: false` |
| 아이템 텍스트 | `"title \| genres"` (예: `Toy Story (1995) \| Adventure\|Animation\|…`) | 어댑터 `load_items` |
| 메타 KG / 유저 프로필 facet | `genre` 하나 / `Frequent genres: Drama (412), …` | 어댑터 클래스 속성 |
| 유저 프로필 제목 목록 | **train 이력 전체에서 평점 높은 15편**(동점은 최신순), 장르 분포도 이력 전체 집계 → `Favorite movies (highest rated first): …` | 어댑터 `title_order` / `title_n` / `profile_window` |
| 태그 (`tags.csv`, genome) | **사용 안 함** (전 기간 집계 → 누수) | — |
| 참고: 이전 레포 k=30 core | 유저 133,684 · 아이템 15,785 · 상호작용 24,046,865 | — |

유저 프로필은 이 레포의 정확한 train-only 프로필이라 이전 레포("이력 앞 80%")와 텍스트·intent가 다르다.
제목 목록을 "최근 시청"이 아니라 "평점 상위"로 고른 이유(2026-11-10): MovieLens timestamp는 평점 **입력** 시각이라
가입 때 수백 편을 한꺼번에 매긴 유저의 "최근 50편"은 의미 없는(대개 제목 알파벳순) 덩어리다. 명시적 평점이 취향 신호로 더 강하고,
최근성은 모델의 recency 채널이 timestamp로 따로 다룬다. 제목만 보여주고 평점 숫자는 넣지 않는다.

남은 우려(조용히 바꾸지 말 것):
- 아이템 텍스트가 제목+장르뿐이라 LLM 아이템 intent가 장르를 되풀이할 가능성이 크다(= genre 메타 채널과 중복).
  intent 채널을 의미 있게 하려면 외부 정적 텍스트(예: `links.csv` → TMDB 줄거리)가 필요. 파일럿 리포트의 샘플로 먼저 확인.
- timestamp는 "평점 입력 시각"이라 가입 직후 몰아서 평가한 구간이 많다. TO의 "최근 소비" 의미가 Goodreads와 다르다.
- `prompts/step1_intents.txt`는 아직 범용 문구다(영화 도메인용으로 다듬을지 파일럿 결과 보고 결정).

## 5. 실행 (Windows, 기존 GPU 환경)

준비: `.env`에 Luxia 키(아래 "키 여러 개" 참고). Python은 기존 `%USERPROFILE%\.venvs\goodreads-ikgr`를 그대로 쓴다
(없으면 스크립트가 `scripts\setup_venv.ps1`로 만든다).

**intent augmentation (LLM 단계 B/C)** — 탐색기에서 더블클릭:

| 파일 | 하는 일 |
|---|---|
| `scripts\intent_prep.bat` | **무료 준비만** (LLM 호출 없음): ml-25m 다운로드 → k=30 core → TO·TO_GLOBAL 두 split의 train-only 프로필 → 이전 레포 규모와 대조 → 두 split의 LLM 비용 추정 |
| `scripts\intent_pilot.bat` | ml-25m 다운로드·압축해제(없으면) → 무료 준비 `0ATPS` → **파일럿**: 실제 프로필 300명+아이템 100개(~800회 호출)로 Luxia 처리속도를 재서 전체 소요시간을 외삽하고, intent 샘플을 `run_k30_TO_pilot\pilot_report.md`에 남김 |
| `scripts\intent_full.bat` | (준비가 끝났으면 건너뜀) → 전체 step1·step2 + KG pack(D). 첫 유료 호출 전에 비용 추정과 y/N 확인 |
| `scripts\intent_run_7h.bat` | 전체 step1·step2를 **최대 7시간만**(확인 없이) 돌리고 캐시 저장 후 일시정지. 같은 파일을 다시 실행하면 이어서 |
| `scripts\intent_run_1h.bat` / `_2h` / `_5h` / `_6h` / `_9h` | 같은 동작, 최대 1·2·5·6·9시간. GPU는 step2가 RAG 입력을 처음 만들 때만 잠깐 씀 |
| `scripts\intent_finish_step1_5h.bat` | step1(B)만 최대 5시간(`-PaidSteps B`), step2는 시작하지 않음. GPU 숨김 |
| `scripts\intent_stop.bat` | 실행 중인 step1·step2를 바로 일시정지(캐시 저장). 아무 `intent_run_*`로 이어서 |

모두 Ctrl+C로 멈추고 같은 파일을 다시 실행하면 이어서 한다. 파일럿의 step1 결과는 본 실행이 재사용한다.

**데이터 규모 = 이전 MovieLens 레포와 동일 (k=30 전체).** 이전 레포의 최종 패키지는 ml-25m 전체를 k=30으로 거른
유저 133,684 · 아이템 15,785 · 상호작용 24,046,865(TO·TO_GLOBAL 공통)였다(1k/5k 샘플은 중간 점검용). 같은 입력
(장르 없는 영화 제외 후 평점 24,973,468개)에 같은 (k,k)-core 정의라 숫자가 정확히 같아야 하고, `run_pipeline.py`가
단계 0·A 직후 이 기준값과 대조해 `run_k30_<SPLIT>\scale_check.json`에 남긴다. **다르면 유료 단계(B/C)를 막는다**
(`--ignore_scale_check`로 의도적으로만 우회). 기준값은 `data_adapters/movielens.py`의 `REFERENCE`.

**소요 시간 (k=30, TO)** — 호출 수 ≈ (유저 ~13.4만 + 아이템 ~1.6만) × 2단계 ≈ **30만 회**:

| Luxia 처리속도 | 근거 | 전체 B/C |
|---|---|---|
| ~4.8 회/초 (동시 8, 제한 없음) | 2026-11 Goodreads step2 실측(15,198회/53분) | **약 17시간** |
| ~1 회/초 (429 후 1초 간격) | 2026-07 이전 MovieLens 레포에서 Luxia가 동시 8에 429 → 1초 페이싱 | **약 3.5일** |
| **~1.1 회/초 (≈4,000회/시간)** | **2026-11-10 이 레포 실측** (TO step1: 아이템 15,784 + 유저 13,062회, `run_k30_TO/llm_progress_step1.csv`). 429가 계속 나서 간격이 0.5–1초에 머묾. 유저 프롬프트가 아이템보다 10배 길어도 속도가 같음 → 토큰이 아니라 **요청 수 제한** | 남은 TO+TO_GLOBAL B/C 약 48만 회 → **약 5일** |
| **~1.5 회/초 (≈5,400회/시간)** | **2026-11-10 키 3개 병렬 1시간 실측** (5,443회; 키별 1,935 / 1,736 / 1,772회, 429는 11 / 18 / 10회). 키 1개 대비 **+34%** — 3배가 아니므로 제한이 순수 키 단위는 아님(계정·서비스 공용 제한이 섞인 것으로 보임). 키별 요청 간격이 1.4–3.6초까지 올라감 | 남은 약 47.7만 회 → **약 3.7일** |
| **~1.2 회/초 (≈4,400회/시간)** | **2026-11-11 키 3개 병렬, 6시간+5시간 실측** (26,047회 / 22,482회). 1시간 테스트의 +34%는 초반 효과였고 장시간 평균은 키 1개와 거의 같음(+7~11%). 요청을 거의 안 보내는 키도 429를 받음 → 제한은 **계정·서비스 공용**. 키 하나가 대부분을 처리하게 됨(합계는 동일). 빨라지려면 다른 계정의 키가 필요 | 11/11 밤 기준 남은 약 43만 회 → **약 97시간(4일)** |

어느 쪽인지는 계정 한도에 달려 있어서 파일럿 리포트의 `projected`가 실제 값이다. 무료 준비(0ATPS)는 25M 행 처리로 약 15–30분,
step2 첫 실행은 mpnet 인코더(~420MB) 다운로드가 붙는다. 비용은 `S` 단계가 계산한다(Goodreads 관측 단가 기준 $0.50–0.61/1천 회 → 30만 회면 약 $150–180, 정가 기준은 그보다 낮음).

**step2의 RAG 입력은 split마다 한 번만 만든다**(`ikgr_core/rag.py`, 2026-11-12). 첫 step2 실행이 step1 결과에서
① intent 어휘(정렬된 고유 exact intent, `intent_vocab.json`), ② 그 임베딩(`intents_emb.npy`, D단계 `build_kg.py`가 사용),
③ 유저·아이템 프로필마다 가장 비슷한 `knn_k`(100)개 후보 intent(`rag_candidates.npz`, **정확한 코사인 top-k**)를 만들고
`rag_manifest.json`(인코더·k·어휘/프로필 해시)을 마지막에 쓴다. 이어서 실행할 때는 이 파일을 그대로 읽고 **다시 인코딩하지 않는다**.
인코더는 GPU가 보이면 GPU에서 돈다(`IKGR_ENCODER_DEVICE=auto|cuda|cpu`, 배치 `IKGR_ENCODER_BATCH`; TO 기준 GPU로 수 분, CPU로 1–2시간).
예전 Annoy/sklearn 근사 인덱스는 없앴다(매 실행 재구축·Annoy의 비결정성 때문에 일시정지 전후 후보가 달라질 수 있었음).
API 호출 없이 이 파일만 만들려면 `python step2.py --prepare_only` (`IKGR_CONFIG=run_k30_TO/config.yaml`).
step2 결과가 쌓인 뒤 step1 결과가 바뀌면 두 어휘가 섞이지 않게 **exit 2로 멈춘다**(`step2_cache.json`을 지우고 다시 시작하거나,
`IKGR_STEP2_REBUILD=1`로 재구축하고 기존 결과 유지). 이 RAG 파일 3개 + `rag_manifest.json`은 팀원과 결과를 공유할 때 같이 넘긴다.

직접 실행할 때:

```bash
python run_pipeline.py --k 30 --split TO --steps 0ATPS        # 무료 준비 + LLM 비용 추정
python pilot_llm.py --k 30 --split TO                          # 파일럿 (처리속도·소요시간·샘플)
python run_pipeline.py --k 30 --split TO --steps BCD           # LLM(확인 프롬프트) → KG pack
python run_pipeline.py --k 30 --split TO --steps ER --seeds 2020 --specs IKGR_lgcn_dyn,LightGCN
python paired_compare.py --run_dir run_k30_TO --split TO --a IKGR_lgcn_dyn --b LightGCN IKGR_lgcn_dyn_nointent --by_activity
```

**시간 제한·추세:** `run_pipeline.py --max_hours H`면 H시간 뒤 새 호출을 시작하지 않고 캐시를 저장한 뒤 exit 5로 멈춘다(같은 명령으로 재개).
실행 중 `run_k30_<SPLIT>\STOP` 파일을 만들면(`scripts\intent_stop.bat`) 같은 방식으로 바로 일시정지한다(Ctrl+C 없이, 다음 실행 시작 때 자동 삭제).
ps1의 `-Until HH:mm`은 오늘 그 시각까지로 시간 제한을 건다. `-RebuildProfiles`는 프로필 형식을 바꾼 뒤 두 split의 프로필을 다시 만든다
(아이템 intent 캐시는 그대로 재사용; `scripts\intent_reprofile_run.bat` = 재생성 후 오늘 19:32까지 이어서 실행).
step1/step2는 1분마다 `run_k30_<SPLIT>\llm_progress_step{1,2}.csv`에 누적 호출 수(아이템/유저), 구간·평균 처리속도,
429·HTTP·기타 오류 수, 현재 요청 간격, 키 상태를 남긴다(재개한 실행은 같은 파일에 이어 붙음 → 그대로 추세 그래프).
아이템을 먼저 처리한다(카탈로그 텍스트만 쓰므로 split·유저 프로필 형식과 무관하게 재사용됨). Windows에서는 실행 중 절전을 막는다.

LLM 속도 관련 환경변수(요청 간격은 **키마다 따로** 관리된다): `IKGR_LLM_KEY_MODE`(`parallel` 기본 = 앞쪽 키 `IKGR_LLM_ACTIVE_KEYS`개(기본 3, 0이면 전부) 동시 호출 / `sequential` = 한 번에 키 하나),
`IKGR_LLM_WORKERS_PER_KEY`(키당 동시 요청 수, 기본 8 → 동시 호출 키 3개면 워커 24), `IKGR_STEP1_WORKERS` / `IKGR_STEP2_WORKERS`(설정하면 전체 워커 수를 고정),
`IKGR_LLM_MIN_INTERVAL_SECONDS`(처음부터 요청 간격, 기본 0), `IKGR_LLM_429_MIN_INTERVAL_SECONDS`(그 키의 첫 429 이후 간격, 기본 1.0),
`IKGR_LLM_PACE_RECOVER_EVERY`(그 키가 429 없이 이만큼 성공하면 간격을 `IKGR_LLM_PACE_RECOVER_FACTOR`=0.75배로 완화, 기본 200),
`IKGR_LLM_PACE_RECOVER_SECONDS`(또는 마지막 429·완화 뒤 이만큼 429가 없으면 한 단계 완화, 기본 60초 — 10초 간격에 묶인 키가 성공 횟수를 못 채워 몇 시간씩 느려지는 것 방지; 둘 다 0이면 끝까지 유지),
`IKGR_CACHE_SAVE_EVERY`(캐시 저장 간격, 기본 500회).

중단/재개는 Goodreads 레포와 같다: Ctrl+C 후 같은 명령을 다시 실행하면 0/A/T/P는 산출물이 있으면 건너뛰고,
B/C는 LLM 캐시에 없는 프로필만 호출하고, B/C/D는 입력 해시가 같으면 통째로 건너뛰고(`--redo D` 등으로 강제),
E는 끝난 spec×seed를 유지한다. `eval_slices.py`·`model_ikgr.py`·`corona_retriever.py`나 설정을 바꾸면
끝난 결과도 다시 학습한다(의도된 안전장치).

### Luxia API 키 여러 개 (병렬 호출 + 예비 키 + 소진 시 제외)

`.env`에 키를 여러 개 넣으면 step1/step2가 **앞쪽 키 3개를 동시에** 호출한다(`ikgr_core/key_lanes.py`, `IKGR_LLM_KEY_MODE=parallel` 기본,
개수는 `IKGR_LLM_ACTIVE_KEYS`). 그 뒤의 키(`LUXIA_API_KEY_4`, `_5`, …)는 **예비 키**로 대기하다가, 쓰던 키 하나가 소진되면
`.env` 순서대로 하나씩 그 자리를 이어받는다(로그 `[llm-keys] standby … takes over`, 진행 로그 `keys` 열 `#4 standby`).
제한이 계정 공용이라 키를 더 동시에 써도 빨라지지 않으므로 예비 키는 크레딧만 아껴 둔다. 값이 비어 있는 줄(`LUXIA_API_KEY_4=`)은 무시된다.
키마다 요청 간격(429 쿨다운·AIMD)과 동시 요청 상한(`IKGR_LLM_WORKERS_PER_KEY`, 기본 8)이 따로 있어서, 429를 받은 키만 느려지고
새 요청은 지금 가장 빨리 보낼 수 있는 키로 간다. 크레딧/토큰이 떨어진 키는 빠지고 남은 키가 이어받는다(`ikgr_core/key_pool.py`).
Luxia 제한이 **키 단위**면 키 개수만큼 빨라지고, **계정 단위**(같은 계정의 키들)면 총 처리량은 비슷하고 429만 늘어난다.
어느 쪽인지는 진행 로그 `keys` 열(`parallel: #1 0.56s 4012ok/20x429 | #2 …`)과 `step1_throughput.json`의 `per_key`로 확인한다.
예전처럼 한 키씩 소진될 때까지 쓰려면 `IKGR_LLM_KEY_MODE=sequential`.
키 순서는 `LUXIA_API_KEY` → `LUXIA_API_KEY_1`, `_2`, … → `LUXIA_API_KEYS=k1,k2` (섞어 써도 됨).

```ini
LUXIA_API_KEY=첫번째키
LUXIA_API_KEY_2=두번째키
LUXIA_API_KEY_3=세번째키
LUXIA_API_KEY_4=예비키1     # 위 키 중 하나가 소진되면 투입
LUXIA_API_KEY_5=예비키2
```

| 응답 | 판단 | 동작 |
|---|---|---|
| HTTP 401 / 402 / 403 | 키 무효·미결제·차단 | 그 키를 빼고 같은 요청을 즉시 다른 키로 재시도 |
| 본문에 소진 문구(`insufficient`, `quota`, `credit`, `balance`, `크레딧`, `잔액`, `충전`, `부족` …) — HTTP 200 오류 JSON 포함 | 크레딧/토큰 소진 | 그 키를 빼고 즉시 다른 키 |
| 일반 HTTP 429 (본문에 소진 문구 없음) | rate limit | **그 키만** 쿨다운·간격 증가, backoff 후 재시도(가장 빨리 보낼 수 있는 키로) |
| 한 키가 성공 없이 429만 `IKGR_LLM_ROTATE_AFTER_429_SEC`초(기본 600) 이상 | 사실상 사용 불가 | 그 키를 뺌 (0이면 끔) |
| 모든 키 소진 | — | step1/step2가 캐시 저장 후 **exit 4로 중단**. 키를 추가/충전하고 같은 명령 재실행하면 남은 것만 이어서 호출 |

- Luxia 문서에는 크레딧 소진 응답 형식이 나와 있지 않다. 그래서 처음 보는 오류 응답은 `[llm] luxia HTTP <코드> (<판단>): <본문>`으로
  한 번씩 로그에 찍힌다. 실제 소진 메시지가 위 패턴에 안 걸리면 `IKGR_LLM_EXHAUSTED_PATTERN=<정규식>`으로 추가하면 된다.
- 키는 로그에 `#2/3 (...끝4자리)`로만 표시된다. 소진 상태는 프로세스 안에서만 유지되고, 재실행하면 1번 키부터 다시 확인한다(충전된 키 자동 재사용).
- 각 단계 시작 시 `[step1] API keys: parallel over 3 key(s): up to 8 request(s) in flight per key, 24 worker threads`가 찍힌다.

### 팀원과 나눠 돌리기 (GPU, E 단계)

LLM 단계는 API PC 한 대에서 돌리고, 팀원은 E 단계만 seed를 나눠 돌린다. 할 일 목록은 `team-mate` 브랜치의 `TODO.md`.

| 스크립트 | 누가 | 하는 일 |
|---|---|---|
| `scripts/pack_data_bundle.py --split TO` | API PC (D 끝난 뒤) | 평가 입력(TO는 공용 `.inter` 포함, 프로필·provenance, `kg_pack.pt`, `meta_kg_pack.pt`)을 `data_bundle/`에 95MB 분할 tgz + sha256 + 파일 목록으로 묶음. git으로 전달 (Goodreads와 같은 방식). `interactions_k30.csv`는 `.inter`와 같은 내용이라 제외 |
| `scripts/unpack_data_bundle.py` | 팀원 | sha256 확인 후 복원 (`--split TO`, `--check`) |
| `scripts/merge_results.py --split TO --inputs a.zip b_dir ...` | 결과 모으는 사람 | 팀원별 `slice_eval_<SPLIT>_result.json` + `per_user_<SPLIT>/`(폴더 또는 zip)를 합침. `run_signature`가 다르면(다른 코드·데이터·프로토콜로 평가) 합치지 않고 이유를 출력 |

**학습 설정 (2026-11-13):** 모든 spec 공통으로 배치 8192, LightGCN 인접행렬 CSR(`config.yaml`의 `train_batch_size`, `sparse_format`;
`run_pipeline.py --batch / --sparse`, env `IKGR_BATCH / IKGR_SPARSE`). LightGCN 계열(`LightGCN`, `IKGR_lgcn_*`)은 배치마다
그래프 전체(MovieLens TO 약 3,900만 엔트리)를 전파해서 epoch 시간이 배치 개수에 비례한다. RTX 3060 Ti 실측
(`scripts\benchmark_speed.bat` → `scripts/benchmark_train_speed.py`): LightGCN 2048·COO 114분/epoch(40 epoch 76시간),
2048·CSR 39분, **8192·CSR 9.8분(6.5시간)**, 16384·CSR 5.0분. CSR은 COO와 계산이 같다(`tests/test_lgcn_backbone.py`).
8192 × 40 epoch은 이전 MovieLens 레포(2048 × 12 epoch)와 업데이트 횟수가 비슷하다. 참고: 이전 레포의 `LightGCN`은
그래프 전파가 실제로 일어났다면 512차원·12 epoch으로 seed당 60시간이 넘었을 텐데, 전체 평가가 31시간에 끝났다
(그 레포의 `dok_matrix._update = dict.update` shim은 SciPy ≥ 1.13에서 인접행렬을 비운다) → 그 수치는 전파 없는 변형일 가능성이 크다.

평가 서명(`eval_slices.py`, schema 4)은 기계와 무관하다: spec이 실제로 읽는 데이터 파일만(intent spec: provenance + `kg_pack.pt`,
메타 KG spec: `meta_kg_pack.pt`), 그 **내용**으로 해시하고(.gz는 압축 해제 후, .pt는 텐서 값), 코드는 줄바꿈을 정규화해서,
경로는 레포 기준 상대경로로 기록한다. 그래서 intent 없는 spec 결과는 나중에 `kg_pack.pt`가 생겨도 무효화되지 않고,
팀원 PC마다 다시 만든 같은 내용의 파일도 같은 서명을 낸다.

`scripts\intent_global_step1_7h.bat` / `intent_global_run_7h.bat`: TO_GLOBAL의 step1만 / step1→step2→D (step2의 첫 GPU 인코딩은 낮에).

## 6. 테스트

```bash
python -m pytest -q tests/                         # split 재현(실제 RecBole 대조), train-only 프로필, LightGCN 동치, 메타 KG, rerank, API 키 교체
python tests/smoke_e2e.py --workdir <임시폴더>      # 합성 데이터 + 가짜 LLM/인코더로 0→R 전 단계 + 누수 가드 음성 테스트
```

API 비용 0, GPU 불필요. 합성 데이터는 `data_adapters/generic_csv.py` 형식이다.

## 7. Goodreads 레포에서 바뀐 점 / 옮기지 않은 것

바뀐 점
- 메타 KG pack이 **format 2**(`{"relations": {이름: {item_nodes, vocab, n_nodes}}}`)로 바뀌었다. 모델·retriever가
  relation 개수에 무관하게 동작한다. Goodreads의 `meta_kg_pack.pt`(author/publisher/shelf 고정 키)와는 호환되지 않는다.
- 유저 프로필 형식이 어댑터 facet으로 정의된다(Goodreads의 "Frequent authors / Common shelves"가 facet 2개로 일반화).
  ubiquitous 값 제거는 모든 facet에 적용된다. `legacy` 프로필 스타일은 없앴다.
- timestamp clip 창(Goodreads 2006–2017 하드코딩)이 파라미터가 됐다.
- 기본 spec·인기도 페널티 대상 spec을 Goodreads 마지막 비교(LightGCN 백본 포함 7개)에 맞췄다.
- `corona_retriever`의 relation 가중치 기본값 처리 수정(가중치 미지정 relation = 1.0).

옮기지 않은 것 (필요하면 원본 레포에 그대로 있다)
- Goodreads 데이터·`data/`·`run*/`·`log_tensorboard/`·캐시·`_share_bundle/`
- 레거시 보고서/로그: `AGENTS.md`(진행 기록), `IKGR_REPORT.md`, `DYNLLM_*`, `CORONA_*`, `explain.md`, `path.md`, `result_example.md`
- 참고용 원본 레포 사본: `CORONA-main/`, `dynmLLM/`
- 레거시 도구: `run_pipeline_auto.py`, `step3.py`, `run_baselines.py`, `run_rerank_orchestrator.py`, `verify_*.py`,
  `add_timestamps.py`, `convert_to_recbole_atomic.py`, `sample_dataset.py`, `build_intent_banks.py`(모델이 안 씀),
  `goodreads_preprocess.py`(→ 어댑터로 대체), Goodreads 전용 실행 스크립트(`*.bat`, `run_goodreads_revalidation.ps1`)
- 미사용 코드: `ikgr_core/layers.py`, `ikgr_core/graph.py`, `model_ikgr.py` 안의 KGCN/TransH/intent_aware_score
- 분석 도구 `analyze_user_segments.py`: shelf 엔트로피 등 Goodreads 메타 pack 형식에 묶여 있어 제외
