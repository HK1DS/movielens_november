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
step1.py, step2.py             B, C  LLM intent 추출 / RAG 확장 (캐시·재개·Ctrl+C 안전)
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
| `recent_label`, `profile_facets` | P | 유저 프로필 형식: `"<recent_label>: t1; t2 … | <facet.label>: v1, v2 …"` |
| `meta_relations` | D, E | 메타 KG 노드 타입 (relation별 `max_per_item`, `min_df`) |
| `ts_clip` | T | timestamp clamp 창 (`dataset.ts_clip`으로 덮어쓰기 가능) |
| `source_files()` | D | 내용이 바뀌면 KG pack을 다시 만들 원본 파일 목록 |

**어댑터 누수 규칙:** 아이템 메타데이터는 정적 카탈로그 정보여야 한다. 유저 활동을 전 기간에 걸쳐 모은 신호
(유저가 단 태그, 리뷰, 인기 shelf 카운트 등)는 검증/테스트 기간 행동을 포함한다. 쓰려면 train 구간으로
다시 만들거나, 최소한 그 결정을 기록해 둔다. 누수 가드는 **유저 프로필** 쪽만 검사하고 아이템 메타데이터는 검사하지 않는다.

### MovieLens TODO (`data_adapters/movielens.py`)

버전은 **ml-25m 고정**(평점 25,000,095 · 유저 162,541 · 영화 62,423). 이전 레포(`hk1ds_jwkim_movielens`의
`prepare_movielens.py`)가 같은 데이터로 한 선택은 아래와 같고, 재현 확인용 기준값으로 쓰면 된다
(여기서 아직 확정한 건 아님):

| 항목 | 이전 레포 |
|---|---|
| positive | 모든 평점을 implicit interaction으로 유지 (threshold 없음) |
| `(no genres listed)` 영화 | 평점째 제외 → 평점 24,973,468 · 유저 162,541 · 영화 54,479 |
| 아이템 텍스트 / 메타 | `"title \| genres"` / 장르만 메타 relation(`categories`) |
| k=30 core | 유저 133,684 · 아이템 15,785 · 상호작용 24,046,865 |
| 유저 프로필 | 유저별 이력 앞 80% (이 레포는 정확한 train-only 프로필로 대체 → 프로필 텍스트는 달라짐) |

- [ ] 규모: k=30 전체면 유저 약 13만. B/C 단계는 고유 프로필당 LLM 호출 1회씩(이전 레포 TO_GLOBAL: 프로필 약 11.3만 개,
      토큰 약 1.88억)이고 LightGCN은 배치마다 그래프 전체를 전파한다. `--steps 0ATPS`로 비용부터 보고, 필요하면 k를 올리거나 유저 샘플링.
- [ ] positive 정의: 모든 평점을 implicit positive로 볼지(이전 레포·Goodreads 방식), rating ≥ 4만 볼지 → `dataset.min_rating`.
- [ ] `(no genres listed)` 영화: 이전 레포처럼 뺄지, 장르 노드 없이 남길지.
- [ ] 아이템 텍스트: `movies.csv`엔 제목+장르뿐이라 LLM 아이템 intent가 장르를 되풀이할 가능성이 크다
      (= 메타 KG의 genre 채널과 중복). intent 채널을 의미 있게 하려면 외부 정적 텍스트(예: `links.csv` → TMDB 줄거리)가 필요.
- [ ] 태그: `tags.csv`와 tag genome은 전 기간 집계라 위 누수 규칙에 걸린다. 빼거나 train 구간으로 재구성.
- [ ] timestamp: MovieLens는 "평점 입력 시각"이라 가입 직후 몰아서 평가한 구간이 많다. 동점은 split 재현이 처리하지만,
      TO의 "최근 소비" 의미가 Goodreads(읽은 날짜)와 다르다는 점은 해석할 때 감안.
- [ ] `prompts/step1_intents.txt` 문구를 영화 도메인에 맞게.

## 5. 실행 (Windows, 기존 GPU 환경)

```powershell
powershell -ExecutionPolicy Bypass -File scripts\setup_venv.ps1     # 기존 %USERPROFILE%\.venvs\goodreads-ikgr 재사용
copy .env.example .env                                              # LUXIA_API_KEY 입력
```

```bash
python run_pipeline.py --k 30 --split TO --steps 0ATPS        # 무료 준비 + LLM 비용 추정
python run_pipeline.py --k 30 --split TO --steps BCDER        # LLM(확인 프롬프트) → KG → 학습/평가 → 리포트
python run_pipeline.py --k 30 --split TO --steps ER --seeds 2021,2022 --specs IKGR_lgcn_dyn,LightGCN
python paired_compare.py --run_dir run_k30_TO --split TO --a IKGR_lgcn_dyn --b LightGCN IKGR_lgcn_dyn_nointent --by_activity
```

중단/재개는 Goodreads 레포와 같다: Ctrl+C 후 같은 명령을 다시 실행하면 0/A/T/P는 산출물이 있으면 건너뛰고,
B/C는 LLM 캐시에 없는 프로필만 호출하고, B/C/D는 입력 해시가 같으면 통째로 건너뛰고(`--redo D` 등으로 강제),
E는 끝난 spec×seed를 유지한다. `eval_slices.py`·`model_ikgr.py`·`corona_retriever.py`나 설정을 바꾸면
끝난 결과도 다시 학습한다(의도된 안전장치).

## 6. 테스트

```bash
python -m pytest -q tests/                         # split 재현(실제 RecBole 대조), train-only 프로필, LightGCN 동치, 메타 KG, rerank
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
