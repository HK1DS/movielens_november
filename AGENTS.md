# AGENTS.md — 작업 규칙

구조와 실행법은 README.md. 이 파일은 코드를 고칠 때 깨면 안 되는 것만 적는다.

## 불변 조건
1. **평가 기준 파일은 하나:** `data/kc_k{K}/ikgr-custom/ikgr-custom.inter`. 프로필·KG·평가가 모두 이 파일에서 나온다.
2. **유저 프로필은 train-only:** `build_trainonly_profiles.py`는 `ikgr_core/temporal_split.py`가 재현한 RecBole split의
   profile-safe train 행만 쓴다. split 규칙(`SPLITS`)을 바꾸면 `eval_slices._config`의 `eval_args`도 같이 바꾸고
   `tests/test_temporal_split.py`가 통과해야 한다.
3. **누수 가드를 끄지 않는다:** `eval_slices._check_profile_leakage`(provenance ⊆ RecBole train, KG 유저 ⊆ 프로필 유저, split 일치).
4. **split별 산출물 분리:** `run_k{K}_{SPLIT}/`. 다른 split의 프로필/intent/KG를 섞지 않는다.
5. **공통 학습 프로토콜:** 모든 spec(BPR/LightGCN 포함)이 같은 epochs/eval_step/stopping_step/valid_metric/lr/reg/embedding_size.
6. **데이터셋 종속 코드는 `data_adapters/`에만.** 다른 파일에서 raw 파일을 직접 읽지 않는다.
7. **아이템 메타데이터는 정적이어야 한다.** 전 기간 유저 활동 집계(태그, 리뷰 등)를 쓰면 README §4의 결정을 기록한다.
8. LightGCN shim(`eval_slices.py` 상단 `dok_matrix._update`)은 SciPy ≥ 1.13에서 인접행렬이 비는 버그 수정이다. 지우지 않는다
   (`tests/test_lgcn_backbone.py`).

## 변경 후 확인
- `python -m pytest -q tests/`
- 파이프라인 단계나 어댑터 계약을 건드렸으면 `python tests/smoke_e2e.py --workdir <tmp>`

## 진행 기록
- 2026-11-10: Goodreads 레포(`졸작_11월/Goodreads_experiment-main`)에서 아키텍처 뼈대 이식.
  데이터셋 버전은 **MovieLens 25M (`ml-25m`)** 으로 고정 — 이전 MovieLens 레포(HK1DS/hk1ds_jwkim_movielens)와 동일.
  (어댑터는 같은 날 아래에서 구현)
- 2026-11-10: Luxia API 키 여러 개 + 소진 시 자동 교체(`ikgr_core/key_pool.py`). 전체 소진이면 step1/2가 exit 4로 중단 후 재실행 시 이어서.
  Luxia의 실제 소진 응답 형식은 아직 미확인 — 첫 소진 때 `[llm] luxia HTTP ...` 로그를 보고 필요하면 패턴 보강.
- 2026-11-10: MovieLens 어댑터 구현 — 이전 레포와 같은 선택(평점 전부 implicit, `(no genres listed)` 제외,
  item text `title | genres`, genre만 메타/facet, 태그 미사용). Luxia 공유 페이싱(`ikgr_core/rate_limit.py`, 이전 레포에서 이식;
  시작은 무제한, 첫 429에 1초 간격), 캐시 저장 500회 간격, `pilot_llm.py` + `scripts\intent_pilot.bat` / `intent_full.bat`.
- 2026-11-10: 데이터 규모를 이전 MovieLens 레포 최종 패키지와 동일하게 고정(ml-25m, k=30: 133,684 / 15,785 / 24,046,865).
  단계 0·A 크기를 어댑터 `REFERENCE`와 대조(`scale_check.json`), 불일치 시 B/C 차단. `scripts\intent_prep.bat` = 무료 준비만(TO+TO_GLOBAL).
- 2026-11-10: `--max_hours` 시간 제한(exit 5 일시정지/재개), 분 단위 추세 로그 `llm_progress_step{1,2}.csv`, 페이싱 완화(AIMD),
  아이템 우선 처리, Windows 절전 방지. `scripts\intent_run_7h.bat`로 TO step1 7시간 실행 시작(추세 보고 전체 소요시간 판단).
- 2026-11-10: MovieLens 유저 프로필 형식 변경 — "최근 10편 + 최근 50편 장르" → **train 이력 전체에서 평점 상위 15편(동점 최신순) +
  이력 전체 장르 분포** (bulk-rating 때문에 최근 목록이 무의미). 어댑터 계약에 `title_order`/`title_n`/`profile_window` 추가
  (기본값은 이전 동작 그대로). 아이템 호출이 끝나기 전에 바꿔 버려진 유저 호출 없음. 외부 일시정지용 STOP 파일(`scripts\intent_stop.bat`).
- 2026-11-10: 실측 결과 Luxia는 토큰이 아니라 요청 수 제한(약 4,000회/시간, 키 1개). **키 병렬 호출**로 변경 —
  `ikgr_core/key_lanes.py`: 키마다 요청 간격·동시 요청 상한(기본 8)을 따로 두고 모든 키를 동시에 호출, 소진된 키만 빠짐.
  `IKGR_LLM_KEY_MODE=sequential`이면 이전 동작. 1시간 실측(키 3개): 5,443회 = 시간당 약 5,400회, 키 1개(약 4,030회) 대비 +34%.
  3배가 아니라 제한이 순수 키 단위는 아님 → 기본값은 parallel 유지, 남은 B/C 예상 약 3.7일.
- 2026-11-11: 장시간 실측(6h 26,047회, 5h 22,482회)은 시간당 약 4,400회로 키 1개와 거의 같음 → 제한은 계정·서비스 공용.
  parallel은 손해가 없어 그대로 둠. 실행 스크립트 `scripts\intent_run_{1,5,6,7,9}h.bat`(9h는 GPU를 숨기고 API만).
