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
  `data_adapters/movielens.py`는 아직 TODO.
