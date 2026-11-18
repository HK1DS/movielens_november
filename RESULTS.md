# 결과 요약 — MovieLens 25M, k=30 (2026-11-18 21시 기준)

- 데이터: 유저 133,684 · 아이템 15,785 · 인터랙션 24,046,865 (평점 전부 implicit).
- 공통 학습: 임베딩 64, 배치 8192, lr 1e-3, 최대 40 epoch, 2 epoch마다 검증(NDCG@10), 4번 연속 개선 없으면 종료, 평가 배치 128명. 모든 spec seed 2020·2021·2022.
- 평가: full-sort, test. 표의 값은 3 seed 평균(± 표준편차). paired 검정은 유저 단위(seed 평균 → 유저별 차이, 95% bootstrap CI, Wilcoxon).
- spec: `IKGR_kgoff` = MF, `IKGR_dyn_nointent` = MF + 메타 KG(장르) + recency, `IKGR_dyn` = + LLM intent,
  `IKGR_lgcn_*` = 같은 구성을 LightGCN 백본 위에. BPR·LightGCN은 RecBole 기준선.
- 출처: 진욱(MF 계열, intent spec) + 은선(1단계 MF 기준선 seed 일부, 2단계 `LightGCN`·`IKGR_lgcn_dyn_nointent`, 11/18 병합, 서명 일치 확인).
- 자세한 표: [`run_k30_TO/report_TO.md`](run_k30_TO/report_TO.md), [`run_k30_TO_GLOBAL/report_TO_GLOBAL.md`](run_k30_TO_GLOBAL/report_TO_GLOBAL.md),
  [`run_k30_TO_GLOBAL/report_TO_GLOBAL_warm.md`](run_k30_TO_GLOBAL/report_TO_GLOBAL_warm.md).

## TO (유저별 시간순 80/10/10, 테스트 유저 133,684명) — 완료

| spec | NDCG@10 | Recall@10 | tail Recall@10 | Coverage@10 | best epoch |
|---|---:|---:|---:|---:|---:|
| `IKGR_dyn` | **0.0893** ± 0.0002 | 0.0651 | 0.0021 | 0.410 | 22 |
| `IKGR_dyn_nointent` | 0.0892 ± 0.0004 | 0.0650 | 0.0022 | 0.413 | 23 |
| `IKGR_lgcn_dyn_nointent` | 0.0883 ± 0.0003 | 0.0634 | 0.0014 | 0.321 | 38 |
| `BPR` | 0.0877 ± 0.0005 | 0.0636 | 0.0024 | 0.457 | 33 |
| `IKGR_kgoff` (MF) | 0.0876 ± 0.0002 | 0.0636 | 0.0023 | 0.450 | 32 |
| `IKGR_lgcn_dyn` | 0.0876 ± 0.0001 | 0.0626 | 0.0009 | 0.276 | 22 |
| `LightGCN` | 0.0848 ± 0.0002 | 0.0595 | 0.0008 | 0.287 | 39 |

paired (NDCG@10, 133,684명):

| 비교 | 차이 | 95% CI | p | 해석 |
|---|---:|---|---:|---|
| `IKGR_dyn_nointent` vs `IKGR_kgoff` | +1.8% | [+0.0013, +0.0019] | 1e-08 | 메타 KG + recency 효과 |
| `IKGR_dyn` vs `IKGR_dyn_nointent` | +0.1% | [−0.0001, +0.0003] | 0.12 | intent 효과 없음 (MF 백본) |
| `IKGR_lgcn_dyn` vs `IKGR_lgcn_dyn_nointent` | −0.8% | [−0.0010, −0.0005] | 5e-09 | intent가 오히려 낮음 (LightGCN 백본) |
| `IKGR_dyn` vs `BPR` | +1.9% | [+0.0013, +0.0020] | 7e-09 | |
| `IKGR_lgcn_dyn_nointent` vs `LightGCN` | +4.2% | [+0.0032, +0.0039] | 1e-65 | LightGCN 위에서도 메타 KG 효과 |
| `IKGR_lgcn_dyn_nointent` vs `IKGR_dyn_nointent` | −1.0% | [−0.0012, −0.0006] | 3e-39 | 이 데이터에선 MF 백본이 더 좋음 |

## TO_GLOBAL (전체 시간순 70/10/20, 테스트 유저 27,854명) — `IKGR_lgcn_dyn`은 seed 2020만 (2021·2022 은선 진행 중)

테스트 유저의 88.6%(24,668명)는 학습 기록이 0이다. 이 유저들의 점수는 학습되지 않은 유저 임베딩을 모델이 어떻게 처리하느냐로
갈리므로(개인화가 아님) **비교는 학습 기록이 있는 3,186명(warm) 기준**으로 한다. 검증·조기 종료도 warm 유저 기준(11/16 변경, `eval_global_warm.py`).

| spec | NDCG@10 (warm) | tail Recall@10 (warm) | NDCG@10 (전체) | NDCG@10 (cold) | Coverage@10 (전체) | best epoch |
|---|---:|---:|---:|---:|---:|---:|
| `IKGR_dyn` | **0.2246** ± 0.0004 | 0.0008 | 0.0418 | 0.0182 | 0.742 | 14 |
| `IKGR_lgcn_dyn_nointent` | 0.2245 ± 0.0002 | 0.0004 | 0.0490 | 0.0264 | 0.510 | 34 |
| `BPR` | 0.2242 ± 0.0007 | 0.0005 | 0.0558 | 0.0341 | 0.827 | 17 |
| `IKGR_kgoff` (MF) | 0.2241 ± 0.0017 | 0.0005 | 0.0534 | 0.0314 | 0.781 | 18 |
| `IKGR_dyn_nointent` | 0.2231 ± 0.0004 | 0.0006 | 0.0426 | 0.0193 | 0.691 | 14 |
| `LightGCN` | 0.2173 ± 0.0013 | 0.0003 | 0.0978 | 0.0824 | 0.603 | 32 |
| `IKGR_lgcn_dyn` (seed 2020만) | 0.2253 | 0.0004 | 0.0474 | 0.0244 | 0.513 | 29 |

paired (NDCG@10, warm 3,186명):

| 비교 | 차이 | 95% CI | p |
|---|---:|---|---:|
| `IKGR_dyn` vs `IKGR_dyn_nointent` | +0.7% | [−0.0001, +0.0031] | 0.047 |
| `IKGR_dyn` vs `BPR` | +0.2% | [−0.0024, +0.0032] | 0.33 |
| `IKGR_dyn_nointent` vs `IKGR_kgoff` | −0.4% | [−0.0037, +0.0018] | 0.92 |
| `IKGR_dyn` vs `LightGCN` | +3.4% | [+0.0041, +0.0105] | 9e-08 |
| `IKGR_lgcn_dyn_nointent` vs `LightGCN` | +3.3% | [+0.0042, +0.0102] | 2e-08 |

`IKGR_lgcn_dyn`은 seed 2020 하나뿐이라 같은 seed끼리만 비교(warm, `--seeds 2020`): vs `IKGR_lgcn_dyn_nointent` +0.3%
(CI [−0.0018, +0.0029], p=0.79), vs `IKGR_dyn` +0.1% (p=0.69), vs `BPR` +0.8% (p=0.074), vs `LightGCN` +4.0% (p=8e-07).
검증 점수는 11/17 PC가 멈춘 같은 seed 실행과 epoch마다 소수점 넷째 자리까지 같았다(재현 확인).

## 지금까지 읽히는 것

1. **intent(LLM) 효과는 확인되지 않는다.** TO에서 MF 백본 +0.1%(유의하지 않음), LightGCN 백본 −0.8%(유의하게 낮음).
   TO_GLOBAL warm에서 MF +0.7%(p=0.047, CI가 0에 걸침), LightGCN +0.3%(seed 2020, p=0.79). Goodreads 결과(no-intent 대비 우위 미확인)와 같은 방향.
2. **이득은 메타 KG + recency 쪽에서 나온다(TO).** MF +1.8%, LightGCN +4.2%. TO_GLOBAL warm에서는 MF 기준 차이가 없다(−0.4%, p=0.92).
3. **이 데이터에서는 LightGCN 백본이 MF보다 낮다.** TO `LightGCN` 0.0848 < `BPR` 0.0877, `IKGR_lgcn_*` < `IKGR_dyn*`. 학습도 5~10배 오래 걸린다.
4. **롱테일 노출:** TO에서 KG/intent 모델의 tail Recall·Coverage는 BPR보다 낮다(0.0021 vs 0.0024). TO_GLOBAL warm의 tail Recall은
   `IKGR_dyn`이 가장 높지만(0.0008 vs BPR 0.0005) 절대값이 매우 작다.
5. **TO_GLOBAL 전체 유저 점수는 cold 유저가 좌우한다.** `LightGCN`이 전체 0.0978로 가장 높은 이유는 cold 유저 0.0824 때문이고,
   warm 유저에서는 가장 낮다(0.2173). cold 유저에겐 train-only 프로필이 없어서 intent도 쓸 수 없다 → 콜드 스타트는 별도 처리(인기도 등)가 필요하다는 한계로 적는다.

학습 시간은 GPU가 섞여 있어(진욱 RTX 3060 Ti, 은선 RTX 2070 SUPER / 임대 RTX 3090) 표에서 뺐다. 3060 Ti 기준: MF 계열 30~40분(TO), LightGCN 계열 4.6~6.6시간(TO).
