# Paired comparisons (TO_GLOBAL, test)

Unit = user (metric averaged over seeds first). CI = 95% bootstrap over users; p = Wilcoxon signed-rank; W/T/L = % users where A wins/ties/loses.

| A vs B | metric | users | A | B | diff (rel) | 95% CI | p | W/T/L % |
|---|---|---:|---:|---:|---:|---|---:|---|
| IKGR_dyn vs IKGR_dyn_nointent | ndcg@10 | 27,854 | 0.0418 | 0.0426 | -0.0008 (-1.8%) | [-0.0014, -0.0001] | 0.23 | 23/53/23 |
| IKGR_dyn vs IKGR_dyn_nointent | tail_recall@10 | 27,269 | 0.0010 | 0.0007 | +0.0003 (+45.8%) | [+0.0002, +0.0004] | 7.8e-10 | 8/87/6 |

**IKGR_dyn vs IKGR_dyn_nointent — ndcg@10 by train-activity quintile**

| bucket | activity ≤ | users | diff | win % | loss % |
|---|---:|---:|---:|---:|---:|
| Q1 | 0 | 24,668 | -0.0011 | 21 | 22 |
| Q5 | 5371 | 3,186 | +0.0015 | 37 | 34 |

**IKGR_dyn vs IKGR_dyn_nointent — tail_recall@10 by train-activity quintile**

| bucket | activity ≤ | users | diff | win % | loss % |
|---|---:|---:|---:|---:|---:|
| Q1 | 0 | 24,210 | +0.0003 | 8 | 6 |
| Q5 | 5371 | 3,059 | +0.0002 | 4 | 2 |

# Accuracy / exposure Pareto (TO_GLOBAL, seed means)

★ = not dominated on (NDCG@10, tail Recall@10); ◆ = not dominated on (NDCG@10, Coverage@10).

| result | seeds | NDCG@10 | ±sd | tail R@10 | Cov@10 | best epoch | frontier |
|---|---:|---:|---:|---:|---:|---:|---|
| IKGR_dyn_nointent | 3 | 0.0426 | 0.0018 | 0.0007 | 0.6913 | 14 | ★◆ |
| IKGR_dyn | 3 | 0.0418 | 0.0022 | 0.0010 | 0.7423 | 14 | ★◆ |
| IKGR_dyn_pop0p25 | 3 | 0.0327 | 0.0021 | 0.0011 | 0.7805 | 14 | ★◆ |
| IKGR_dyn_nointent_pop0p25 | 3 | 0.0327 | 0.0019 | 0.0010 | 0.7292 | 14 |  |
| IKGR_dyn_pop0p5 | 3 | 0.0229 | 0.0026 | 0.0015 | 0.7999 | 14 | ★◆ |
| IKGR_dyn_nointent_pop0p5 | 3 | 0.0221 | 0.0025 | 0.0013 | 0.7526 | 14 |  |
| IKGR_dyn_pop1p0 | 3 | 0.0063 | 0.0011 | 0.0012 | 0.5699 | 14 |  |
| IKGR_dyn_nointent_pop1p0 | 3 | 0.0057 | 0.0011 | 0.0012 | 0.5220 | 14 |  |
