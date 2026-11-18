# Paired comparisons (TO_GLOBAL, test)

Unit = user (metric averaged over seeds first). CI = 95% bootstrap over users; p = Wilcoxon signed-rank; W/T/L = % users where A wins/ties/loses.

| A vs B | metric | users | A | B | diff (rel) | 95% CI | p | W/T/L % |
|---|---|---:|---:|---:|---:|---|---:|---|
| IKGR_dyn vs BPR | ndcg@10 | 27,854 | 0.0418 | 0.0558 | -0.0140 (-25.1%) | [-0.0149, -0.0132] | 1.4e-164 | 21/47/32 |
| IKGR_dyn vs BPR | tail_recall@10 | 27,269 | 0.0010 | 0.0006 | +0.0003 (+52.5%) | [+0.0002, +0.0004] | 1.1e-06 | 8/86/6 |
| IKGR_dyn vs LightGCN | ndcg@10 | 27,854 | 0.0418 | 0.0978 | -0.0560 (-57.3%) | [-0.0573, -0.0547] | 0 | 15/31/54 |
| IKGR_dyn vs LightGCN | tail_recall@10 | 27,269 | 0.0010 | 0.0006 | +0.0004 (+71.0%) | [+0.0003, +0.0005] | 1.3e-28 | 8/87/5 |
| IKGR_dyn vs IKGR_kgoff | ndcg@10 | 27,854 | 0.0418 | 0.0534 | -0.0116 (-21.8%) | [-0.0125, -0.0108] | 9.7e-120 | 21/48/31 |
| IKGR_dyn vs IKGR_kgoff | tail_recall@10 | 27,269 | 0.0010 | 0.0006 | +0.0004 (+70.1%) | [+0.0003, +0.0005] | 7.5e-09 | 8/86/6 |
| IKGR_dyn vs IKGR_dyn_nointent | ndcg@10 | 27,854 | 0.0418 | 0.0426 | -0.0008 (-1.8%) | [-0.0014, -0.0001] | 0.23 | 23/53/23 |
| IKGR_dyn vs IKGR_dyn_nointent | tail_recall@10 | 27,269 | 0.0010 | 0.0007 | +0.0003 (+45.8%) | [+0.0002, +0.0004] | 7.8e-10 | 8/87/6 |
| IKGR_lgcn_dyn_nointent vs LightGCN | ndcg@10 | 27,854 | 0.0490 | 0.0978 | -0.0488 (-49.9%) | [-0.0501, -0.0475] | 0 | 19/29/52 |
| IKGR_lgcn_dyn_nointent vs LightGCN | tail_recall@10 | 27,269 | 0.0008 | 0.0006 | +0.0003 (+43.9%) | [+0.0002, +0.0003] | 9.1e-12 | 7/88/5 |
| IKGR_lgcn_dyn vs IKGR_lgcn_dyn_nointent | ndcg@10 | 27,854 | 0.0477 | 0.0490 | -0.0013 (-2.7%) | [-0.0021, -0.0006] | 0.004 | 26/47/27 |
| IKGR_lgcn_dyn vs IKGR_lgcn_dyn_nointent | tail_recall@10 | 27,269 | 0.0009 | 0.0008 | +0.0001 (+13.0%) | [-0.0000, +0.0002] | 0.093 | 7/87/6 |
| IKGR_lgcn_dyn_nointent vs IKGR_dyn_nointent | ndcg@10 | 27,854 | 0.0490 | 0.0426 | +0.0064 (+15.1%) | [+0.0058, +0.0070] | 5.8e-92 | 28/53/19 |
| IKGR_lgcn_dyn_nointent vs IKGR_dyn_nointent | tail_recall@10 | 27,269 | 0.0008 | 0.0007 | +0.0002 (+22.7%) | [+0.0001, +0.0002] | 0.11 | 6/89/5 |

**IKGR_dyn vs BPR — ndcg@10 by train-activity quintile**

| bucket | activity ≤ | users | diff | win % | loss % |
|---|---:|---:|---:|---:|---:|
| Q1 | 0 | 24,668 | -0.0159 | 19 | 32 |
| Q5 | 5371 | 3,186 | +0.0004 | 37 | 35 |

**IKGR_dyn vs BPR — tail_recall@10 by train-activity quintile**

| bucket | activity ≤ | users | diff | win % | loss % |
|---|---:|---:|---:|---:|---:|
| Q1 | 0 | 24,210 | +0.0003 | 8 | 7 |
| Q5 | 5371 | 3,059 | +0.0003 | 5 | 3 |

**IKGR_dyn vs LightGCN — ndcg@10 by train-activity quintile**

| bucket | activity ≤ | users | diff | win % | loss % |
|---|---:|---:|---:|---:|---:|
| Q1 | 0 | 24,668 | -0.0642 | 12 | 57 |
| Q5 | 5371 | 3,186 | +0.0074 | 40 | 32 |

**IKGR_dyn vs LightGCN — tail_recall@10 by train-activity quintile**

| bucket | activity ≤ | users | diff | win % | loss % |
|---|---:|---:|---:|---:|---:|
| Q1 | 0 | 24,210 | +0.0004 | 9 | 5 |
| Q5 | 5371 | 3,059 | +0.0005 | 7 | 1 |

**IKGR_dyn vs IKGR_kgoff — ndcg@10 by train-activity quintile**

| bucket | activity ≤ | users | diff | win % | loss % |
|---|---:|---:|---:|---:|---:|
| Q1 | 0 | 24,668 | -0.0132 | 19 | 30 |
| Q5 | 5371 | 3,186 | +0.0005 | 37 | 36 |

**IKGR_dyn vs IKGR_kgoff — tail_recall@10 by train-activity quintile**

| bucket | activity ≤ | users | diff | win % | loss % |
|---|---:|---:|---:|---:|---:|
| Q1 | 0 | 24,210 | +0.0004 | 8 | 6 |
| Q5 | 5371 | 3,059 | +0.0003 | 5 | 3 |

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

**IKGR_lgcn_dyn_nointent vs LightGCN — ndcg@10 by train-activity quintile**

| bucket | activity ≤ | users | diff | win % | loss % |
|---|---:|---:|---:|---:|---:|
| Q1 | 0 | 24,668 | -0.0560 | 16 | 55 |
| Q5 | 5371 | 3,186 | +0.0072 | 38 | 30 |

**IKGR_lgcn_dyn_nointent vs LightGCN — tail_recall@10 by train-activity quintile**

| bucket | activity ≤ | users | diff | win % | loss % |
|---|---:|---:|---:|---:|---:|
| Q1 | 0 | 24,210 | +0.0003 | 8 | 5 |
| Q5 | 5371 | 3,059 | +0.0001 | 3 | 1 |

**IKGR_lgcn_dyn vs IKGR_lgcn_dyn_nointent — ndcg@10 by train-activity quintile**

| bucket | activity ≤ | users | diff | win % | loss % |
|---|---:|---:|---:|---:|---:|
| Q1 | 0 | 24,668 | -0.0017 | 25 | 27 |
| Q5 | 5371 | 3,186 | +0.0012 | 33 | 32 |

**IKGR_lgcn_dyn vs IKGR_lgcn_dyn_nointent — tail_recall@10 by train-activity quintile**

| bucket | activity ≤ | users | diff | win % | loss % |
|---|---:|---:|---:|---:|---:|
| Q1 | 0 | 24,210 | +0.0001 | 8 | 7 |
| Q5 | 5371 | 3,059 | +0.0000 | 2 | 1 |

**IKGR_lgcn_dyn_nointent vs IKGR_dyn_nointent — ndcg@10 by train-activity quintile**

| bucket | activity ≤ | users | diff | win % | loss % |
|---|---:|---:|---:|---:|---:|
| Q1 | 0 | 24,668 | +0.0071 | 26 | 17 |
| Q5 | 5371 | 3,186 | +0.0014 | 37 | 35 |

**IKGR_lgcn_dyn_nointent vs IKGR_dyn_nointent — tail_recall@10 by train-activity quintile**

| bucket | activity ≤ | users | diff | win % | loss % |
|---|---:|---:|---:|---:|---:|
| Q1 | 0 | 24,210 | +0.0002 | 6 | 5 |
| Q5 | 5371 | 3,059 | -0.0002 | 2 | 5 |

# Accuracy / exposure Pareto (TO_GLOBAL, seed means)

★ = not dominated on (NDCG@10, tail Recall@10); ◆ = not dominated on (NDCG@10, Coverage@10).

| result | seeds | NDCG@10 | ±sd | tail R@10 | Cov@10 | best epoch | frontier |
|---|---:|---:|---:|---:|---:|---:|---|
| BPR | 3 | 0.0558 | 0.0030 | 0.0006 | 0.8272 | 17 | ★◆ |
| IKGR_kgoff | 3 | 0.0534 | 0.0011 | 0.0006 | 0.7813 | 18 |  |
| IKGR_lgcn_dyn | 3 | 0.0474 | 0.0000 | 0.0009 | 0.5128 | 29 | ★ |
| IKGR_dyn_nointent | 3 | 0.0426 | 0.0018 | 0.0007 | 0.6913 | 14 |  |
| IKGR_dyn | 3 | 0.0418 | 0.0022 | 0.0010 | 0.7423 | 14 | ★ |
| BPR_pop0p25 | 3 | 0.0401 | 0.0027 | 0.0008 | 0.8727 | 17 | ◆ |
| IKGR_kgoff_pop0p25 | 3 | 0.0391 | 0.0008 | 0.0007 | 0.8244 | 18 |  |
| IKGR_lgcn_dyn_pop0p25 | 3 | 0.0379 | 0.0000 | 0.0012 | 0.5723 | 29 | ★ |
| IKGR_dyn_nointent_pop0p25 | 3 | 0.0327 | 0.0019 | 0.0010 | 0.7292 | 14 |  |
| IKGR_dyn_pop0p25 | 3 | 0.0327 | 0.0021 | 0.0011 | 0.7805 | 14 |  |
| IKGR_lgcn_dyn_pop0p5 | 3 | 0.0304 | 0.0000 | 0.0015 | 0.6185 | 29 | ★ |
| IKGR_kgoff_pop0p5 | 3 | 0.0240 | 0.0003 | 0.0010 | 0.8582 | 18 |  |
| IKGR_dyn_pop0p5 | 3 | 0.0229 | 0.0026 | 0.0015 | 0.7999 | 14 |  |
| BPR_pop0p5 | 3 | 0.0222 | 0.0026 | 0.0011 | 0.9044 | 17 | ◆ |
| IKGR_dyn_nointent_pop0p5 | 3 | 0.0221 | 0.0025 | 0.0013 | 0.7526 | 14 |  |
| IKGR_lgcn_dyn_pop1p0 | 3 | 0.0116 | 0.0000 | 0.0018 | 0.5263 | 29 | ★ |
| IKGR_dyn_pop1p0 | 3 | 0.0063 | 0.0011 | 0.0012 | 0.5699 | 14 |  |
| IKGR_dyn_nointent_pop1p0 | 3 | 0.0057 | 0.0011 | 0.0012 | 0.5220 | 14 |  |
| BPR_pop1p0 | 3 | 0.0039 | 0.0002 | 0.0006 | 0.6855 | 17 |  |
| IKGR_kgoff_pop1p0 | 3 | 0.0036 | 0.0000 | 0.0005 | 0.6742 | 18 |  |
