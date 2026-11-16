# Paired comparisons (TO, test)

Unit = user (metric averaged over seeds first). CI = 95% bootstrap over users; p = Wilcoxon signed-rank; W/T/L = % users where A wins/ties/loses.

| A vs B | metric | users | A | B | diff (rel) | 95% CI | p | W/T/L % |
|---|---|---:|---:|---:|---:|---|---:|---|
| IKGR_dyn vs BPR | ndcg@10 | 133,684 | 0.0893 | 0.0877 | +0.0016 (+1.9%) | [+0.0013, +0.0020] | 7.1e-09 | 30/41/29 |
| IKGR_dyn vs BPR | tail_recall@10 | 69,683 | 0.0021 | 0.0024 | -0.0002 (-10.3%) | [-0.0004, -0.0001] | 0.016 | 1/97/2 |
| IKGR_dyn vs IKGR_kgoff | ndcg@10 | 133,684 | 0.0893 | 0.0876 | +0.0017 (+2.0%) | [+0.0014, +0.0021] | 2.2e-14 | 30/42/28 |
| IKGR_dyn vs IKGR_kgoff | tail_recall@10 | 69,683 | 0.0021 | 0.0023 | -0.0002 (-7.9%) | [-0.0004, -0.0000] | 0.036 | 1/97/2 |
| IKGR_dyn vs IKGR_dyn_nointent | ndcg@10 | 133,684 | 0.0893 | 0.0892 | +0.0001 (+0.1%) | [-0.0001, +0.0003] | 0.12 | 28/45/27 |
| IKGR_dyn vs IKGR_dyn_nointent | tail_recall@10 | 69,683 | 0.0021 | 0.0022 | -0.0001 (-5.4%) | [-0.0003, +0.0000] | 0.62 | 1/98/1 |

**IKGR_dyn vs BPR — ndcg@10 by train-activity quintile**

| bucket | activity ≤ | users | diff | win % | loss % |
|---|---:|---:|---:|---:|---:|
| Q1 | 38 | 27,673 | +0.0015 | 20 | 20 |
| Q2 | 58 | 26,292 | +0.0015 | 24 | 23 |
| Q3 | 97 | 26,653 | +0.0023 | 29 | 27 |
| Q4 | 192 | 26,455 | +0.0019 | 35 | 33 |
| Q5 | 9852 | 26,611 | +0.0010 | 41 | 41 |

**IKGR_dyn vs BPR — tail_recall@10 by train-activity quintile**

| bucket | activity ≤ | users | diff | win % | loss % |
|---|---:|---:|---:|---:|---:|
| Q1 | 56 | 14,064 | -0.0002 | 1 | 1 |
| Q2 | 99 | 13,980 | -0.0003 | 1 | 1 |
| Q3 | 168 | 13,968 | -0.0001 | 1 | 1 |
| Q4 | 305 | 13,772 | -0.0003 | 1 | 2 |
| Q5 | 9852 | 13,899 | -0.0003 | 3 | 4 |

**IKGR_dyn vs IKGR_kgoff — ndcg@10 by train-activity quintile**

| bucket | activity ≤ | users | diff | win % | loss % |
|---|---:|---:|---:|---:|---:|
| Q1 | 38 | 27,673 | +0.0013 | 20 | 20 |
| Q2 | 58 | 26,292 | +0.0016 | 24 | 23 |
| Q3 | 97 | 26,653 | +0.0025 | 29 | 27 |
| Q4 | 192 | 26,455 | +0.0019 | 35 | 32 |
| Q5 | 9852 | 26,611 | +0.0013 | 41 | 40 |

**IKGR_dyn vs IKGR_kgoff — tail_recall@10 by train-activity quintile**

| bucket | activity ≤ | users | diff | win % | loss % |
|---|---:|---:|---:|---:|---:|
| Q1 | 56 | 14,064 | -0.0003 | 1 | 1 |
| Q2 | 99 | 13,980 | -0.0000 | 1 | 1 |
| Q3 | 168 | 13,968 | -0.0001 | 1 | 1 |
| Q4 | 305 | 13,772 | -0.0003 | 1 | 1 |
| Q5 | 9852 | 13,899 | -0.0002 | 3 | 4 |

**IKGR_dyn vs IKGR_dyn_nointent — ndcg@10 by train-activity quintile**

| bucket | activity ≤ | users | diff | win % | loss % |
|---|---:|---:|---:|---:|---:|
| Q1 | 38 | 27,673 | -0.0003 | 18 | 18 |
| Q2 | 58 | 26,292 | +0.0003 | 22 | 21 |
| Q3 | 97 | 26,653 | +0.0004 | 26 | 25 |
| Q4 | 192 | 26,455 | +0.0003 | 33 | 32 |
| Q5 | 9852 | 26,611 | -0.0002 | 40 | 40 |

**IKGR_dyn vs IKGR_dyn_nointent — tail_recall@10 by train-activity quintile**

| bucket | activity ≤ | users | diff | win % | loss % |
|---|---:|---:|---:|---:|---:|
| Q1 | 56 | 14,064 | -0.0001 | 1 | 1 |
| Q2 | 99 | 13,980 | -0.0002 | 1 | 1 |
| Q3 | 168 | 13,968 | -0.0001 | 1 | 1 |
| Q4 | 305 | 13,772 | -0.0002 | 1 | 1 |
| Q5 | 9852 | 13,899 | -0.0000 | 2 | 3 |

# Accuracy / exposure Pareto (TO, seed means)

★ = not dominated on (NDCG@10, tail Recall@10); ◆ = not dominated on (NDCG@10, Coverage@10).

| result | seeds | NDCG@10 | ±sd | tail R@10 | Cov@10 | best epoch | frontier |
|---|---:|---:|---:|---:|---:|---:|---|
| IKGR_dyn | 3 | 0.0893 | 0.0002 | 0.0021 | 0.4103 | 22 | ★◆ |
| IKGR_lgcn_dyn | 2 | 0.0875 | 0.0000 | 0.0008 | 0.2659 | 19 |  |
| BPR | 3 | 0.0871 | 0.0000 | 0.0023 | 0.4310 | 29 | ★◆ |
| IKGR_lgcn_dyn_pop0p25 | 2 | 0.0834 | 0.0002 | 0.0022 | 0.3808 | 19 |  |
| IKGR_dyn_pop0p25 | 3 | 0.0789 | 0.0002 | 0.0092 | 0.6775 | 22 | ★◆ |
| IKGR_lgcn_dyn_pop0p5 | 2 | 0.0756 | 0.0003 | 0.0067 | 0.5494 | 19 |  |
| BPR_pop0p25 | 3 | 0.0715 | 0.0000 | 0.0130 | 0.7786 | 29 | ★◆ |
| IKGR_dyn_pop0p5 | 3 | 0.0580 | 0.0006 | 0.0246 | 0.9052 | 22 | ★◆ |
| BPR_pop0p5 | 3 | 0.0425 | 0.0000 | 0.0295 | 0.9577 | 29 | ★◆ |
| IKGR_lgcn_dyn_pop1p0 | 2 | 0.0344 | 0.0005 | 0.0157 | 0.8907 | 19 |  |
| IKGR_dyn_pop1p0 | 3 | 0.0039 | 0.0001 | 0.0118 | 0.7616 | 22 |  |
| BPR_pop1p0 | 3 | 0.0028 | 0.0000 | 0.0085 | 0.6178 | 29 |  |
