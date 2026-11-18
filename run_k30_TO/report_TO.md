# Paired comparisons (TO, test)

Unit = user (metric averaged over seeds first). CI = 95% bootstrap over users; p = Wilcoxon signed-rank; W/T/L = % users where A wins/ties/loses.

| A vs B | metric | users | A | B | diff (rel) | 95% CI | p | W/T/L % |
|---|---|---:|---:|---:|---:|---|---:|---|
| IKGR_dyn vs BPR | ndcg@10 | 133,684 | 0.0893 | 0.0877 | +0.0016 (+1.9%) | [+0.0013, +0.0020] | 7.1e-09 | 30/41/29 |
| IKGR_dyn vs BPR | tail_recall@10 | 69,683 | 0.0021 | 0.0024 | -0.0002 (-10.3%) | [-0.0004, -0.0001] | 0.016 | 1/97/2 |
| IKGR_dyn vs LightGCN | ndcg@10 | 133,684 | 0.0893 | 0.0848 | +0.0046 (+5.4%) | [+0.0041, +0.0050] | 6.3e-183 | 31/44/25 |
| IKGR_dyn vs LightGCN | tail_recall@10 | 69,683 | 0.0021 | 0.0008 | +0.0013 (+170.8%) | [+0.0012, +0.0015] | 1e-15 | 2/98/0 |
| IKGR_dyn vs IKGR_kgoff | ndcg@10 | 133,684 | 0.0893 | 0.0876 | +0.0017 (+2.0%) | [+0.0014, +0.0021] | 2.2e-14 | 30/42/28 |
| IKGR_dyn vs IKGR_kgoff | tail_recall@10 | 69,683 | 0.0021 | 0.0023 | -0.0002 (-7.9%) | [-0.0004, -0.0000] | 0.036 | 1/97/2 |
| IKGR_dyn vs IKGR_dyn_nointent | ndcg@10 | 133,684 | 0.0893 | 0.0892 | +0.0001 (+0.1%) | [-0.0001, +0.0003] | 0.12 | 28/45/27 |
| IKGR_dyn vs IKGR_dyn_nointent | tail_recall@10 | 69,683 | 0.0021 | 0.0022 | -0.0001 (-5.4%) | [-0.0003, +0.0000] | 0.62 | 1/98/1 |
| IKGR_lgcn_dyn_nointent vs LightGCN | ndcg@10 | 133,684 | 0.0883 | 0.0848 | +0.0035 (+4.2%) | [+0.0032, +0.0039] | 1.3e-65 | 27/48/24 |
| IKGR_lgcn_dyn_nointent vs LightGCN | tail_recall@10 | 69,683 | 0.0014 | 0.0008 | +0.0006 (+74.5%) | [+0.0004, +0.0007] | 0.049 | 1/99/0 |
| IKGR_lgcn_dyn vs IKGR_lgcn_dyn_nointent | ndcg@10 | 133,684 | 0.0876 | 0.0883 | -0.0007 (-0.8%) | [-0.0010, -0.0005] | 5.1e-09 | 25/50/26 |
| IKGR_lgcn_dyn vs IKGR_lgcn_dyn_nointent | tail_recall@10 | 69,683 | 0.0009 | 0.0014 | -0.0005 (-36.9%) | [-0.0006, -0.0004] | 0.026 | 0/99/1 |
| IKGR_lgcn_dyn_nointent vs IKGR_dyn_nointent | ndcg@10 | 133,684 | 0.0883 | 0.0892 | -0.0009 (-1.0%) | [-0.0012, -0.0006] | 2.8e-39 | 26/46/29 |
| IKGR_lgcn_dyn_nointent vs IKGR_dyn_nointent | tail_recall@10 | 69,683 | 0.0014 | 0.0022 | -0.0009 (-39.1%) | [-0.0010, -0.0007] | 1.7e-11 | 0/98/2 |

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

**IKGR_dyn vs LightGCN — ndcg@10 by train-activity quintile**

| bucket | activity ≤ | users | diff | win % | loss % |
|---|---:|---:|---:|---:|---:|
| Q1 | 38 | 27,673 | +0.0116 | 23 | 15 |
| Q2 | 58 | 26,292 | +0.0055 | 26 | 19 |
| Q3 | 97 | 26,653 | +0.0040 | 30 | 24 |
| Q4 | 192 | 26,455 | +0.0022 | 36 | 29 |
| Q5 | 9852 | 26,611 | -0.0008 | 42 | 38 |

**IKGR_dyn vs LightGCN — tail_recall@10 by train-activity quintile**

| bucket | activity ≤ | users | diff | win % | loss % |
|---|---:|---:|---:|---:|---:|
| Q1 | 56 | 14,064 | +0.0022 | 1 | 0 |
| Q2 | 99 | 13,980 | +0.0019 | 1 | 0 |
| Q3 | 168 | 13,968 | +0.0011 | 1 | 0 |
| Q4 | 305 | 13,772 | +0.0007 | 1 | 0 |
| Q5 | 9852 | 13,899 | +0.0007 | 4 | 1 |

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

**IKGR_lgcn_dyn_nointent vs LightGCN — ndcg@10 by train-activity quintile**

| bucket | activity ≤ | users | diff | win % | loss % |
|---|---:|---:|---:|---:|---:|
| Q1 | 38 | 27,673 | +0.0071 | 19 | 15 |
| Q2 | 58 | 26,292 | +0.0029 | 21 | 19 |
| Q3 | 97 | 26,653 | +0.0031 | 26 | 23 |
| Q4 | 192 | 26,455 | +0.0022 | 32 | 29 |
| Q5 | 9852 | 26,611 | +0.0022 | 40 | 37 |

**IKGR_lgcn_dyn_nointent vs LightGCN — tail_recall@10 by train-activity quintile**

| bucket | activity ≤ | users | diff | win % | loss % |
|---|---:|---:|---:|---:|---:|
| Q1 | 56 | 14,064 | +0.0015 | 1 | 0 |
| Q2 | 99 | 13,980 | +0.0007 | 1 | 0 |
| Q3 | 168 | 13,968 | +0.0003 | 0 | 0 |
| Q4 | 305 | 13,772 | +0.0003 | 1 | 0 |
| Q5 | 9852 | 13,899 | +0.0001 | 2 | 1 |

**IKGR_lgcn_dyn vs IKGR_lgcn_dyn_nointent — ndcg@10 by train-activity quintile**

| bucket | activity ≤ | users | diff | win % | loss % |
|---|---:|---:|---:|---:|---:|
| Q1 | 38 | 27,673 | -0.0004 | 16 | 16 |
| Q2 | 58 | 26,292 | -0.0011 | 19 | 20 |
| Q3 | 97 | 26,653 | -0.0002 | 23 | 24 |
| Q4 | 192 | 26,455 | -0.0003 | 29 | 30 |
| Q5 | 9852 | 26,611 | -0.0017 | 36 | 39 |

**IKGR_lgcn_dyn vs IKGR_lgcn_dyn_nointent — tail_recall@10 by train-activity quintile**

| bucket | activity ≤ | users | diff | win % | loss % |
|---|---:|---:|---:|---:|---:|
| Q1 | 56 | 14,064 | -0.0014 | 0 | 1 |
| Q2 | 99 | 13,980 | -0.0006 | 0 | 1 |
| Q3 | 168 | 13,968 | -0.0002 | 0 | 0 |
| Q4 | 305 | 13,772 | -0.0003 | 0 | 1 |
| Q5 | 9852 | 13,899 | -0.0001 | 1 | 2 |

**IKGR_lgcn_dyn_nointent vs IKGR_dyn_nointent — ndcg@10 by train-activity quintile**

| bucket | activity ≤ | users | diff | win % | loss % |
|---|---:|---:|---:|---:|---:|
| Q1 | 38 | 27,673 | -0.0048 | 15 | 21 |
| Q2 | 58 | 26,292 | -0.0022 | 19 | 24 |
| Q3 | 97 | 26,653 | -0.0005 | 24 | 27 |
| Q4 | 192 | 26,455 | +0.0002 | 31 | 33 |
| Q5 | 9852 | 26,611 | +0.0028 | 39 | 39 |

**IKGR_lgcn_dyn_nointent vs IKGR_dyn_nointent — tail_recall@10 by train-activity quintile**

| bucket | activity ≤ | users | diff | win % | loss % |
|---|---:|---:|---:|---:|---:|
| Q1 | 56 | 14,064 | -0.0009 | 0 | 1 |
| Q2 | 99 | 13,980 | -0.0015 | 0 | 1 |
| Q3 | 168 | 13,968 | -0.0009 | 0 | 1 |
| Q4 | 305 | 13,772 | -0.0006 | 0 | 1 |
| Q5 | 9852 | 13,899 | -0.0005 | 1 | 4 |

# Accuracy / exposure Pareto (TO, seed means)

★ = not dominated on (NDCG@10, tail Recall@10); ◆ = not dominated on (NDCG@10, Coverage@10).

| result | seeds | NDCG@10 | ±sd | tail R@10 | Cov@10 | best epoch | frontier |
|---|---:|---:|---:|---:|---:|---:|---|
| IKGR_dyn | 3 | 0.0893 | 0.0002 | 0.0021 | 0.4103 | 22 | ★◆ |
| IKGR_lgcn_dyn | 3 | 0.0876 | 0.0001 | 0.0009 | 0.2763 | 22 |  |
| BPR | 3 | 0.0871 | 0.0000 | 0.0023 | 0.4310 | 29 | ★◆ |
| IKGR_lgcn_dyn_pop0p25 | 3 | 0.0832 | 0.0004 | 0.0025 | 0.3994 | 22 | ★ |
| IKGR_dyn_pop0p25 | 3 | 0.0789 | 0.0002 | 0.0092 | 0.6775 | 22 | ★◆ |
| IKGR_lgcn_dyn_pop0p5 | 3 | 0.0750 | 0.0009 | 0.0074 | 0.5812 | 22 |  |
| BPR_pop0p25 | 3 | 0.0715 | 0.0000 | 0.0130 | 0.7786 | 29 | ★◆ |
| IKGR_dyn_pop0p5 | 3 | 0.0580 | 0.0006 | 0.0246 | 0.9052 | 22 | ★◆ |
| BPR_pop0p5 | 3 | 0.0425 | 0.0000 | 0.0295 | 0.9577 | 29 | ★◆ |
| IKGR_lgcn_dyn_pop1p0 | 3 | 0.0332 | 0.0017 | 0.0160 | 0.9033 | 22 |  |
| IKGR_dyn_pop1p0 | 3 | 0.0039 | 0.0001 | 0.0118 | 0.7616 | 22 |  |
| BPR_pop1p0 | 3 | 0.0028 | 0.0000 | 0.0085 | 0.6178 | 29 |  |
