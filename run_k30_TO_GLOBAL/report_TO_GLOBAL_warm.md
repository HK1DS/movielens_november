# Paired comparisons (TO_GLOBAL, test, users with >= 1 train interaction(s))

Unit = user (metric averaged over seeds first). CI = 95% bootstrap over users; p = Wilcoxon signed-rank; W/T/L = % users where A wins/ties/loses.

| A vs B | metric | users | A | B | diff (rel) | 95% CI | p | W/T/L % |
|---|---|---:|---:|---:|---:|---|---:|---|
| IKGR_dyn vs BPR | ndcg@10 | 3,186 | 0.2246 | 0.2242 | +0.0004 (+0.2%) | [-0.0024, +0.0032] | 0.33 | 37/28/35 |
| IKGR_dyn vs BPR | tail_recall@10 | 3,059 | 0.0008 | 0.0005 | +0.0003 (+49.1%) | [+0.0000, +0.0006] | 0.0075 | 5/92/3 |
| IKGR_dyn vs LightGCN | ndcg@10 | 3,186 | 0.2246 | 0.2173 | +0.0074 (+3.4%) | [+0.0041, +0.0105] | 8.8e-08 | 40/28/32 |
| IKGR_dyn vs LightGCN | tail_recall@10 | 3,059 | 0.0008 | 0.0003 | +0.0005 (+192.6%) | [+0.0003, +0.0008] | 1.8e-10 | 7/92/1 |
| IKGR_dyn vs IKGR_kgoff | ndcg@10 | 3,186 | 0.2246 | 0.2241 | +0.0005 (+0.2%) | [-0.0024, +0.0035] | 0.39 | 37/27/36 |
| IKGR_dyn vs IKGR_kgoff | tail_recall@10 | 3,059 | 0.0008 | 0.0005 | +0.0003 (+48.4%) | [+0.0000, +0.0006] | 0.016 | 5/93/3 |
| IKGR_dyn vs IKGR_dyn_nointent | ndcg@10 | 3,186 | 0.2246 | 0.2231 | +0.0015 (+0.7%) | [-0.0001, +0.0031] | 0.047 | 37/28/34 |
| IKGR_dyn vs IKGR_dyn_nointent | tail_recall@10 | 3,059 | 0.0008 | 0.0006 | +0.0002 (+35.5%) | [+0.0000, +0.0005] | 0.071 | 4/93/2 |
| IKGR_lgcn_dyn_nointent vs LightGCN | ndcg@10 | 3,186 | 0.2245 | 0.2173 | +0.0072 (+3.3%) | [+0.0042, +0.0102] | 2.2e-08 | 38/31/30 |
| IKGR_lgcn_dyn_nointent vs LightGCN | tail_recall@10 | 3,059 | 0.0004 | 0.0003 | +0.0001 (+48.6%) | [+0.0001, +0.0002] | 0.031 | 3/96/1 |
| IKGR_lgcn_dyn vs IKGR_lgcn_dyn_nointent | ndcg@10 | 3,186 | 0.2257 | 0.2245 | +0.0012 (+0.6%) | [-0.0001, +0.0025] | 0.18 | 33/35/32 |
| IKGR_lgcn_dyn vs IKGR_lgcn_dyn_nointent | tail_recall@10 | 3,059 | 0.0004 | 0.0004 | +0.0000 (+6.3%) | [-0.0000, +0.0001] | 0.55 | 2/97/1 |
| IKGR_lgcn_dyn_nointent vs IKGR_dyn_nointent | ndcg@10 | 3,186 | 0.2245 | 0.2231 | +0.0014 (+0.6%) | [-0.0009, +0.0036] | 0.12 | 37/29/35 |
| IKGR_lgcn_dyn_nointent vs IKGR_dyn_nointent | tail_recall@10 | 3,059 | 0.0004 | 0.0006 | -0.0002 (-31.2%) | [-0.0003, -0.0000] | 0.00017 | 2/93/5 |

**IKGR_dyn vs BPR — ndcg@10 by train-activity quintile**

| bucket | activity ≤ | users | diff | win % | loss % |
|---|---:|---:|---:|---:|---:|
| Q1 | 109 | 640 | -0.0088 | 35 | 40 |
| Q2 | 250 | 636 | +0.0049 | 38 | 33 |
| Q3 | 441 | 637 | +0.0003 | 35 | 34 |
| Q4 | 808 | 636 | +0.0056 | 40 | 30 |
| Q5 | 5371 | 637 | +0.0002 | 39 | 37 |

**IKGR_dyn vs BPR — tail_recall@10 by train-activity quintile**

| bucket | activity ≤ | users | diff | win % | loss % |
|---|---:|---:|---:|---:|---:|
| Q1 | 114 | 612 | -0.0003 | 1 | 0 |
| Q2 | 261 | 613 | +0.0004 | 4 | 1 |
| Q3 | 458 | 610 | +0.0002 | 4 | 2 |
| Q4 | 826 | 612 | +0.0008 | 5 | 2 |
| Q5 | 5371 | 612 | +0.0002 | 12 | 7 |

**IKGR_dyn vs LightGCN — ndcg@10 by train-activity quintile**

| bucket | activity ≤ | users | diff | win % | loss % |
|---|---:|---:|---:|---:|---:|
| Q1 | 109 | 640 | +0.0050 | 39 | 36 |
| Q2 | 250 | 636 | +0.0038 | 39 | 31 |
| Q3 | 441 | 637 | +0.0091 | 39 | 29 |
| Q4 | 808 | 636 | +0.0102 | 42 | 29 |
| Q5 | 5371 | 637 | +0.0086 | 42 | 33 |

**IKGR_dyn vs LightGCN — tail_recall@10 by train-activity quintile**

| bucket | activity ≤ | users | diff | win % | loss % |
|---|---:|---:|---:|---:|---:|
| Q1 | 114 | 612 | -0.0002 | 1 | 0 |
| Q2 | 261 | 613 | +0.0006 | 5 | 0 |
| Q3 | 458 | 610 | +0.0006 | 6 | 0 |
| Q4 | 826 | 612 | +0.0011 | 6 | 1 |
| Q5 | 5371 | 612 | +0.0006 | 16 | 2 |

**IKGR_dyn vs IKGR_kgoff — ndcg@10 by train-activity quintile**

| bucket | activity ≤ | users | diff | win % | loss % |
|---|---:|---:|---:|---:|---:|
| Q1 | 109 | 640 | -0.0086 | 36 | 40 |
| Q2 | 250 | 636 | +0.0056 | 38 | 33 |
| Q3 | 441 | 637 | +0.0004 | 35 | 35 |
| Q4 | 808 | 636 | +0.0055 | 39 | 34 |
| Q5 | 5371 | 637 | -0.0001 | 39 | 37 |

**IKGR_dyn vs IKGR_kgoff — tail_recall@10 by train-activity quintile**

| bucket | activity ≤ | users | diff | win % | loss % |
|---|---:|---:|---:|---:|---:|
| Q1 | 114 | 612 | +0.0000 | 1 | 1 |
| Q2 | 261 | 613 | +0.0003 | 4 | 2 |
| Q3 | 458 | 610 | -0.0001 | 4 | 2 |
| Q4 | 826 | 612 | +0.0009 | 5 | 2 |
| Q5 | 5371 | 612 | +0.0002 | 11 | 6 |

**IKGR_dyn vs IKGR_dyn_nointent — ndcg@10 by train-activity quintile**

| bucket | activity ≤ | users | diff | win % | loss % |
|---|---:|---:|---:|---:|---:|
| Q1 | 109 | 640 | +0.0034 | 41 | 33 |
| Q2 | 250 | 636 | +0.0015 | 35 | 33 |
| Q3 | 441 | 637 | -0.0005 | 34 | 36 |
| Q4 | 808 | 636 | -0.0001 | 36 | 36 |
| Q5 | 5371 | 637 | +0.0031 | 40 | 34 |

**IKGR_dyn vs IKGR_dyn_nointent — tail_recall@10 by train-activity quintile**

| bucket | activity ≤ | users | diff | win % | loss % |
|---|---:|---:|---:|---:|---:|
| Q1 | 114 | 612 | +0.0000 | 1 | 1 |
| Q2 | 261 | 613 | +0.0003 | 3 | 1 |
| Q3 | 458 | 610 | +0.0001 | 4 | 1 |
| Q4 | 826 | 612 | +0.0006 | 4 | 3 |
| Q5 | 5371 | 612 | +0.0001 | 9 | 7 |

**IKGR_lgcn_dyn_nointent vs LightGCN — ndcg@10 by train-activity quintile**

| bucket | activity ≤ | users | diff | win % | loss % |
|---|---:|---:|---:|---:|---:|
| Q1 | 109 | 640 | +0.0008 | 39 | 34 |
| Q2 | 250 | 636 | +0.0083 | 37 | 30 |
| Q3 | 441 | 637 | +0.0078 | 37 | 28 |
| Q4 | 808 | 636 | +0.0127 | 39 | 29 |
| Q5 | 5371 | 637 | +0.0067 | 40 | 31 |

**IKGR_lgcn_dyn_nointent vs LightGCN — tail_recall@10 by train-activity quintile**

| bucket | activity ≤ | users | diff | win % | loss % |
|---|---:|---:|---:|---:|---:|
| Q1 | 114 | 612 | +0.0000 | 1 | 0 |
| Q2 | 261 | 613 | +0.0001 | 2 | 0 |
| Q3 | 458 | 610 | +0.0001 | 2 | 1 |
| Q4 | 826 | 612 | +0.0002 | 3 | 1 |
| Q5 | 5371 | 612 | +0.0002 | 7 | 2 |

**IKGR_lgcn_dyn vs IKGR_lgcn_dyn_nointent — ndcg@10 by train-activity quintile**

| bucket | activity ≤ | users | diff | win % | loss % |
|---|---:|---:|---:|---:|---:|
| Q1 | 109 | 640 | +0.0008 | 33 | 36 |
| Q2 | 250 | 636 | +0.0000 | 30 | 32 |
| Q3 | 441 | 637 | +0.0021 | 32 | 28 |
| Q4 | 808 | 636 | +0.0014 | 34 | 31 |
| Q5 | 5371 | 637 | +0.0020 | 36 | 30 |

**IKGR_lgcn_dyn vs IKGR_lgcn_dyn_nointent — tail_recall@10 by train-activity quintile**

| bucket | activity ≤ | users | diff | win % | loss % |
|---|---:|---:|---:|---:|---:|
| Q1 | 114 | 612 | -0.0001 | 0 | 0 |
| Q2 | 261 | 613 | +0.0000 | 1 | 0 |
| Q3 | 458 | 610 | +0.0001 | 2 | 1 |
| Q4 | 826 | 612 | +0.0001 | 1 | 1 |
| Q5 | 5371 | 612 | -0.0000 | 4 | 3 |

**IKGR_lgcn_dyn_nointent vs IKGR_dyn_nointent — ndcg@10 by train-activity quintile**

| bucket | activity ≤ | users | diff | win % | loss % |
|---|---:|---:|---:|---:|---:|
| Q1 | 109 | 640 | -0.0009 | 39 | 35 |
| Q2 | 250 | 636 | +0.0060 | 37 | 32 |
| Q3 | 441 | 637 | -0.0018 | 33 | 36 |
| Q4 | 808 | 636 | +0.0023 | 37 | 34 |
| Q5 | 5371 | 637 | +0.0011 | 38 | 36 |

**IKGR_lgcn_dyn_nointent vs IKGR_dyn_nointent — tail_recall@10 by train-activity quintile**

| bucket | activity ≤ | users | diff | win % | loss % |
|---|---:|---:|---:|---:|---:|
| Q1 | 114 | 612 | +0.0002 | 1 | 1 |
| Q2 | 261 | 613 | -0.0001 | 1 | 3 |
| Q3 | 458 | 610 | -0.0004 | 1 | 4 |
| Q4 | 826 | 612 | -0.0004 | 1 | 5 |
| Q5 | 5371 | 612 | -0.0003 | 3 | 12 |

# Per-result means (users with >= 1 train interaction(s), seed means per user)

| result | seeds | users (ndcg@10) | ndcg@10 | tail_recall@10 |
|---|---:|---:|---:|---:|
| IKGR_lgcn_dyn | 3 | 3,186 | 0.2257 | 0.0004 |
| IKGR_dyn | 3 | 3,186 | 0.2246 | 0.0008 |
| IKGR_lgcn_dyn_nointent | 3 | 3,186 | 0.2245 | 0.0004 |
| BPR | 3 | 3,186 | 0.2242 | 0.0005 |
| IKGR_kgoff | 3 | 3,186 | 0.2241 | 0.0005 |
| IKGR_dyn_nointent | 3 | 3,186 | 0.2231 | 0.0006 |
| LightGCN | 3 | 3,186 | 0.2173 | 0.0003 |
