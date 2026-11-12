# Memory (peak RSS / peak footprint, MiB, median of 9)

Typhon dir: `/Users/mind/orca/codon/typhon/target/release`  
uptime:  2:06  up 3 days,  7:08, 7 users, load averages: 23.45 21.29 21.50

| program | Codon RSS | Typhon RSS | ratio | Codon fp | Typhon fp |
|---|---|---|---|---|---|
| empty | 7.2 | 2.2 | 0.30 | 2.4 | 1.6 |
| hello | 7.2 | 2.2 | 0.30 | 2.4 | 1.6 |
| np_matmul | 201.8 | 177.3 | 0.88 | 204.3 | 171.0 |
| np_solve_chol | 191.0 | 175.2 | 0.92 | 187.9 | 186.6 |
| np_eigh_svd | 104.9 | 90.8 | 0.87 | 100.3 | 87.1 |
| np_small_linalg | 11.6 | 10.7 | 0.92 | 4.5 | 4.2 |
| np_arc_distance | 714.3 | 585.4 | 0.82 | 952.7 | 919.5 |
| np_jacobi_2d | 20.8 | 13.8 | 0.66 | 13.8 | 12.9 |
| np_covariance | 283.9 | 283.1 | 1.00 | 278.2 | 277.8 |
