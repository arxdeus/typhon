# Memory (peak RSS / peak footprint, MiB, median of 9)

Typhon dir: `/Users/mind/orca/codon/typhon/target/release`  
uptime:  0:35  up 3 days,  5:38, 7 users, load averages: 1.58 1.52 2.15

| program | Codon RSS | Typhon RSS | ratio | Codon fp | Typhon fp |
|---|---|---|---|---|---|
| empty | 7.2 | 2.2 | 0.30 | 2.4 | 1.6 |
| hello | 7.2 | 2.2 | 0.30 | 2.4 | 1.6 |
| np_matmul | 211.4 | 177.4 | 0.84 | 204.3 | 171.0 |
| np_solve_chol | 203.0 | 201.0 | 0.99 | 195.9 | 194.6 |
| np_eigh_svd | 98.5 | 119.3 | 1.21 | 91.3 | 112.8 |
| np_small_linalg | 11.7 | 10.7 | 0.92 | 4.4 | 4.1 |
| np_arc_distance | 959.3 | 767.1 | 0.80 | 952.8 | 766.5 |
| np_jacobi_2d | 20.9 | 13.8 | 0.66 | 13.8 | 12.9 |
| np_covariance | 285.1 | 284.2 | 1.00 | 278.1 | 277.8 |
