# Memory (peak RSS / peak footprint, MiB, median of 9)

Typhon dir: `/Users/mind/orca/codon/typhon/target/release`  
uptime: 22:19  up 3 days,  3:21, 7 users, load averages: 1.83 2.62 5.59

| program | Codon RSS | Typhon RSS | ratio | Codon fp | Typhon fp |
|---|---|---|---|---|---|
| empty | 7.2 | 2.2 | 0.30 | 2.4 | 1.6 |
| hello | 7.2 | 2.2 | 0.30 | 2.4 | 1.6 |
| np_matmul | 211.4 | 177.4 | 0.84 | 204.3 | 171.0 |
| np_solve_chol | 141.9 | 201.1 | 1.42 | 134.8 | 194.7 |
| np_eigh_svd | 105.2 | 104.6 | 0.99 | 98.0 | 98.0 |
| np_small_linalg | 11.7 | 10.7 | 0.92 | 4.4 | 4.1 |
| np_arc_distance | 959.3 | 843.5 | 0.88 | 952.7 | 843.0 |
| np_jacobi_2d | 20.9 | 13.9 | 0.66 | 13.8 | 12.9 |
| np_covariance | 285.2 | 284.2 | 1.00 | 278.2 | 277.8 |
