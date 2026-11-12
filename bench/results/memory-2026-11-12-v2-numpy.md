# Memory (peak RSS / peak footprint, MiB, median of 3)

Typhon dir: `/Users/mind/orca/codon/typhon/target/release`  
uptime: 21:18  up 3 days,  2:20, 7 users, load averages: 1.80 2.81 4.34

| program | Codon RSS | Typhon RSS | ratio | Codon fp | Typhon fp |
|---|---|---|---|---|---|
| empty | 7.2 | 2.7 | 0.37 | 2.4 | 2.1 |
| hello | 7.2 | 2.7 | 0.37 | 2.4 | 2.1 |
| np_matmul | 178.9 | 210.0 | 1.17 | 171.8 | 203.6 |
| np_solve_chol | 233.2 | 202.0 | 0.87 | 226.1 | 195.5 |
| np_eigh_svd | 105.2 | 127.1 | 1.21 | 98.0 | 120.5 |
| np_small_linalg | 11.7 | 11.2 | 0.96 | 4.5 | 4.7 |
| np_arc_distance | 991.4 | 767.8 | 0.77 | 984.8 | 767.3 |
| np_jacobi_2d | 20.9 | 14.4 | 0.69 | 13.8 | 13.4 |
| np_covariance | 285.2 | 284.7 | 1.00 | 278.2 | 278.3 |
