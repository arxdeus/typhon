# Memory (peak RSS / peak footprint, MiB, median of 3)

Typhon dir: `/Users/mind/.jcode/scratch/memaudit/baseline`  
uptime: 11:17  up 2 days, 16:19, 8 users, load averages: 1.94 4.91 5.12

| program | Codon RSS | Typhon RSS | ratio | Codon fp | Typhon fp | CPython RSS |
|---|---|---|---|---|---|---|
| empty | 7.2 | 6.7 | 0.92 | 2.4 | 2.4 | 8.1 |
| hello | 7.2 | 6.8 | 0.93 | 2.4 | 2.4 | 8.1 |
| binary_trees | 45.5 | 56.9 | 1.25 | 38.4 | 50.3 | - |
| chaos | 56.6 | 143.5 | 2.54 | 49.4 | 136.9 | 43.3 |
| fannkuch | 7.4 | 6.9 | 0.93 | 2.5 | 2.5 | 8.6 |
| float | 493.4 | 492.9 | 1.00 | 486.6 | 486.6 | 2642.0 |
| go | 13.1 | 12.6 | 0.96 | 6.0 | 6.0 | - |
| mandelbrot | 9.4 | 8.9 | 0.95 | 4.5 | 4.5 | 12.5 |
| nbody | 7.4 | 6.9 | 0.93 | 2.5 | 2.5 | 8.5 |
| primes | 7.4 | 6.9 | 0.93 | 2.5 | 2.5 | 8.2 |
| set_partition | 11.6 | 11.0 | 0.96 | 4.5 | 4.5 | 8.3 |
| spectral_norm | 13.1 | 15.5 | 1.18 | 6.0 | 9.0 | 8.3 |
