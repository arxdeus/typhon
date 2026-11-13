// Mobile check for the Typhon logpower+LDA library. Usage: driver <lib> <datadir> <tmpdir>
#include <dlfcn.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
typedef int64_t i64;
static double cpu(void) { struct timespec t; clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &t); return t.tv_sec + t.tv_nsec * 1e-9; }
static void *rd(const char *dir, const char *n, size_t sz) {
  char p[1024]; snprintf(p, sizeof p, "%s/%s", dir, n);
  FILE *f = fopen(p, "rb"); if (!f) { perror(p); exit(2); }
  void *b = malloc(sz); if (fread(b, 1, sz, f) != sz) { fprintf(stderr, "short %s\n", p); exit(2); } fclose(f); return b;
}
#define SYM(n, R, A) R (*n)A = (R(*)A)dlsym(L, #n); if (!n) { fprintf(stderr, "missing %s\n", #n); return 3; }
int main(int argc, char **argv) {
  if (argc < 4) return 1;
  void *L = dlopen(argv[1], RTLD_NOW); if (!L) { fprintf(stderr, "dlopen: %s\n", dlerror()); return 3; }
  SYM(create, i64, (void)) SYM(feat, i64, (double *, i64, double *))
  SYM(fit, i64, (i64, double *, i64 *, i64))
  SYM(predict, i64, (i64, double *, i64, i64 *, double *))
  SYM(save, i64, (i64, const char *)) SYM(load, i64, (const char *)) SYM(n_trials, i64, (i64))
  enum { N = 60 };
  double *X = rd(argv[2], "X.bin", N * 3000 * 8), *rp = rd(argv[2], "ref_proba.bin", N * 3 * 8), *rf = rd(argv[2], "ref_feat.bin", N * 12 * 8);
  i64 *y = rd(argv[2], "y.bin", N * 8), *rl = rd(argv[2], "ref_labels.bin", N * 8);
  double ff[N * 12], pr[N * 3], p1[3]; i64 lab[N], l1;
  feat(X, N, ff); double df = 0; for (int i = 0; i < N * 12; i++) df = fmax(df, fabs(ff[i] - rf[i]));
  double t0 = cpu(); i64 h = create(); i64 rc = fit(h, X, y, N); double tfit = cpu() - t0;
  printf("fit rc=%lld n_trials=%lld\n", (long long)rc, (long long)n_trials(h));
  predict(h, X, N, lab, pr);
  double dp = 0; int le = 0; for (int i = 0; i < N * 3; i++) dp = fmax(dp, fabs(pr[i] - rp[i]));
  for (int i = 0; i < N; i++) le += lab[i] != rl[i];
  int R = 200; t0 = cpu(); for (int i = 0; i < R; i++) predict(h, X, 1, &l1, p1); double tp = (cpu() - t0) / R;
  char path[1100]; snprintf(path, sizeof path, "%s/model.txt", argv[3]);
  save(h, path); i64 h2 = load(path); double dl = 0; int ll = 0;
  double pr2[N * 3]; i64 lab2[N]; predict(h2, X, N, lab2, pr2);
  for (int i = 0; i < N * 3; i++) dl = fmax(dl, fabs(pr2[i] - pr[i]));
  for (int i = 0; i < N; i++) ll += lab2[i] != lab[i];
  printf("RESULT feat_maxabs=%.3e proba_maxabs=%.3e label_mismatch=%d load_proba_diff=%.3e load_label_mismatch=%d\n", df, dp, le, dl, ll);
  printf("TIMING fit60_ms=%.3f predict1_ms=%.4f\n", tfit * 1e3, tp * 1e3);
  return 0;
}
