// Android driver for libeegnet.so: argv lib X.f64 y.i64 weights.safetensors probs_out.f64
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
typedef long long i64;
static double now(void) { struct timespec t; clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &t); return t.tv_sec * 1e3 + t.tv_nsec / 1e6; }
int main(int c, char **v) {
  void *L = dlopen(v[1], RTLD_NOW); if (!L) { printf("%s\n", dlerror()); return 1; }
  i64 (*create)(i64) = dlsym(L, "create");
  i64 (*lw)(i64, const char *) = dlsym(L, "load_weights");
  i64 (*predict)(i64, double *, i64, double *) = dlsym(L, "predict");
  double (*step)(i64, double *, i64 *, i64) = dlsym(L, "train_step");
  double *X = malloc(240 * 3000 * 8); i64 *y = malloc(240 * 8); double out[64 * 3];
  FILE *f = fopen(v[2], "rb"); fread(X, 8, 240 * 3000, f); fclose(f);
  f = fopen(v[3], "rb"); fread(y, 8, 240, f); fclose(f);
  i64 h = create(0); lw(h, v[4]);
  predict(h, X, 64, out);
  f = fopen(v[5], "wb"); fwrite(out, 8, 64 * 3, f); fclose(f);
  double t = now(); for (int i = 0; i < 100; i++) predict(h, X, 1, out);
  printf("predict1 %.3f ms\n", (now() - t) / 100);
  t = now(); double l0 = 0, l = 0;
  for (int e = 0; e < 5; e++) for (int i = 0; i < 240; i += 64) { int n = i + 64 > 240 ? 240 - i : 64; l = step(h, X + i * 3000, y + i, n); if (!e && !i) l0 = l; }
  printf("5 epochs %.1f ms loss %.6f -> %.6f\n", now() - t, l0, l);
  return 0;
}
