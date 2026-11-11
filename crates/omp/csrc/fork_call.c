/* Variadic entry points of the OpenMP runtime ABI that stable Rust cannot
 * define: __kmpc_fork_call collects its pointer arguments into an array and
 * forwards to Rust; typhon_omp_invoke_microtask spreads them back out when a
 * team member runs the outlined microtask (like libomp's
 * __kmp_invoke_microtask). */
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define TYPHON_OMP_MAX_ARGS 16

typedef struct ident ident_t;
typedef void (*kmpc_micro)(int32_t *, int32_t *, ...); /* only for the fork signature */

extern void typhon_omp_fork_call(ident_t *loc, int32_t argc, void *microtask,
                                 void **args);

void __kmpc_fork_call(ident_t *loc, int32_t argc, kmpc_micro microtask, ...) {
  void *args[TYPHON_OMP_MAX_ARGS];
  va_list ap;
  if (argc < 0 || argc > TYPHON_OMP_MAX_ARGS) {
    fprintf(stderr, "typhon-omp: __kmpc_fork_call: unsupported argc %d (max %d)\n",
            (int)argc, TYPHON_OMP_MAX_ARGS);
    abort();
  }
  va_start(ap, microtask);
  for (int32_t i = 0; i < argc; i++)
    args[i] = va_arg(ap, void *);
  va_end(ap);
  typhon_omp_fork_call(loc, argc, (void *)microtask, args);
}

/* Microtasks are ordinary (non-variadic) functions, so each arity must be called
 * through a matching non-variadic pointer type: on Apple arm64 variadic arguments
 * are passed on the stack, unlike named ones. */
#define P void *
#define MT(...) ((void (*)(int32_t *, int32_t *, ##__VA_ARGS__))fn)

void typhon_omp_invoke_microtask(void *fn, int32_t *gtid, int32_t *tid, int32_t argc,
                                 void **p) {
  switch (argc) {
  case 0: MT()(gtid, tid); break;
  case 1: MT(P)(gtid, tid, p[0]); break;
  case 2: MT(P, P)(gtid, tid, p[0], p[1]); break;
  case 3: MT(P, P, P)(gtid, tid, p[0], p[1], p[2]); break;
  case 4: MT(P, P, P, P)(gtid, tid, p[0], p[1], p[2], p[3]); break;
  case 5: MT(P, P, P, P, P)(gtid, tid, p[0], p[1], p[2], p[3], p[4]); break;
  case 6: MT(P, P, P, P, P, P)(gtid, tid, p[0], p[1], p[2], p[3], p[4], p[5]); break;
  case 7: MT(P, P, P, P, P, P, P)(gtid, tid, p[0], p[1], p[2], p[3], p[4], p[5], p[6]); break;
  case 8: MT(P, P, P, P, P, P, P, P)(gtid, tid, p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7]); break;
  case 9: MT(P, P, P, P, P, P, P, P, P)(gtid, tid, p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]); break;
  case 10: MT(P, P, P, P, P, P, P, P, P, P)(gtid, tid, p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8], p[9]); break;
  case 11: MT(P, P, P, P, P, P, P, P, P, P, P)(gtid, tid, p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8], p[9], p[10]); break;
  case 12: MT(P, P, P, P, P, P, P, P, P, P, P, P)(gtid, tid, p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8], p[9], p[10], p[11]); break;
  case 13: MT(P, P, P, P, P, P, P, P, P, P, P, P, P)(gtid, tid, p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8], p[9], p[10], p[11], p[12]); break;
  case 14: MT(P, P, P, P, P, P, P, P, P, P, P, P, P, P)(gtid, tid, p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8], p[9], p[10], p[11], p[12], p[13]); break;
  case 15: MT(P, P, P, P, P, P, P, P, P, P, P, P, P, P, P)(gtid, tid, p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8], p[9], p[10], p[11], p[12], p[13], p[14]); break;
  case 16: MT(P, P, P, P, P, P, P, P, P, P, P, P, P, P, P, P)(gtid, tid, p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8], p[9], p[10], p[11], p[12], p[13], p[14], p[15]); break;
  default:
    fprintf(stderr, "typhon-omp: unsupported microtask argc %d\n", (int)argc);
    abort();
  }
}
