#include <stddef.h>
#include <string.h>
void GC_allow_register_threads(void){}
size_t GC_get_prof_stats(void*, size_t);
size_t GC_get_prof_stats_unsafe(void*p, size_t n){ return GC_get_prof_stats(p,n); }
