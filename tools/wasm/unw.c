#include <stdio.h>
#include <stdlib.h>
int _Unwind_RaiseException(void*e){fputs("typhon(wasm): uncaught exception (no EH on wasm32 M1)\n",stderr);abort();}
