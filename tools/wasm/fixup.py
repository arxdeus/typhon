#!/usr/bin/env python3
"""Prototype wasm32 ABI fixup on Typhon textual IR (what codegen must do natively).
1. size_t runtime/libc params/returns: i64 -> i32 (trunc/zext at call sites).
2. {ptr,i64} by-value args -> pointer to stack copy (wasm C ABI passes aggregates indirectly).
3. main -> __main_argc_argv.
"""
import re, sys

SIZE_T = {  # name: (param indices that are size_t, returns size_t)
    'seq_alloc': ([0], False), 'seq_alloc_atomic': ([0], False),
    'seq_alloc_uncollectable': ([0], False), 'seq_alloc_atomic_uncollectable': ([0], False),
    'seq_realloc': ([1, 2], False), 'memcmp': ([2], False), 'strlen': ([], True),
    'memcpy': ([2], False), 'memmove': ([2], False), 'memset': ([2], False),
}
src = open(sys.argv[1]).read()
out, n = [], [0]

def fresh():
    n[0] += 1
    return f'%__w{n[0]}'

def split_args(s):
    depth, cur, res = 0, '', []
    for c in s:
        if c in '({[<': depth += 1
        if c in ')}]>': depth -= 1
        if c == ',' and depth == 0:
            res.append(cur.strip()); cur = ''
        else:
            cur += c
    if cur.strip(): res.append(cur.strip())
    return res

AGG = '{ ptr, i64 }'
for line in src.split('\n'):
    m = re.match(r'^declare (.*?)@([\w.]+)\((.*)\)(.*)$', line)
    if m and m.group(2) in SIZE_T:
        idx, ret = SIZE_T[m.group(2)]
        ps = split_args(m.group(3))
        ps = [re.sub(r'^i64', 'i32', p) if i in idx else p for i, p in enumerate(ps)]
        r = m.group(1).replace('i64', 'i32') if ret else m.group(1)
        out.append(f'declare {r}@{m.group(2)}({", ".join(ps)}){m.group(4)}'); continue
    if m and AGG in m.group(3):
        ps = [('ptr' if p.startswith(AGG) else p) for p in split_args(m.group(3))]
        out.append(f'declare {m.group(1)}@{m.group(2)}({", ".join(ps)}){m.group(4)}'); continue
    if line.startswith('define') and '@main(' in line:
        line = line.replace('@main(', '@__main_argc_argv(')
    c = re.match(r'^(\s*)(%[\w.]+ = )?((?:tail |musttail )?(?:call|invoke)) (.*?)@([\w.]+)\((.*)\)(.*)$', line)
    if c and (c.group(5) in SIZE_T or AGG in c.group(6)):
        ind, lhs, kw, rty, fn, args, rest = c.groups()
        pre = []
        al = split_args(args)
        idx, ret = SIZE_T.get(fn, ([], False))
        for i, a in enumerate(al):
            if i in idx and a.startswith('i64'):
                v = fresh(); pre.append(f'{ind}{v} = trunc {a} to i32'); al[i] = f'i32 {v}'
            elif a.startswith(AGG):
                p = fresh(); pre.append(f'{ind}{p} = alloca {AGG}')
                pre.append(f'{ind}store {a}, ptr {p}'); al[i] = f'ptr {p}'
        out.extend(pre)
        if any('alloca' in p for p in pre):
            kw = kw.replace('musttail ', '').replace('tail ', '')
        if ret and lhs:
            t = fresh()
            out.append(f'{ind}{t} = {kw} {rty.replace("i64", "i32")}@{fn}({", ".join(al)}){rest}')
            out.append(f'{ind}{lhs}zext i32 {t} to i64'); continue
        out.append(f'{ind}{lhs or ""}{kw} {rty}@{fn}({", ".join(al)}){rest}'); continue
    out.append(line)
open(sys.argv[2], 'w').write('\n'.join(out))
