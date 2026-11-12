# usage: mk.py expected.txt src.py out.py
import re,sys
exp={}
for l in open(sys.argv[1]):
    m=re.match(r'^(\w+) = (.*)$',l.strip())
    if m: exp[m.group(1)]=m.group(2)
out=[]
for line in open(sys.argv[2]).read().split('\n'):
    m=re.match(r'^(\s*)SHOW\("(\w+)", (.*), ([0-9.e-]+)\)$',line)
    m2=re.match(r'^(\s*)EQ\("(\w+)", (.*)\)$',line)
    m3=re.match(r'^(\s*)NUM\("(\w+)", (.*), ([0-9.e-]+)\)$',line)
    if m:
        i,k,e,t=m.groups(); out.append(f'{i}assert close({e}, {exp[k]}, {t})  # {k}')
    elif m2:
        i,k,e=m2.groups(); out.append(f'{i}assert {e} == {exp[k]}  # {k}')
    elif m3:
        i,k,e,t=m3.groups(); out.append(f'{i}assert closef({e}, {exp[k]}, {t})  # {k}')
    else: out.append(line)
open(sys.argv[3],'w').write('\n'.join(out))
