import ctypes as C, numpy as np, time, sys, os
sys.path.insert(0,'/Users/mind/orca/codon/typhon/test/bci')
from gen_logpower_lda import make_data, build
lib=C.CDLL('/Users/mind/.jcode/scratch/bci-logpower/lib/liblogpower.dylib')
D=C.POINTER(C.c_double); I=C.POINTER(C.c_int64)
for n,a,r in [('create',[],C.c_int64),('feat',[D,C.c_int64,D],C.c_int64),('fit',[C.c_int64,D,I,C.c_int64],C.c_int64),
  ('add_trials',[C.c_int64,D,I,C.c_int64],C.c_int64),('refit',[C.c_int64],C.c_int64),
  ('predict',[C.c_int64,D,C.c_int64,I,D],C.c_int64),('save',[C.c_int64,C.c_char_p],C.c_int64),('load',[C.c_char_p],C.c_int64)]:
    f=getattr(lib,n); f.argtypes=a; f.restype=r
d=lambda a:a.ctypes.data_as(D); ii=lambda a:a.ctypes.data_as(I)
def rel(a,b): return float(np.max(np.abs(a-b)/np.maximum(np.abs(b),1e-12)))
X,y=make_data(60); y=y.astype(np.int64)
rng=np.random.RandomState(1); X2,y2=make_data(30); X2=X2*1.1+rng.randn(*X2.shape); y2=y2.astype(np.int64)
m=build(); m.fit(X,y)
h=lib.create(); out=np.zeros((60,12))
print('feat',lib.feat(d(X),60,d(out)),'relerr',rel(out,m[:-1].transform(X)))
print('fit',lib.fit(h,d(X),ii(y),60))
lab=np.zeros(60,np.int64); pr=np.zeros((60,3))
print('nc',lib.predict(h,d(X),60,ii(lab),d(pr)))
print('labels equal',(lab==m.predict(X)).all(),'proba relerr',rel(pr,m.predict_proba(X)),'abs',np.abs(pr-m.predict_proba(X)).max())
# noisy test set
Xt=X2; mp=m.predict_proba(Xt); lab2=np.zeros(30,np.int64); pr2=np.zeros((30,3)); lib.predict(h,d(Xt),30,ii(lab2),d(pr2))
print('test-set labels equal',(lab2==m.predict(Xt)).all(),'abs proba',np.abs(pr2-mp).max())
# calibration add + refit
print('add',lib.add_trials(h,d(X2),ii(y2),30),'refit',lib.refit(h))
m2=build(); m2.fit(np.concatenate([X,X2]),np.concatenate([y,y2]))
Xe,ye=make_data(12); Xe=Xe+rng.randn(*Xe.shape)*3
lab3=np.zeros(12,np.int64); pr3=np.zeros((12,3)); lib.predict(h,d(Xe),12,ii(lab3),d(pr3))
print('refit labels equal',(lab3==m2.predict(Xe)).all(),'abs proba',np.abs(pr3-m2.predict_proba(Xe)).max(),'rel',rel(pr3,m2.predict_proba(Xe)))
# save/load
lib.save(h,b'/tmp/bci_model.txt'); h2=lib.load(b'/tmp/bci_model.txt')
lab4=np.zeros(12,np.int64); pr4=np.zeros((12,3)); lib.predict(h2,d(Xe),12,ii(lab4),d(pr4))
print('load identical',(lab4==lab3).all(),np.abs(pr4-pr3).max()); print('refit after load',lib.refit(h2),np.abs(pr4-pr3).max())
lib.predict(h2,d(Xe),12,ii(lab4),d(pr4)); print('after refit diff',np.abs(pr4-pr3).max())
# latency, 600 trials
Xb,yb=make_data(600); yb=yb.astype(np.int64); hb=lib.create()
t=time.process_time(); lib.fit(hb,d(Xb),ii(yb),600); print('typhon fit600 cpu s',time.process_time()-t)
t=time.process_time(); mb=build(); mb.fit(Xb,yb); print('python fit600 cpu s',time.process_time()-t)
labb=np.zeros(600,np.int64); prb=np.zeros((600,3))
t=time.process_time(); lib.predict(hb,d(Xb),600,ii(labb),d(prb)); tt=time.process_time()-t
t=time.process_time(); mb.predict(Xb); tp=time.process_time()-t
print('predict600 typhon/python ms per trial',tt/600*1e3,tp/600*1e3)
x1=Xb[:1].copy(); l1=np.zeros(1,np.int64); p1=np.zeros((1,3))
t=time.process_time()
for _ in range(200): lib.predict(hb,d(x1),1,ii(l1),d(p1))
tt=(time.process_time()-t)/200
t=time.process_time()
for _ in range(200): mb.predict(x1)
print('single trial ms typhon/python',tt*1e3,(time.process_time()-t)/200*1e3)
