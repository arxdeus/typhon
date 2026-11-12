#!/usr/bin/env python3
"""Typhon vs Codon vs Cython vs CPython benchmark harness.

Usage:
  compare.py check            # correctness pass with tiny args
  compare.py measure [names]  # serial measurement pass -> results/<date>.{json,md}
  compare.py report FILE.json # re-render markdown from a raw json
  compare.py --set numpy check|measure [names]  # NumPy set (src_np/), results/<date>-numpy.*

See README.md in this directory for the method.
"""
import datetime, json, math, os, platform, shutil, statistics, subprocess, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
TYPHON = HERE.parent
ROOT = TYPHON.parent
SRC = HERE / "src"
TYPED = HERE / "cython_typed"
RESULTS = HERE / "results"
SCRATCH = Path(os.environ.get("JCODE_SCRATCH_DIR", "/tmp")) / "typhon-bench"
PY = Path(os.environ.get("BENCH_PYTHON", Path(os.environ.get("JCODE_SCRATCH_DIR", "/tmp")) / "cyenv/bin/python"))
CODON = ROOT / "build/codon"
TYPHON_BIN = TYPHON / "target/release/typhon"
ENV = dict(os.environ, CODON_PATH=str(ROOT / "stdlib"))

# name -> (measurement args, tiny correctness args). "{out}" = scratch output file.
BENCH = {
    "binary_trees": (["17"], ["10"]),
    "chaos": (["{out}"], ["{out}"]),          # size hard-coded (2048^2, 1e6 iters)
    "fannkuch": (["10"], ["7"]),
    "float": (["10000000"], ["10000"]),
    "go": ([], []),                           # size hard-coded (100 games)
    "mandelbrot": (["512"], ["64"]),
    "nbody": (["3000000"], ["1000"]),
    "primes": (["30000"], ["1000"]),
    "set_partition": (["14"], ["8"]),
    "spectral_norm": (["40"], ["1"]),
}
# NumPy set: same .py for every backend, no typed Cython. Codon's numpy needs Accelerate
# linked explicitly on macOS (Typhon links it itself).
BENCH_NP = {
    "np_matmul": (["2048", "60"], ["128", "2"]),
    "np_solve_chol": (["2000", "40"], ["64", "2"]),
    "np_eigh_svd": (["1000", "25"], ["40", "2"]),
    "np_small_linalg": (["500000"], ["1000"]),
    "np_arc_distance": (["10000000", "15"], ["1000", "2"]),
    "np_jacobi_2d": (["700", "2000"], ["30", "5"]),
    "np_covariance": (["4000", "5000"], ["50", "60"]),
}
SET = "scalar"
if sys.argv[1:2] == ["--set"]:
    SET = sys.argv[2]; del sys.argv[1:3]
    if SET == "numpy":
        SRC, BENCH = HERE / "src_np", BENCH_NP
    elif SET != "scalar":
        sys.exit(f"unknown set {SET}")
CODON_LINK = ["-linker-flags", "-framework Accelerate"] if SET == "numpy" and platform.system() == "Darwin" else []
RTOL = 1e-5 if SET == "numpy" else 1e-6
BACKENDS = ["cpython", "cython", "cython_typed", "codon", "typhon"]
LABEL = {"cpython": "CPython 3.14", "cython": "Cython (plain .py)", "cython_typed": "Cython (typed .pyx)",
         "codon": "Codon (C++)", "typhon": "Typhon (Rust)"}
RUNS = {"cpython": (0, 3), "cython": (0, 3), "cython_typed": (1, 5), "codon": (1, 5), "typhon": (1, 5)}
SETUP_PY = """from setuptools import setup, Extension
from Cython.Build import cythonize
setup(ext_modules=cythonize([Extension({mod!r}, [{src!r}], extra_compile_args={flags!r})],
      language_level=3, force=True, quiet=True))
"""


# typed .pyx variants take these sizes from env vars instead of argv[1]
TYPED_ENV = {"float": "FLOAT_POINTS", "mandelbrot": "MANDEL_N", "spectral_norm": "SPECTRAL_LOOPS"}


def sh(cmd, cwd=None, timeout=3600, env=None):
    t = time.perf_counter()
    p = subprocess.run(cmd, cwd=cwd, env=env or ENV, capture_output=True, text=True, timeout=timeout)
    return p, time.perf_counter() - t


def uptime():
    return subprocess.run(["uptime"], capture_output=True, text=True).stdout.strip()


def has_typed(name):
    return (TYPED / f"{name}.pyx").exists()


def build(name, backend):
    """Build artifact for (name, backend). Returns (seconds, run_cmd_factory) or raises."""
    d = SCRATCH / backend / name
    d.mkdir(parents=True, exist_ok=True)
    if backend == "cpython":
        return 0.0
    if backend in ("codon", "typhon"):
        exe = CODON if backend == "codon" else TYPHON_BIN
        p, dt = sh([str(exe), "build", "-release", "-exe"] + (CODON_LINK if backend == "codon" else []) + ["-o", str(d / name), str(SRC / f"{name}.py")])
    else:
        src = SRC / f"{name}.py" if backend == "cython" else TYPED / f"{name}.pyx"
        if backend == "cython" and (SRC / f"{name}_cy.py").exists():
            src = SRC / f"{name}_cy.py"  # annotation-only variant for syntax Cython rejects
        dst = d / f"{name}{src.suffix}"
        shutil.copy(src, dst); src = dst
        for f in d.glob(f"{name}*.so"):
            f.unlink()
        (d / "setup.py").write_text(SETUP_PY.format(mod=name, src=src.name, flags=['-O3'] + (['-ffp-contract=off'] if backend == 'cython_typed' else [])))
        p, dt = sh([str(PY), "setup.py", "build_ext", "--inplace"], cwd=d)
    if p.returncode != 0:
        raise RuntimeError(f"build {backend}/{name} failed:\n{p.stdout[-2000:]}\n{p.stderr[-3000:]}")
    return dt


def run_cmd(name, backend, args):
    d = SCRATCH / backend / name
    args = [a.replace("{out}", str(d / "out.ppm")) for a in args]
    if backend in ("codon", "typhon"):
        return [str(d / name)] + args, d
    if backend == "cpython":
        return [str(PY), str(SRC / f"{name}.py")] + args, d
    drv = f"import sys; sys.argv={[name + '.py'] + args!r}; sys.path.insert(0, {str(d)!r}); import {name}"
    return [str(PY), "-c", drv], d


def run(name, backend, args):
    cmd, d = run_cmd(name, backend, args)
    env = dict(ENV)
    if backend == "cython_typed" and name in TYPED_ENV and args:
        env[TYPED_ENV[name]] = args[0]
    p, wall = sh(cmd, cwd=d, env=env)
    if p.returncode != 0:
        raise RuntimeError(f"run {backend}/{name} rc={p.returncode}: {p.stderr[-2000:]}")
    lines = p.stdout.strip().splitlines()
    internal = float(lines[-1])  # every program prints its timer last
    out = lines[:-1]
    ppm = d / "out.ppm"
    if ppm.exists():
        import hashlib
        out.append("ppm-sha256 " + hashlib.sha256(ppm.read_bytes()).hexdigest()[:16])
        ppm.unlink()
    return internal, wall, out


def same(a, b, rtol=None):
    rtol = rtol or RTOL
    if len(a) != len(b):
        return False
    for x, y in zip(a, b):
        tx, ty = x.split(), y.split()
        if len(tx) != len(ty):
            return False
        for u, v in zip(tx, ty):
            if u == v:
                continue
            try:
                fu, fv = float(u), float(v)
            except ValueError:
                return False
            if not math.isclose(fu, fv, rel_tol=rtol, abs_tol=1e-12):
                return False
    return True


def backends_for(name):
    return [b for b in BACKENDS if b != "cython_typed" or has_typed(name)]


def check(names):
    bad = []
    for name in names:
        ref = None
        for b in backends_for(name):
            try:
                build(name, b)
                _, _, out = run(name, b, BENCH[name][1])
            except Exception as e:
                print(f"FAIL {name:14} {b:13} {str(e)[:300]}")
                bad.append((name, b, "error"))
                continue
            if ref is None:
                ref = out
            ok = same(ref, out)
            print(f"{'ok  ' if ok else 'DIFF'} {name:14} {b:13} {' | '.join(out)[:90]}")
            if not ok:
                bad.append((name, b, out))
    print("mismatches:", bad or "none")
    return bad


def env_info():
    q = lambda c: subprocess.run(c, capture_output=True, text=True, env=ENV).stdout.strip()
    return {
        "date": datetime.datetime.now().isoformat(timespec="seconds"),
        "cpu": q(["sysctl", "-n", "machdep.cpu.brand_string"]),
        "ncpu": q(["sysctl", "-n", "hw.ncpu"]),
        "memsize": q(["sysctl", "-n", "hw.memsize"]),
        "macos": q(["sw_vers", "-productVersion"]),
        "set": SET,
        "numpy": q([str(PY), "-c", "import numpy; print(numpy.__version__)"]) if SET == "numpy" else None,
        "python": q([str(PY), "-c", "import sys; print(sys.version.split()[0])"]),
        "cython": q([str(PY), "-c", "import Cython; print(Cython.__version__)"]),
        "cc": q(["cc", "--version"]).splitlines()[0],
        "codon": q([str(CODON), "--version"]),
        "typhon": q([str(TYPHON_BIN), "--version"]),
        "git": q(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"]),
        "args": {k: v[0] for k, v in BENCH.items()},
    }


def measure(names, out_json):
    data = {"env": env_info(), "bench": {}}
    if out_json.exists():
        data = json.loads(out_json.read_text())  # resume
    for bi, name in enumerate(names):
        if name in data["bench"]:
            continue
        bs = backends_for(name)
        k = bi % len(bs)
        order = bs[k:] + bs[:k]  # rotate start backend per benchmark
        rec = {"uptime_before": uptime(), "order": order, "compile": {}, "runs": {}}
        print(f"== {name} order={order}\n   {rec['uptime_before']}", flush=True)
        for b in order:  # compile times: 3 builds each
            if b != "cpython":
                rec["compile"][b] = [build(name, b) for _ in range(3)]
        args = BENCH[name][0]
        warm = {b: RUNS[b][0] for b in order}
        left = {b: RUNS[b][1] for b in order}
        rec["runs"] = {b: {"internal": [], "wall": []} for b in order}
        rec["output"] = {}
        # interleave: round-robin one run per backend per round
        while any(warm.values()) or any(left.values()):
            for b in order:
                if warm[b]:
                    run(name, b, args); warm[b] -= 1
                elif left[b]:
                    i, w, out = run(name, b, args); left[b] -= 1
                    rec["runs"][b]["internal"].append(i); rec["runs"][b]["wall"].append(w)
                    rec["output"][b] = out
                    print(f"   {b:13} internal={i:8.3f}s wall={w:8.3f}s", flush=True)
        ref = rec["output"].get("cpython")
        rec["match"] = {b: same(ref, o) for b, o in rec["output"].items()} if ref else {}
        rec["uptime_after"] = uptime()
        data["bench"][name] = rec
        out_json.write_text(json.dumps(data, indent=1))
    return data


def gmean(xs):
    xs = [x for x in xs if x]
    return math.exp(sum(map(math.log, xs)) / len(xs)) if xs else float("nan")


def report(data):
    B = [b for b in BACKENDS if any(b in r["runs"] for r in data["bench"].values())]
    names = list(data["bench"])
    med = lambda r, b, k="internal": statistics.median(r["runs"][b][k]) if b in r["runs"] else None
    mn = lambda r, b: min(r["runs"][b]["internal"]) if b in r["runs"] else None
    f = lambda x, p=3: "n/a" if x is None else f"{x:.{p}f}"
    e = data["env"]
    L = [f"# Typhon benchmark comparison ({e['date']})", "",
         f"Machine: {e['cpu']}, {e['ncpu']} logical CPUs, {int(e['memsize'])//2**30} GiB, macOS {e['macos']}. "
         f"Python {e['python']}, Cython {e['cython']}, {e['cc']}, codon {e['codon']}, typhon {e['typhon']}, git {e['git']}.", "",
         "Times are the program's internal timer (same region for every backend). "
         "Median of 5 runs after 1 warm-up for Codon/Typhon/typed Cython, median of 3 for CPython/plain Cython. "
         "Machine was shared and loaded, see load averages below.", "",
         "## Runtime, median seconds (min in parentheses)", "",
         "| benchmark | args | " + " | ".join(LABEL[b] for b in B) + " |", "|---|---|" + "---|" * len(B)]
    for n in names:
        r = data["bench"][n]
        L.append(f"| {n} | {' '.join(e['args'][n]) or '-'} | " + " | ".join(
            f"{f(med(r, b))} ({f(mn(r, b))})" if b in r["runs"] else "n/a" for b in B) + " |")
    L += ["", "## Speedup vs CPython (median / median)", "", "| benchmark | " + " | ".join(LABEL[b] for b in B[1:]) + " |", "|---|" + "---|" * (len(B) - 1)]
    sp = {b: [] for b in B}
    for n in names:
        r = data["bench"][n]; c = med(r, "cpython")
        row = []
        for b in B[1:]:
            v = c / med(r, b) if b in r["runs"] else None
            if v: sp[b].append(v)
            row.append(f(v, 1) + "x" if v else "n/a")
        L.append(f"| {n} | " + " | ".join(row) + " |")
    L.append("| **geomean** | " + " | ".join(f"**{gmean(sp[b]):.1f}x**" + ("" if len(sp[b]) == len(names) else f" ({len(sp[b])} benches)") for b in B[1:]) + " |")
    L += ["", "## Typhon / Codon ratio (>1 = Typhon slower)", "", "| benchmark | runtime (internal) | process wall | compile time |", "|---|---|---|---|"]
    rr, rw, rc = [], [], []
    for n in names:
        r = data["bench"][n]
        a = med(r, "typhon") / med(r, "codon"); w = med(r, "typhon", "wall") / med(r, "codon", "wall")
        c = statistics.median(r["compile"]["typhon"]) / statistics.median(r["compile"]["codon"])
        rr.append(a); rw.append(w); rc.append(c)
        L.append(f"| {n} | {a:.3f} | {w:.3f} | {c:.3f} |")
    L.append(f"| **geomean** | **{gmean(rr):.3f}** | **{gmean(rw):.3f}** | **{gmean(rc):.3f}** |")
    CB = [b for b in B if b != "cpython"]
    L += ["", "## Compile time, median of 3 builds (s)", "", "Codon/Typhon: `build -release -exe`. Cython: cythonize + clang -O3 + link (setup.py build_ext).", "",
          "| benchmark | " + " | ".join(LABEL[b] for b in CB) + " |", "|---|" + "---|" * len(CB)]
    cg = {b: [] for b in CB}
    for n in names:
        r = data["bench"][n]; row = []
        for b in CB:
            v = statistics.median(r["compile"][b]) if b in r["compile"] else None
            if v: cg[b].append(v)
            row.append(f(v, 2))
        L.append(f"| {n} | " + " | ".join(row) + " |")
    L.append("| **geomean** | " + " | ".join(f"**{gmean(cg[b]):.2f}**" for b in CB) + " |")
    L += ["", "## Process wall time, median (s)", "", "| benchmark | " + " | ".join(LABEL[b] for b in B) + " |", "|---|" + "---|" * len(B)]
    for n in names:
        r = data["bench"][n]
        L.append(f"| {n} | " + " | ".join(f(med(r, b, "wall")) for b in B) + " |")
    L += ["", "## Output match vs CPython and load", "", "| benchmark | mismatching backends | uptime before | uptime after |", "|---|---|---|---|"]
    for n in names:
        r = data["bench"][n]; mm = [b for b, ok in r.get("match", {}).items() if not ok]
        la = lambda s: s.split("load averages:")[-1].strip()
        L.append(f"| {n} | {', '.join(mm) or 'none'} | {la(r['uptime_before'])} | {la(r['uptime_after'])} |")
    return "\n".join(L) + "\n"


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "check"
    names = sys.argv[2:] or list(BENCH)
    if cmd == "check":
        sys.exit(1 if check(names) else 0)
    if cmd == "report":
        p = Path(sys.argv[2]); p.with_suffix(".md").write_text(report(json.loads(p.read_text()))); return
    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / f"{os.environ.get('BENCH_TAG', datetime.date.today().isoformat())}{'-numpy' if SET == 'numpy' else ''}.json"
    data = measure(list(BENCH) if cmd == "measure" and not sys.argv[2:] else names, out)
    out.with_suffix(".md").write_text(report(data))
    print(out.with_suffix(".md").read_text())


if __name__ == "__main__":
    main()
