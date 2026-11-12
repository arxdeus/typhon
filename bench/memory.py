#!/usr/bin/env python3
"""Peak-memory harness: Codon (C++) vs Typhon (any binary/dylib dir) vs CPython.

Usage:
  memory.py [--typhon-dir DIR] [--label L] [--runs 3] [--cpython] [names...]
DIR must hold `typhon` and `libcodonrt.dylib` (default: target/release).
Built Typhon exes are retargeted to @rpath/libcodonrt.dylib with rpath = DIR,
so they use exactly that dylib. Metrics from `/usr/bin/time -l`.
Output: results/memory-<date>[-label].{json,md}
"""
import argparse, datetime, json, os, re, statistics, subprocess, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
TYPHON = HERE.parent
ROOT = TYPHON.parent
SRC = HERE / "src"
RESULTS = HERE / "results"
SCRATCH = Path(os.environ.get("JCODE_SCRATCH_DIR", "/tmp")) / "memaudit"
CODON = ROOT / "build/codon"
ENV = dict(os.environ, CODON_PATH=str(ROOT / "stdlib"))
PY = os.environ.get("BENCH_PYTHON", "python3")

BENCH = {
    "binary_trees": ["17"], "chaos": ["{out}"], "fannkuch": ["10"],
    "float": ["10000000"], "go": [], "mandelbrot": ["512"], "nbody": ["3000000"],
    "primes": ["30000"], "set_partition": ["14"], "spectral_norm": ["40"],
}
TRIVIAL = {"empty": "pass\n", "hello": 'print("hello")\n'}


def sh(cmd, **kw):
    return subprocess.run(cmd, capture_output=True, text=True, env=kw.pop("env", ENV), **kw)


def build(backend, name, src, tdir):
    d = SCRATCH / "exe" / backend
    d.mkdir(parents=True, exist_ok=True)
    exe = d / name
    tool = CODON if backend == "codon" else Path(tdir) / "typhon"
    p = sh([str(tool), "build", "-release", "-exe", "-o", str(exe), str(src)])
    if p.returncode:
        raise RuntimeError(f"build {backend}/{name}: {p.stderr[-1500:]}")
    if backend == "typhon":
        for l in sh(["otool", "-L", str(exe)]).stdout.splitlines()[1:]:
            m = re.match(r"\s*(\S*libcodonrt\.dylib)", l)
            if m and m.group(1) != "@rpath/libcodonrt.dylib":
                sh(["install_name_tool", "-change", m.group(1), "@rpath/libcodonrt.dylib", str(exe)])
        rp = sh(["otool", "-l", str(exe)]).stdout
        for old in re.findall(r"path (\S+) \(offset", rp):
            if old != str(Path(tdir).resolve()):
                sh(["install_name_tool", "-delete_rpath", old, str(exe)])
        if str(Path(tdir).resolve()) not in rp:
            sh(["install_name_tool", "-add_rpath", str(Path(tdir).resolve()), str(exe)])
        sh(["codesign", "-f", "-s", "-", str(exe)])
    return exe


def measure(cmd, cwd):
    p = sh(["/usr/bin/time", "-l"] + cmd, cwd=cwd)
    if p.returncode:
        raise RuntimeError(f"run {cmd} rc={p.returncode}: {p.stderr[-800:]}")
    rss = int(re.search(r"(\d+)\s+maximum resident set size", p.stderr).group(1))
    fp = int(re.search(r"(\d+)\s+peak memory footprint", p.stderr).group(1))
    return rss, fp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--typhon-dir", default=str(TYPHON / "target/release"))
    ap.add_argument("--label", default="")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--cpython", action="store_true")
    ap.add_argument("names", nargs="*")
    a = ap.parse_args()
    (SCRATCH / "src").mkdir(parents=True, exist_ok=True)
    progs = {}
    for n, text in TRIVIAL.items():
        f = SCRATCH / "src" / f"{n}.py"; f.write_text(text); progs[n] = (f, [])
    for n, args in BENCH.items():
        progs[n] = (SRC / f"{n}.py", args)
    if a.names:
        progs = {k: v for k, v in progs.items() if k in a.names}
    backends = ["codon", "typhon"] + (["cpython"] if a.cpython else [])
    res = {"uptime_start": subprocess.getoutput("uptime"), "typhon_dir": a.typhon_dir,
           "runs": a.runs, "rows": {}}
    for n, (src, args) in progs.items():
        row = res["rows"][n] = {}
        exes = {}
        for b in backends:
            try:
                exes[b] = [PY, str(src)] if b == "cpython" else [str(build(b, n, src, a.typhon_dir))]
            except Exception as e:
                row[b] = {"error": str(e)[:300]}; print(e, file=sys.stderr)
        out = SCRATCH / "out.ppm"
        argv = [x.replace("{out}", str(out)) for x in args]
        samples = {b: [] for b in exes}
        for _ in range(a.runs):          # round-robin, serial
            for b, cmd in exes.items():
                try:
                    samples[b].append(measure(cmd + argv, str(SCRATCH)))
                except Exception as e:
                    row[b] = {"error": str(e)[:300]}
        for b, s in samples.items():
            if s and "error" not in row.get(b, {}):
                row[b] = {"rss": int(statistics.median(x[0] for x in s)),
                          "footprint": int(statistics.median(x[1] for x in s)),
                          "rss_all": [x[0] for x in s]}
        row["uptime"] = subprocess.getoutput("uptime")
        print(n, {b: row[b].get("rss") for b in backends if b in row}, flush=True)
    day = datetime.date.today().isoformat() + (f"-{a.label}" if a.label else "")
    (RESULTS / f"memory-{day}.json").write_text(json.dumps(res, indent=1))
    mb = lambda v: f"{v/1048576:.1f}" if v else "-"
    L = [f"# Memory (peak RSS / peak footprint, MiB, median of {a.runs})", "",
         f"Typhon dir: `{a.typhon_dir}`  ", f"uptime: {res['uptime_start']}", "",
         "| program | Codon RSS | Typhon RSS | ratio | Codon fp | Typhon fp |" + (" CPython RSS |" if a.cpython else ""),
         "|---|---|---|---|---|---|" + ("---|" if a.cpython else "")]
    for n, row in res["rows"].items():
        c, t = row.get("codon", {}), row.get("typhon", {})
        r = f"{t['rss']/c['rss']:.2f}" if c.get("rss") and t.get("rss") else "-"
        L.append(f"| {n} | {mb(c.get('rss'))} | {mb(t.get('rss'))} | {r} | {mb(c.get('footprint'))} | {mb(t.get('footprint'))} |"
                 + (f" {mb(row.get('cpython', {}).get('rss'))} |" if a.cpython else ""))
    (RESULTS / f"memory-{day}.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


main()
