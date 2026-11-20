"""Smoke-test a relocated release bundle using only its installed layout."""

import os
from pathlib import Path
import subprocess
import sys
import tempfile


def main():
    bundle = Path(sys.argv[1]).resolve()
    suffix = ".exe" if os.name == "nt" else ""
    compiler = bundle / "bin" / f"typhon{suffix}"
    env = os.environ.copy()
    for name in ("TYPHON_STDLIB_DIR", "TYPHON_PATH", "TYPHON_LIB_DIR"):
        env.pop(name, None)
    env["PATH"] = str(bundle / "bin") + os.pathsep + env["PATH"]
    source = '''import numpy as np
import unicodedata

x = np.array([1.0, 2.0, 3.0, 4.0])
assert np.mean(x) == 2.5
assert unicodedata.normalize("NFC", "e\\u0301") == "é"
events = []
def fail():
    raise ValueError("expected")
try:
    try:
        fail()
    except ValueError:
        events.append(1)
        try:
            raise TypeError("nested")
        except TypeError:
            events.append(2)
        raise
    finally:
        events.append(3)
except ValueError:
    events.append(4)
assert events == [1, 2, 3, 4]
print("typhon release smoke OK")
'''
    with tempfile.TemporaryDirectory(prefix="typhon-smoke-") as directory:
        work = Path(directory)
        program = work / "smoke.py"
        program.write_text(source, encoding="utf-8")
        executable = work / f"smoke{suffix}"
        for command in (
            [str(compiler), "version"],
            [str(compiler), "run", "-release", str(program)],
            [str(compiler), "build", "-release", "-exe", "-o", str(executable), str(program)],
            [str(executable)],
        ):
            print("+", " ".join(command), flush=True)
            subprocess.run(command, cwd=work, env=env, check=True, timeout=180)


if __name__ == "__main__":
    main()
