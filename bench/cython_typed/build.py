"""Build typed Cython benchmarks with -O3.

Usage: python build.py [BUILD_DIR] [name ...]
BUILD_DIR defaults to $JCODE_SCRATCH_DIR/cython_typed_build (or ./build).
Extension modules land directly in BUILD_DIR; run them with run.py.
"""
import os
import sys
from setuptools import Extension
from setuptools.dist import Distribution
from Cython.Build import cythonize

HERE = os.path.dirname(os.path.abspath(__file__))
NAMES = ["binary_trees", "chaos", "fannkuch", "float", "go", "mandelbrot",
         "nbody", "primes", "set_partition", "spectral_norm"]


def main():
    args = sys.argv[1:]
    default = os.path.join(os.environ.get("JCODE_SCRATCH_DIR", HERE), "cython_typed_build")
    out = os.path.abspath(args[0]) if args else default
    names = args[1:] or NAMES
    os.makedirs(out, exist_ok=True)
    exts = [Extension(n, [os.path.join(HERE, n + ".pyx")],
                      extra_compile_args=["-O3", "-ffp-contract=off"]) for n in names]
    mods = cythonize(exts, build_dir=os.path.join(out, "c"), quiet=True,
                     compiler_directives={"language_level": 3})
    dist = Distribution({"ext_modules": mods})
    cmd = dist.get_command_obj("build_ext")
    cmd.build_lib = out
    cmd.build_temp = os.path.join(out, "tmp")
    cmd.ensure_finalized()
    cmd.run()
    print("built into", out)


if __name__ == "__main__":
    main()
