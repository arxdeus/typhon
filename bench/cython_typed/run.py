"""Run a built typed-Cython benchmark: python run.py BUILD_DIR NAME [args...]

Each module executes its benchmark at import time (module-level code, same
as the .py original), reading sys.argv[1:] like the original script.
"""
import importlib
import sys

build_dir, name = sys.argv[1], sys.argv[2]
sys.path.insert(0, build_dir)
sys.argv = [name] + sys.argv[3:]
importlib.import_module(name)
