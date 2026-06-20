#!/usr/bin/env python3
"""Scaling benchmark for the LJ Langevin droplet examples.

Runs each implementation (NumPy/CuPy Python, JAX Python, C++ CPU/GPU) over a
range of droplet sizes and reports the per-step wall time, so one can see how
performance develops with system size. Each backend is invoked as a subprocess
(its own process/runtime) and the printed ``ms/step`` is parsed.
"""

import argparse
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def run(cmd, env=None):
    """Run a command, return (atoms, pairs, ms_per_step) parsed from stdout."""
    try:
        out = subprocess.run(cmd, env=env, capture_output=True, text=True,
                             timeout=1800)
    except subprocess.TimeoutExpired:
        return None
    text = out.stdout + out.stderr
    m_atoms = re.search(r"atoms=(\d+)", text)
    m_pairs = re.search(r"pairs~(\d+)", text)
    m_ms = re.search(r"([\d.]+)\s+ms/step", text)
    if not (m_atoms and m_ms):
        return None
    pairs = int(m_pairs.group(1)) if m_pairs else 0
    return int(m_atoms.group(1)), pairs, float(m_ms.group(1))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--build", default=os.path.join(HERE, "..", "..", "build"),
                    help="CMake build directory (for the C++ binaries and the "
                         "Python extension)")
    ap.add_argument("--sizes", type=int, nargs="+", default=[3, 4, 5, 6, 8])
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--jax-python", default=None,
                    help="Python interpreter with JAX (enables the JAX rows)")
    ap.add_argument("--jax-steps", type=int, default=6,
                    help="Few steps for JAX, which recompiles each step")
    args = ap.parse_args()

    build = os.path.abspath(args.build)
    exe = os.path.join(build, "examples", "lj_langevin")
    pkg = os.path.join(HERE, "..", "..", "language_bindings", "python")
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([build, pkg]))
    pyscript = os.path.join(HERE, "lj_langevin.py")
    jaxscript = os.path.join(HERE, "lj_langevin_jax.py")

    def common(ncells, steps):
        return ["--ncells", str(ncells), "--steps", str(steps),
                "--write-every", str(steps + 1), "--out", os.devnull]

    backends = [
        ("numpy (CPU)", lambda nc: ([sys.executable, pyscript, "--device", "cpu"]
                                    + common(nc, args.steps), env)),
        ("cupy (GPU)", lambda nc: ([sys.executable, pyscript, "--device", "gpu"]
                                   + common(nc, args.steps), env)),
        ("C++ (CPU)", lambda nc: ([os.path.join(exe, "lj_langevin_cpu")]
                                  + common(nc, args.steps), env)),
        ("C++ (GPU)", lambda nc: ([os.path.join(exe, "lj_langevin_gpu")]
                                  + common(nc, args.steps), env)),
    ]
    if args.jax_python:
        jcpu = dict(env, JAX_PLATFORMS="cpu")
        backends.append(
            ("JAX (CPU)", lambda nc: ([args.jax_python, jaxscript, "--device",
                                       "cpu"] + common(nc, args.jax_steps), jcpu)))
        # GPU JAX inherits the parent env (set LD_LIBRARY_PATH for the bundled
        # CUDA libs before running this driver).
        backends.append(
            ("JAX (GPU)", lambda nc: ([args.jax_python, jaxscript, "--device",
                                       "gpu"] + common(nc, args.jax_steps), env)))

    results = {}     # name -> {atoms: ms}
    atoms_for = {}   # ncells -> atoms
    for nc in args.sizes:
        for name, build_cmd in backends:
            if not os.path.exists(build_cmd(nc)[0][0]):
                continue
            r = run(*build_cmd(nc))
            if r is None:
                continue
            atoms, pairs, ms = r
            atoms_for[nc] = atoms
            results.setdefault(name, {})[nc] = ms
            print(f"  {name:14s} ncells={nc} atoms={atoms} -> {ms:.2f} ms/step",
                  file=sys.stderr)

    # Markdown table: rows = backend, columns = atom count.
    cols = [nc for nc in args.sizes if nc in atoms_for]
    header = "| Implementation | " + " | ".join(
        f"{atoms_for[nc]} atoms" for nc in cols) + " |"
    sep = "|" + "---|" * (len(cols) + 1)
    print("\n" + header)
    print(sep)
    for name in [b[0] for b in backends]:
        if name not in results:
            continue
        cells = " | ".join(
            (f"{results[name][nc]:.2f}" if nc in results[name] else "—")
            for nc in cols)
        print(f"| {name} | {cells} |")
    print("\n(values are ms/step)")


if __name__ == "__main__":
    main()
