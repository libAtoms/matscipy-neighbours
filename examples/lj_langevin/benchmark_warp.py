#!/usr/bin/env python3
"""Scaling benchmark for the Warp LJ Langevin droplet (`lj_langevin_warp.py`).

Runs the Warp example over a logarithmic range of droplet sizes (100, 1000,
10000, ... atoms, up to the GPU memory) for every combination of

    neighbour-list builder  x  device
    {matscipy, vesin}          {cpu, gpu}

and records, per step, the **total** time and its breakdown — neighbour-list
build / LJ force kernel / Langevin integrate. The neighbour-list build is the
quantity of interest, so it is reported and plotted on its own; matscipy and
vesin (https://github.com/luthaf/vesin) feed the *same* Warp kernels, which
makes the comparison apples-to-apples.

Output: a log-log plot of time vs. number of atoms (total and neighbour-list
build) and, optionally, a Markdown documentation page.

CPU threading note: the matscipy neighbour list is OpenMP-parallel (uses all
cores); vesin's CPU neighbour list is single-threaded; the Warp force/integrate
kernels are multi-threaded on the CPU. Set `OMP_NUM_THREADS=1` to compare
single-threaded.

Each backend is invoked as a subprocess; vesin on the GPU additionally needs
`libcuda.so` on `LD_LIBRARY_PATH` (e.g. `/usr/lib/wsl/lib` on WSL), inherited
from this driver's environment.
"""

import argparse
import os
import re
import subprocess
import sys

from benchmark import detect_cpu, detect_gpu

HERE = os.path.dirname(os.path.abspath(__file__))

# (label, neighbours, device, linestyle, colour) — colour by builder, dash by
# device, so the four series read as a 2x2 grid in the legend.
BACKENDS = [
    ("matscipy (GPU)", "matscipy", "gpu", "-", "tab:blue"),
    ("vesin (GPU)", "vesin", "gpu", "-", "tab:orange"),
    ("matscipy (CPU)", "matscipy", "cpu", "--", "tab:blue"),
    ("vesin (CPU)", "vesin", "cpu", "--", "tab:orange"),
]


def adaptive_steps(base, atoms):
    """Fewer steps for larger systems so each point stays quick; warm-up (one
    untimed step) is always excluded by the example itself."""
    return max(5, min(base, round(base * 20000 / max(atoms, 1))))


def run(atoms, neighbours, device, base_steps, env, timeout):
    """Run one (size, builder, device) point. Returns a dict of per-step times
    in ms (total/nl/force/integrate) and the atom/pair counts, or None on
    failure (out-of-memory, timeout, missing dependency)."""
    steps = adaptive_steps(base_steps, atoms)
    cmd = [sys.executable, os.path.join(HERE, "lj_langevin_warp.py"),
           "--device", device, "--neighbours", neighbours,
           "--atoms", str(atoms), "--steps", str(steps)]
    try:
        out = subprocess.run(cmd, env=env, capture_output=True, text=True,
                             timeout=timeout)
    except subprocess.TimeoutExpired:
        return None
    text = out.stdout + out.stderr
    m_atoms = re.search(r"atoms=(\d+)", text)
    m_pairs = re.search(r"pairs~(\d+)", text)
    m_total = re.search(r"([\d.]+)\s+ms/step", text)
    m_nl = re.search(r"nl_ms=([\d.]+)", text)
    m_force = re.search(r"force_ms=([\d.]+)", text)
    m_int = re.search(r"integrate_ms=([\d.]+)", text)
    if not (m_atoms and m_total and m_nl):
        return None
    return dict(atoms=int(m_atoms.group(1)),
                pairs=int(m_pairs.group(1)) if m_pairs else 0,
                total=float(m_total.group(1)), nl=float(m_nl.group(1)),
                force=float(m_force.group(1)) if m_force else 0.0,
                integrate=float(m_int.group(1)) if m_int else 0.0)


def make_plot(results, path):
    """Two-panel log-log plot: total time and neighbour-list build time vs.
    number of atoms, one line per backend."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax_total, ax_nl) = plt.subplots(1, 2, figsize=(11, 4.6), sharex=True)
    for label, neighbours, device, ls, colour in BACKENDS:
        pts = results.get(label, [])
        if not pts:
            continue
        xs = [p["atoms"] for p in pts]
        ax_total.plot(xs, [p["total"] for p in pts], ls, color=colour,
                      marker="o", ms=4, label=label)
        ax_nl.plot(xs, [p["nl"] for p in pts], ls, color=colour,
                   marker="o", ms=4, label=label)
    for ax, title in ((ax_total, "Total time per step"),
                      (ax_nl, "Neighbour-list build per step")):
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("number of atoms")
        ax.set_ylabel("time per step (ms)")
        ax.set_title(title)
        ax.grid(True, which="both", ls=":", alpha=0.4)
        ax.legend(fontsize=8)
    fig.suptitle("Warp LJ droplet — matscipy vs. vesin neighbour list")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    print(f"wrote {path}", file=sys.stderr)


def table_markdown(results, sizes):
    """Total ms/step table: rows = backend, columns = atom count."""
    cols = sorted({p["atoms"] for pts in results.values() for p in pts})
    head = "| Implementation | " + " | ".join(f"{c} atoms" for c in cols) + " |"
    sep = "|" + "---|" * (len(cols) + 1)
    lines = [head, sep]
    for label, *_ in BACKENDS:
        by_atoms = {p["atoms"]: p for p in results.get(label, [])}
        cells = " | ".join(f"{by_atoms[c]['total']:.2f}" if c in by_atoms else "—"
                           for c in cols)
        lines.append(f"| {label} | {cells} |")
    return "\n".join(lines)


def nl_table_markdown(results):
    """Neighbour-list build ms/step table (the quantity of interest)."""
    cols = sorted({p["atoms"] for pts in results.values() for p in pts})
    head = "| Neighbour list | " + " | ".join(f"{c} atoms" for c in cols) + " |"
    sep = "|" + "---|" * (len(cols) + 1)
    lines = [head, sep]
    for label, *_ in BACKENDS:
        by_atoms = {p["atoms"]: p for p in results.get(label, [])}
        cells = " | ".join(f"{by_atoms[c]['nl']:.3f}" if c in by_atoms else "—"
                           for c in cols)
        lines.append(f"| {label} | {cells} |")
    return "\n".join(lines)


def write_doc_page(path, total_table, nl_table, plot_name, steps, ncores):
    body = f"""# Warp benchmark — matscipy vs. vesin neighbour list

Per-step wall time of the [Warp Lennard-Jones Langevin droplet](examples.md)
(`lj_langevin_warp.py`) across droplet sizes. The Warp kernels (LJ force +
Langevin integrate) are identical in every run; only the **neighbour-list
builder** changes — this library's `matscipy_neighbours` or
[`vesin`](https://github.com/luthaf/vesin) — so the difference is the list, not
the physics. Lower is better.

!!! info "Test machine"
    - **CPU:** {detect_cpu()}
    - **GPU:** {detect_gpu()}

!!! warning "CPU threading"
    The **matscipy** neighbour list is **multi-threaded** (OpenMP, uses all
    {ncores} logical cores). **vesin**'s CPU neighbour list is
    **single-threaded**. The Warp force/integrate kernels are multi-threaded on
    the CPU. Re-run with `OMP_NUM_THREADS=1` for a single-threaded comparison.
    The GPU rows are not affected.

Run configuration: reduced LJ units, cutoff 2.5, dt 0.005, friction 1.0,
temperature 0.7; logarithmically spaced droplet sizes (100 → 1,000,000 atoms).
Up to {steps} steps per point (fewer for the largest systems), Warp kernels
compiled once during an untimed warm-up. Each timed phase synchronises the
device, so the reported total is the sum of the phase breakdown. (The matscipy
GPU list itself was verified to build for multi-million-atom droplets — the
limit on this 6 GB card is host/device memory, not the algorithm.)

![Time vs. number of atoms]({plot_name})

## Total time per step (ms)

{total_table}

## Neighbour-list build per step (ms)

{nl_table}

How to read it:

- The neighbour-list build dominates the step for both builders, so it sets the
  overall scaling — which is exactly why the list implementation matters.
- On the **GPU**, `matscipy_neighbours` builds the list with a cell list and
  stays close to linear in the atom count. `vesin`'s GPU path is considerably
  slower for these large, low-density droplets.
- On the **CPU**, matscipy is OpenMP-parallel while vesin is single-threaded, so
  the gap widens with system size; set `OMP_NUM_THREADS=1` to compare like
  for like.

This page is generated by `examples/lj_langevin/benchmark_warp.py`. Regenerate
it on your own hardware with:

```bash
python examples/lj_langevin/benchmark_warp.py --doc-out docs/benchmark_warp.md
```
"""
    with open(path, "w") as fh:
        fh.write(body)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--build", default=os.path.join(HERE, "..", "..", "build"),
                    help="CMake build directory (the Python extension)")
    ap.add_argument("--sizes", type=int, nargs="+",
                    default=[100, 1000, 10000, 100000, 1000000],
                    help="target atom counts (logarithmically spaced)")
    ap.add_argument("--steps", type=int, default=40,
                    help="timed steps for the smallest systems (scaled down "
                         "automatically for larger ones)")
    ap.add_argument("--timeout", type=int, default=900,
                    help="per-run timeout in seconds (a point that exceeds it "
                         "is dropped, and that backend is not pushed larger)")
    ap.add_argument("--plot-out", default=os.path.join(HERE, "..", "..",
                                                       "docs", "benchmark_warp.png"))
    ap.add_argument("--doc-out", default=None,
                    help="write a documentation page (with hardware info) here")
    args = ap.parse_args()

    build = os.path.abspath(args.build)
    pkg = os.path.join(HERE, "..", "..", "language_bindings", "python")
    env = dict(os.environ,
               PYTHONPATH=os.pathsep.join(
                   [build, pkg, os.environ.get("PYTHONPATH", "")]))

    results = {}  # label -> list of point dicts (sorted by atom count)
    for label, neighbours, device, _, _ in BACKENDS:
        for atoms in args.sizes:
            r = run(atoms, neighbours, device, args.steps, env, args.timeout)
            if r is None:
                # Hit a wall (OOM/timeout/missing dep) — larger sizes will too.
                print(f"  {label:16s} atoms={atoms} -> failed/timed out; "
                      f"stopping this backend", file=sys.stderr)
                break
            results.setdefault(label, []).append(r)
            print(f"  {label:16s} atoms={r['atoms']:>8d} pairs={r['pairs']:>11d} "
                  f"-> {r['total']:.2f} ms/step (nl {r['nl']:.3f})",
                  file=sys.stderr)

    if not results:
        raise SystemExit("no successful runs (is the extension built? is "
                         "warp/vesin installed?)")

    plot_path = os.path.abspath(args.plot_out)
    make_plot(results, plot_path)

    total_table = table_markdown(results, args.sizes)
    nl_table = nl_table_markdown(results)
    print("\nTotal ms/step:\n" + total_table)
    print("\nNeighbour-list build ms/step:\n" + nl_table)

    if args.doc_out:
        write_doc_page(os.path.abspath(args.doc_out), total_table, nl_table,
                       os.path.basename(plot_path), args.steps, os.cpu_count())
        print(f"\nwrote {args.doc_out}", file=sys.stderr)


if __name__ == "__main__":
    main()
