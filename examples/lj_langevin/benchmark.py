#!/usr/bin/env python3
"""Unified scaling benchmark for the LJ Langevin examples.

Runs the per-step wall time across a logarithmic range of sizes for two
systems -- the self-bound **droplet** in vacuum (non-periodic, sparse grid)
and the bulk **liquid** in a periodic box (dense grid, periodic images) -- for
the full cross-product of three dimensions:

    device   : CPU / GPU
    list     : matscipy (this library) / vesin (https://github.com/luthaf/vesin)
    kernels  : Warp / array (NumPy or CuPy) / JAX / C++

Not every combination exists: JAX uses the dense `neighbour_matrix` and the C++
example uses the in-tree C++ core, so neither can be driven by vesin — those
cells are left empty. On the CPU the matscipy neighbour list is run **both
single-threaded** (`OMP_NUM_THREADS=1`) **and multi-threaded** (all cores); the
threading controls the matscipy list (and the C++ OpenMP force loop). vesin's
CPU list is single-threaded.

Each configuration is launched as a subprocess and its printed `ms/step` is
parsed. Output: a console table per system, a JSON file of raw timings, and
per system a log-log plot of time vs. number of atoms, faceted by kernel.
"""

import argparse
import json
import os
import platform
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def usable_cores():
    """Cores this process may run on (respects cgroup/affinity limits, e.g. a
    batch-job allocation), falling back to the machine's logical core count."""
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count()

SYSTEM_ORDER = ["droplet", "liquid"]
SYSTEM_NAME = {"droplet": "LJ droplet (vacuum, non-periodic)",
               "liquid": "LJ liquid (periodic box)"}

KERNEL_NAME = {"warp": "Warp", "array": "array (NumPy/CuPy)", "jax": "JAX",
               "cpp": "C++"}
KERNEL_ORDER = ["warp", "array", "jax", "cpp"]

# Neighbour-list backends. "matscipy" is this library (matscipy_neighbours);
# "matscipy-classic" is the classic matscipy 1.2.0 package (CPU only); "vesin"
# is https://github.com/luthaf/vesin.
NL_ORDER = ["matscipy", "matscipy-classic", "vesin"]
NL_DISPLAY = {"matscipy": "matscipy-neighbours",
              "matscipy-classic": "matscipy 1.2.0", "vesin": "vesin"}

# Plot style per (device, list, threads): colour by list, dash by device, with
# the single-threaded matscipy CPU line dotted. Shared across all facets.
STYLE = {
    ("gpu", "matscipy", None): dict(c="tab:blue", ls="-", m="o",
                                    label="GPU · matscipy-neighbours"),
    ("cpu", "matscipy", "mt"): dict(c="tab:blue", ls="--", m="o",
                                    label="CPU · matscipy-neighbours (mt)"),
    ("cpu", "matscipy", "1t"): dict(c="tab:blue", ls=":", m="x",
                                    label="CPU · matscipy-neighbours (1t)"),
    ("cpu", "matscipy-classic", None): dict(c="tab:green", ls="--", m="^",
                                            label="CPU · matscipy 1.2.0"),
    ("gpu", "vesin", None): dict(c="tab:orange", ls="-", m="s",
                                 label="GPU · vesin"),
    ("cpu", "vesin", None): dict(c="tab:orange", ls="--", m="s",
                                 label="CPU · vesin"),
}


def detect_cpu():
    model = platform.processor() or "unknown CPU"
    try:
        with open("/proc/cpuinfo") as fh:
            for line in fh:
                if line.startswith("model name"):
                    model = line.split(":", 1)[1].strip()
                    break
    except OSError:
        pass
    return f"{model} ({usable_cores()} usable cores)"


def detect_gpu():
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=30)
        names = [n.strip() for n in out.stdout.splitlines() if n.strip()]
    except (OSError, subprocess.SubprocessError):
        names = []
    if not names:
        return "no NVIDIA GPU detected"
    uniq = sorted(set(names))
    if len(names) > 1 and len(uniq) == 1:
        return f"{len(names)}x {uniq[0]}"
    return ", ".join(names)


def nl_devices(nl):
    """Devices a neighbour-list backend can run on (classic matscipy is CPU)."""
    return ["cpu"] if nl == "matscipy-classic" else ["gpu", "cpu"]


def make_configs():
    """The full matrix. vesin and matscipy 1.2.0 only feed the Warp and array
    kernels (JAX needs the dense matrix, C++ uses the in-tree core), so those
    (kernel, list) cells are kept but marked unsupported -> empty in the table.
    Only the matscipy (this library) CPU list is split into single/multi-thread."""
    cfgs = []
    for kernel in KERNEL_ORDER:
        for nl in NL_ORDER:
            supported = nl == "matscipy" or kernel in ("warp", "array")
            for device in nl_devices(nl):
                if device == "cpu" and nl == "matscipy":
                    threads_list = ["mt", "1t"]
                else:
                    threads_list = [None]
                for threads in threads_list:
                    cfgs.append(dict(kernel=kernel, nl=nl, device=device,
                                     threads=threads, supported=supported))
    return cfgs


def label(cfg):
    thr = {"mt": " (mt)", "1t": " (1t)", None: ""}[cfg["threads"]]
    dev = cfg["device"].upper()
    return f"{KERNEL_NAME[cfg['kernel']]} · {NL_DISPLAY[cfg['nl']]} · {dev}{thr}"


def adaptive_steps(base, atoms):
    """Fewer steps for larger systems so each point stays quick."""
    return max(5, min(base, round(base * 20000 / max(atoms, 1))))


def build_command(cfg, system, density, atoms, steps, build, base_env):
    """Return (cmd, env) for one configuration, or None if its binary/script is
    missing."""
    env = dict(base_env)
    if cfg["device"] == "cpu" and cfg["nl"] == "matscipy":
        env["OMP_NUM_THREADS"] = "1" if cfg["threads"] == "1t" else \
            str(usable_cores())
    common = ["--system", system, "--density", str(density),
              "--atoms", str(atoms), "--steps", str(steps),
              "--write-every", str(steps + 1), "--out", os.devnull]
    kernel = cfg["kernel"]
    if kernel == "warp":
        return [sys.executable, os.path.join(HERE, "lj_langevin_warp.py"),
                "--device", cfg["device"], "--neighbours", cfg["nl"],
                "--system", system, "--density", str(density),
                "--atoms", str(atoms), "--steps", str(steps)], env
    if kernel == "array":
        return [sys.executable, os.path.join(HERE, "lj_langevin.py"),
                "--device", cfg["device"], "--neighbours", cfg["nl"]] + common, env
    if kernel == "jax":
        if cfg["device"] == "cpu":
            env["JAX_PLATFORMS"] = "cpu"
        return [sys.executable, os.path.join(HERE, "lj_langevin_jax.py"),
                "--device", cfg["device"]] + common, env
    if kernel == "cpp":
        exe = os.path.join(build, "examples", "lj_langevin",
                           f"lj_langevin_{cfg['device']}")
        if not os.path.exists(exe):
            return None
        return [exe] + common, env
    return None


def run(cfg, system, density, atoms, base_steps, build, base_env, timeout):
    """Run one point; return per-step ms (float) or None on failure."""
    steps = adaptive_steps(base_steps, atoms)
    built = build_command(cfg, system, density, atoms, steps, build, base_env)
    if built is None:
        return None
    cmd, env = built
    try:
        out = subprocess.run(cmd, env=env, capture_output=True, text=True,
                             timeout=timeout)
    except subprocess.TimeoutExpired:
        return None
    text = out.stdout + out.stderr
    m_ms = re.search(r"([\d.]+)\s+ms/step", text)
    return float(m_ms.group(1)) if m_ms else None


# A point whose per-step time grows far faster than the atom count is not a
# kernel time but a memory-thrash artifact: when a working set no longer fits in
# GPU memory, WSL silently spills it to host RAM and the run limps to the end
# (it never cleanly OOMs). Healthy near-linear scaling grows at most ~2.5x per
# 10x atoms here; this catches the ~90x blow-ups (e.g. the JAX dense matrix at
# 1M on a 6 GB card) without touching genuinely-slow-but-real points.
THRASH_GROWTH = 6.0


def predicted_run_seconds(points, atoms, base_steps):
    """Timed-loop wall time of the next size, extrapolated linearly in the atom
    count from the last measured point (None if nothing is measured yet)."""
    if not points:
        return None
    pa, pm = points[-1]
    return pm * (atoms / pa) * adaptive_steps(base_steps, atoms) / 1000


def within_budget(points, atoms, base_steps, budget):
    predicted = predicted_run_seconds(points, atoms, base_steps)
    return budget is None or predicted is None or predicted <= budget


def make_plot(cfgs, sizes, path, system):
    """2x2 log-log facets (one per kernel): time per step vs. number of atoms."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5), sharex=True, sharey=True)
    panels = dict(zip(KERNEL_ORDER, axes.flat))
    for cfg in cfgs:
        pts = cfg.get("points")
        if not pts:
            continue
        st = STYLE[(cfg["device"], cfg["nl"], cfg["threads"])]
        ax = panels[cfg["kernel"]]
        xs = [a for a, _ in pts]
        ys = [t for _, t in pts]
        ax.plot(xs, ys, color=st["c"], ls=st["ls"], marker=st["m"], ms=5,
                label=st["label"])
    for kernel, ax in panels.items():
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(KERNEL_NAME[kernel] + " kernels")
        ax.grid(True, which="both", ls=":", alpha=0.4)
        ax.set_xlabel("number of atoms")
        ax.set_ylabel("time per step (ms)")
        if ax.has_data():
            ax.legend(fontsize=8)
    fig.suptitle(f"{SYSTEM_NAME[system]} — time vs. number of atoms "
                 "(device × neighbour list × kernels)", fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    print(f"wrote {path}", file=sys.stderr)


def table_markdown(cfgs, sizes):
    cols = sizes
    head = "| Configuration | " + " | ".join(f"{c} atoms" for c in cols) + " |"
    sep = "|" + "---|" * (len(cols) + 1)
    lines = [head, sep]
    for cfg in cfgs:
        pts = dict(cfg.get("points", []))
        cells = " | ".join(f"{pts[c]:.2f}" if c in pts else "—" for c in cols)
        lines.append(f"| {label(cfg)} | {cells} |")
    return "\n".join(lines)


def write_doc_page(path, plot_names, meta):
    """Write the documentation page. ``plot_names`` maps system -> image file
    name (relative to the page)."""
    sizes = meta["sizes"]
    cpu_cap = meta.get("max_atoms_cpu")
    cpu_range = (f"\nCPU runs stop at {cpu_cap:,} atoms; the GPU runs cover the "
                 f"full range." if cpu_cap else "")
    sections = {
        "droplet": f"""## Droplet in vacuum

A self-bound liquid droplet in a **non-periodic**, generously padded box: most
of the cell grid is empty, so this exercises the sparse (hashed) grid and a
list without periodic images.

![Droplet: time vs. number of atoms]({plot_names.get("droplet", "")})

How to read it:

- The neighbour-list build dominates the step, so the **list** choice drives the
  scaling: matscipy-neighbours' cell list stays close to linear on both devices
  (on the GPU up to the largest size), the classic matscipy 1.2.0 list is a
  single-threaded CPU reference, and vesin's GPU path grows super-linearly and
  falls far behind for these large, low-density droplets.
- The **kernel** choice mostly shifts the curve: the fused C++/CUDA and Warp
  kernels avoid materialising per-pair arrays, the array (NumPy/CuPy) path is the
  simplest, and JAX `jit`-compiles a dense masked sum.
- On the CPU, the matscipy-neighbours `(mt)` curves pull away from `(1t)` as the
  system grows (at small sizes the thread start-up cost dominates); and from
  about 10⁵ atoms on, even single-threaded matscipy-neighbours `(1t)` is faster
  than the classic matscipy 1.2.0 and vesin CPU lists.
""",
        "liquid": f"""## Bulk liquid in a periodic box

A homogeneous liquid at reduced number density {meta.get("density", 0.8442):g}
in a **fully periodic** cubic box: every grid cell is occupied (dense grid), and
pairs across the boundary carry a non-zero cell shift. At temperature 0.7 the
default density is the Verlet (1967) state point (also the LAMMPS LJ
benchmark's), a liquid at positive pressure well away from coexistence. This is the geometry a
bulk molecular-dynamics or structure-analysis workload sees, and it removes the
vacuum that favours cell lists in the droplet case, so it is the fairer
comparison between the list implementations.

![Liquid: time vs. number of atoms]({plot_names.get("liquid", "")})

How to read it:

- The same list/kernel combinations as above; compare curve by curve with the
  droplet figure. Per atom the liquid has more neighbours within the cutoff
  than the droplet's surface-heavy clusters, so the absolute times are higher
  and the per-pair cost is the better like-for-like number.
- The periodic shifts are folded into the distance vectors by the list (array
  and JAX kernels) or applied from the shift array in the fused kernels (Warp
  and C++), so no kernel wraps positions itself; atoms are free to drift out
  of the box.
""",
    }
    body = f"""# Benchmark

Per-step wall time of the [Lennard-Jones Langevin](examples.md) example for two
systems — a **droplet in vacuum** (non-periodic) and a **bulk liquid in a
periodic box** — across system sizes, for the full cross-product of **device**
(CPU / GPU), **neighbour list** and **kernels** (Warp / array (NumPy/CuPy) /
JAX / C++). Lower is better. The neighbour-list backends are:

- **matscipy-neighbours** — this library (`matscipy_neighbours`), CPU + GPU;
- **matscipy 1.2.0** — the classic [`matscipy`](https://pypi.org/project/matscipy/)
  package's `neighbour_list`, the CPU reference this library descends from;
- **vesin** — [`vesin`](https://github.com/luthaf/vesin), CPU + GPU.

!!! info "Test machine"
    - **CPU:** {meta["cpu"]}
    - **GPU:** {meta["gpu"]}

!!! warning "CPU threading"
    On the CPU the **matscipy-neighbours** list is benchmarked **both
    single-threaded** (`OMP_NUM_THREADS=1`, the `(1t)` curves) **and
    multi-threaded** (all {meta["ncores"]} usable cores, the `(mt)` curves). The
    C++ force loop is OpenMP-parallel and follows the same setting; the Warp and
    array kernels and the JAX backend use their own threading. The classic
    **matscipy 1.2.0** and **vesin** CPU lists are single-threaded. The GPU
    curves are unaffected.

!!! note "Missing curves"
    vesin and matscipy 1.2.0 only feed the Warp and array kernels: JAX uses the
    dense `neighbour_matrix` and the C++ example uses the in-tree C++ core, so
    those panels show matscipy-neighbours only. matscipy 1.2.0 is CPU-only, so it
    has no GPU curve. A curve that ends before the largest size either ran
    **out of GPU memory** at the next size (e.g. the JAX dense neighbour
    matrix, or the per-pair arrays of the array kernels) or was stopped because
    its next run was predicted to take longer than
    {meta.get("max_run_seconds", 60):g} s.

Run configuration: reduced LJ units, cutoff 2.5, dt 0.005, friction 1.0,
temperature 0.7; sizes from {sizes[0]:,} to {sizes[-1]:,} atoms.{cpu_range}
Up to {meta["steps"]} steps per point (fewer for the largest systems; JAX and
Warp are compiled once during an untimed warm-up). Both systems start from an
FCC lattice; the droplet uses lattice constant 1.6, the liquid fills its box at
the stated density.

"""
    body += "\n".join(sections[sy] for sy in meta["systems"] if sy in sections)
    body += """
This page is generated by `examples/lj_langevin/benchmark.py`. Regenerate it on
your own hardware with:

```bash
python examples/lj_langevin/benchmark.py --build build --doc-out docs/benchmark.md
```

The raw timings are written to `--results-out` (JSON); pass that file to
`--replot` to redraw the plots and this page without re-running the benchmark.
`--systems droplet` or `--systems liquid` restricts the run to one system. For
the C++ curves, build with `-DBUILD_EXAMPLES=ON` (and `-DENABLE_CUDA=ON` for
the GPU binary); the others need `pip install jax warp-lang vesin muTimer
matscipy==1.2.0 matplotlib` in the interpreter that runs this driver.
"""
    with open(path, "w") as fh:
        fh.write(body)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--build", default=os.path.join(HERE, "..", "..", "build"),
                    help="CMake build directory (C++ binaries + Python extension)")
    ap.add_argument("--systems", choices=SYSTEM_ORDER, nargs="+",
                    default=list(SYSTEM_ORDER),
                    help="which systems to run (default: both)")
    ap.add_argument("--density", type=float, default=0.8442,
                    help="number density of the periodic liquid (reduced "
                         "units); the default with kT=0.7 is the Verlet (1967) "
                         "liquid state point")
    ap.add_argument("--sizes", type=int, nargs="+",
                    default=[100, 1000, 10000, 100000, 1000000])
    ap.add_argument("--max-atoms-cpu", type=int, default=None,
                    help="skip CPU configurations above this size (GPU runs "
                         "still cover all --sizes)")
    ap.add_argument("--steps", type=int, default=40,
                    help="timed steps for the smallest systems (scaled down "
                         "automatically for larger ones)")
    ap.add_argument("--max-run-seconds", type=float, default=60,
                    help="skip a size (and all larger ones) when its timed loop "
                         "is predicted, from the previous size, to take longer "
                         "than this; keeps slow configurations such as large "
                         "single-threaded CPU runs from dominating the runtime")
    ap.add_argument("--timeout", type=int, default=300,
                    help="per-run timeout in seconds (safety net)")
    ap.add_argument("--plot-dir", default=os.path.join(HERE, "..", "..", "docs"),
                    help="directory for the plots, one benchmark_<system>.png "
                         "per system")
    ap.add_argument("--doc-out", default=None,
                    help="write a documentation page (with hardware info) here")
    ap.add_argument("--results-out", default=None,
                    help="write the raw timings and machine info (JSON) here")
    ap.add_argument("--replot", default=None, metavar="JSON",
                    help="skip the runs; redraw plot/page from a --results-out "
                         "file")
    args = ap.parse_args()

    if args.replot:
        with open(args.replot) as fh:
            saved = json.load(fh)
        if "systems" in saved:
            results = saved["systems"]
        else:   # results file from before the liquid system was added
            results = {"droplet": saved["configs"]}
        saved["meta"].setdefault("systems", list(results))
        finish(results, saved["meta"], args)
        return

    build = os.path.abspath(args.build)
    pkg = os.path.join(HERE, "..", "..", "language_bindings", "python")
    base_env = dict(os.environ,
                    PYTHONPATH=os.pathsep.join(
                        [build, pkg, os.environ.get("PYTHONPATH", "")]))

    results = {}
    for system in args.systems:
        print(f"=== {SYSTEM_NAME[system]}", file=sys.stderr)
        cfgs = make_configs()
        results[system] = cfgs
        for cfg in cfgs:
            if not cfg["supported"]:
                continue
            cfg["points"] = []
            for atoms in args.sizes:
                if (cfg["device"] == "cpu" and args.max_atoms_cpu is not None
                        and atoms > args.max_atoms_cpu):
                    break
                if not within_budget(cfg["points"], atoms, args.steps,
                                     args.max_run_seconds):
                    print(f"  {label(cfg):44s} atoms={atoms} -> predicted run "
                          f"exceeds {args.max_run_seconds:g} s; stopping this "
                          f"configuration", file=sys.stderr)
                    break
                ms = run(cfg, system, args.density, atoms, args.steps, build,
                         base_env, args.timeout)
                if ms is None:
                    print(f"  {label(cfg):44s} atoms={atoms} -> failed/timed "
                          f"out (out of memory?); stopping this configuration",
                          file=sys.stderr)
                    break
                if cfg["points"]:
                    pa, pm = cfg["points"][-1]
                    if (ms / pm) / (atoms / pa) > THRASH_GROWTH:
                        print(f"  {label(cfg):44s} atoms={atoms} -> {ms:.0f} "
                              f"ms/step looks like a memory-thrash artifact "
                              f"(super-linear blow-up); dropping and stopping",
                              file=sys.stderr)
                        break
                cfg["points"].append((atoms, ms))
                print(f"  {label(cfg):44s} atoms={atoms:>8d} -> {ms:.2f} ms/step",
                      file=sys.stderr)

    meta = dict(cpu=detect_cpu(), gpu=detect_gpu(), ncores=usable_cores(),
                systems=args.systems, density=args.density,
                sizes=args.sizes, steps=args.steps,
                timeout=args.timeout, max_run_seconds=args.max_run_seconds,
                max_atoms_cpu=args.max_atoms_cpu)
    if args.results_out:
        with open(args.results_out, "w") as fh:
            json.dump(dict(meta=meta, systems=results), fh, indent=1)
        print(f"wrote {args.results_out}", file=sys.stderr)
    finish(results, meta, args)


def finish(results, meta, args):
    """Plot and print the console table per system, and optionally write the
    doc page."""
    plot_dir = os.path.abspath(args.plot_dir)
    os.makedirs(plot_dir, exist_ok=True)
    plot_names = {}
    for system, cfgs in results.items():
        plot_names[system] = f"benchmark_{system}.png"
        make_plot(cfgs, meta["sizes"], os.path.join(plot_dir, plot_names[system]),
                  system)
        print(f"\n### {SYSTEM_NAME[system]}\n\n"
              + table_markdown(cfgs, meta["sizes"]) + "\n\n(values are ms/step)")
    if args.doc_out:
        write_doc_page(os.path.abspath(args.doc_out), plot_names, meta)
        print(f"\nwrote {args.doc_out}", file=sys.stderr)


if __name__ == "__main__":
    main()
