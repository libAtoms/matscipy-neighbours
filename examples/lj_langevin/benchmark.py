#!/usr/bin/env python3
"""Unified scaling benchmark for the LJ Langevin examples.

Runs the per-step wall time across a logarithmic range of sizes for two
systems -- the bulk **liquid** in a periodic box (dense grid, periodic images)
and the self-bound **droplet** in vacuum (non-periodic, sparse grid) -- for
the full cross-product of three dimensions:

    device   : CPU / GPU
    list     : matscipy (this library) / matscipy 1.2.0 / vesin
               (https://github.com/luthaf/vesin) / NVIDIA ALCHEMI
               (https://github.com/NVIDIA/nvalchemi-toolkit-ops, GPU only)
    kernels  : Warp / array (NumPy or CuPy) / JAX / C++

Not every combination exists: JAX uses a fixed-capacity neighbour matrix, which
only matscipy and ALCHEMI provide, and the C++ example uses the in-tree C++
core — the other cells are left empty. On the GPU the Warp kernels are run on
the pair list and, for matscipy and ALCHEMI, also on the neighbour matrix
(``--format matrix``). On the CPU the matscipy neighbour list is run **both
single-threaded** (`OMP_NUM_THREADS=1`) **and multi-threaded** (all cores); the
threading controls the matscipy list (and the C++ OpenMP force loop). vesin's
CPU list is single-threaded.

Each configuration is launched as a subprocess and its printed `ms/step` is
parsed. Output: a console table per system, a JSON file of raw timings, per
system a log-log plot of time vs. number of atoms faceted by kernel, and a
kernel-comparison plot (matscipy list only, single-threaded CPU and GPU) that
isolates the cost of the Lennard-Jones implementations.
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

SYSTEM_ORDER = ["liquid", "droplet"]
SYSTEM_NAME = {"liquid": "Periodic LJ liquid",
               "droplet": "Non-periodic LJ droplet"}

KERNEL_NAME = {"warp": "Warp", "array": "array (NumPy/CuPy)", "jax": "JAX",
               "cpp": "C++"}
# Facet order: top row array / JAX, bottom row Warp / C++.
KERNEL_ORDER = ["array", "jax", "warp", "cpp"]
KERNEL_COLOUR = {"array": "tab:blue", "jax": "tab:green", "warp": "tab:orange",
                 "cpp": "tab:red"}

# Neighbour-list backends. "matscipy" is this library (matscipy_neighbours);
# "matscipy-classic" is the classic matscipy 1.2.0 package (CPU only); "vesin"
# is https://github.com/luthaf/vesin; "alchemi" is NVIDIA ALCHEMI's
# nvalchemiops (GPU only here).
NL_ORDER = ["matscipy", "matscipy-classic", "vesin", "alchemi"]
NL_DISPLAY = {"matscipy": "matscipy-neighbours",
              "matscipy-classic": "matscipy 1.2.0", "vesin": "vesin",
              "alchemi": "ALCHEMI"}
# Kernels each backend can feed: JAX needs the fixed-capacity neighbour matrix,
# C++ is tied to the in-tree core.
NL_KERNELS = {"matscipy": {"warp", "array", "jax", "cpp"},
              "matscipy-classic": {"warp", "array"},
              "vesin": {"warp", "array"},
              "alchemi": {"warp", "array", "jax"}}

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
    ("gpu", "alchemi", None): dict(c="tab:purple", ls="-", m="v",
                                   label="GPU · ALCHEMI"),
}


def style(cfg):
    """Plot style of a configuration; the neighbour-matrix variant of a GPU
    curve keeps its colour and is drawn dash-dotted with diamonds."""
    st = dict(STYLE[(cfg["device"], cfg["nl"], cfg["threads"])])
    if cfg.get("fmt", "list") == "matrix":
        st.update(ls="-.", m="D", label=st["label"] + " (matrix)")
    return st


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


def _gpu_names(cmd, parse):
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return []
    if out.returncode != 0:   # e.g. nvidia-smi installed but no NVIDIA driver
        return []
    return [n for n in (parse(line) for line in out.stdout.splitlines()) if n]


def detect_gpu():
    """GPU model(s): NVIDIA through nvidia-smi, else AMD through rocm-smi."""
    names = _gpu_names(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                       lambda line: line.strip())
    if not names:   # rocm-smi: "GPU[0]  : Card Series:  AMD Instinct MI300A"
        names = _gpu_names(
            ["rocm-smi", "--showproductname"],
            lambda line: (line.split("Card Series:", 1)[1].strip()
                          if "Card Series:" in line else None))
    if not names:
        return "no GPU detected"
    uniq = sorted(set(names))
    if len(names) > 1 and len(uniq) == 1:
        return f"{len(names)}x {uniq[0]}"
    return ", ".join(names)


def nl_devices(nl):
    """Devices a neighbour-list backend is run on (classic matscipy is CPU
    only; ALCHEMI is benchmarked on the GPU only)."""
    return {"matscipy-classic": ["cpu"], "alchemi": ["gpu"]}.get(nl, ["gpu", "cpu"])


def formats(kernel, nl, device):
    """Neighbour-list formats run for a configuration: the Warp GPU kernels
    also consume the neighbour matrix of the backends that provide one."""
    if kernel == "warp" and device == "gpu" and nl in ("matscipy", "alchemi"):
        return ["list", "matrix"]
    return ["list"]


def make_configs(kernels=KERNEL_ORDER, lists=NL_ORDER, devices=("gpu", "cpu")):
    """The full matrix, optionally restricted to a subset of kernels, lists and
    devices. A backend only feeds the kernels in `NL_KERNELS`; the other
    (kernel, list) cells are kept but marked unsupported -> empty in the
    table. Only the matscipy (this library) CPU list is split into single/multi-
    thread."""
    cfgs = []
    for kernel in [k for k in KERNEL_ORDER if k in kernels]:
        for nl in [n for n in NL_ORDER if n in lists]:
            supported = kernel in NL_KERNELS[nl]
            for device in [d for d in nl_devices(nl) if d in devices]:
                if device == "cpu" and nl == "matscipy":
                    threads_list = ["mt", "1t"]
                else:
                    threads_list = [None]
                for threads in threads_list:
                    for fmt in formats(kernel, nl, device):
                        cfgs.append(dict(kernel=kernel, nl=nl, device=device,
                                         threads=threads, fmt=fmt,
                                         supported=supported))
    return cfgs


def label(cfg):
    thr = {"mt": " (mt)", "1t": " (1t)", None: ""}[cfg["threads"]]
    dev = cfg["device"].upper()
    fmt = " (matrix)" if cfg.get("fmt", "list") == "matrix" else ""
    return (f"{KERNEL_NAME[cfg['kernel']]} · {NL_DISPLAY[cfg['nl']]} · "
            f"{dev}{thr}{fmt}")


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
              "--write-every", "0", "--out", os.devnull]   # no frames in the timing
    kernel = cfg["kernel"]
    if kernel == "warp":
        return [sys.executable, os.path.join(HERE, "lj_langevin_warp.py"),
                "--device", cfg["device"], "--neighbours", cfg["nl"],
                "--format", cfg.get("fmt", "list"), "--system", system, "--density", str(density),
                "--atoms", str(atoms), "--steps", str(steps)], env
    if kernel == "array":
        return [sys.executable, os.path.join(HERE, "lj_langevin.py"),
                "--device", cfg["device"], "--neighbours", cfg["nl"]] + common, env
    if kernel == "jax":
        if cfg["device"] == "cpu":
            env["JAX_PLATFORMS"] = "cpu"
        # By default JAX reserves 75% of GPU memory up front; the neighbour
        # list allocates outside that pool, so it would run out of memory long
        # before the card is full.
        env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
        return [sys.executable, os.path.join(HERE, "lj_langevin_jax.py"),
                "--device", cfg["device"], "--neighbours", cfg["nl"]] + common, env
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
# 1M on a 6 GB card) without touching genuinely-slow-but-real points. Where
# running out of GPU memory fails cleanly (native Linux), a backend with a real
# super-linear step can trip it; `--thrash-growth 0` switches the check off.
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
        st = style(cfg)
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


def make_kernel_plot(results, sizes, path):
    """One panel per system: the four kernels on top of the *same* neighbour
    list (matscipy-neighbours), single-threaded on the CPU and on the GPU. The
    list build is identical within a device, so the spread between curves is
    the cost of the LJ implementation."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    systems = [sy for sy in SYSTEM_ORDER if sy in results]
    fig, axes = plt.subplots(1, len(systems), figsize=(5.5 * len(systems), 4.8),
                             sharex=True, sharey=True, squeeze=False)
    for ax, system in zip(axes.flat, systems):
        for cfg in results[system]:
            pts = cfg.get("points")
            if (not pts or cfg["nl"] != "matscipy"
                    or cfg.get("fmt", "list") != "list"):
                continue
            if cfg["device"] == "gpu":
                ls, mk, dev = "-", "o", "GPU"
            elif cfg["threads"] == "1t":
                ls, mk, dev = ":", "x", "CPU (1t)"
            else:
                continue
            ax.plot([a for a, _ in pts], [t for _, t in pts],
                    color=KERNEL_COLOUR[cfg["kernel"]], ls=ls, marker=mk, ms=5,
                    label=f"{KERNEL_NAME[cfg['kernel']]} · {dev}")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(SYSTEM_NAME[system])
        ax.grid(True, which="both", ls=":", alpha=0.4)
        ax.set_xlabel("number of atoms")
        ax.set_ylabel("time per step (ms)")
        if ax.has_data():
            ax.legend(fontsize=8)
    fig.suptitle("Kernel comparison on the matscipy-neighbours list "
                 "(GPU, and single-threaded CPU)", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    print(f"wrote {path}", file=sys.stderr)


GPU_COMPARE_KERNELS = ["array", "jax"]
GPU_COMPARE_STYLE = {"array": dict(ls="-", m="o"), "jax": dict(ls="--", m="s")}
MACHINE_COLOURS = ["tab:blue", "tab:red", "tab:purple", "tab:brown", "tab:olive"]
# How the array kernel sums the pair forces per atom (meta["array_force_sum"]);
# results files from before the key existed used the weighted bincount.
ARRAY_FORCE_SUM = "segment sum"
ARRAY_FORCE_SUM_OLD = "bincount"


def make_gpu_plot(machines, path):
    """One panel per system: the array and JAX kernels on the matscipy-neighbours
    GPU list, one colour per GPU. ``machines`` is a list of (name, meta,
    systems) with meta and systems as in a results file; the array curves are
    labelled with how the run summed the pair forces."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    systems = [sy for sy in SYSTEM_ORDER
               if any(sy in m_systems for _, _, m_systems in machines)]
    fig, axes = plt.subplots(1, len(systems), figsize=(5.5 * len(systems), 4.8),
                             sharex=True, sharey=True, squeeze=False)
    for ax, system in zip(axes.flat, systems):
        for (name, m_meta, m_systems), colour in zip(machines, MACHINE_COLOURS):
            for cfg in m_systems.get(system, []):
                pts = cfg.get("points")
                if (not pts or cfg["nl"] != "matscipy" or cfg["device"] != "gpu"
                        or cfg.get("fmt", "list") != "list"
                        or cfg["kernel"] not in GPU_COMPARE_KERNELS):
                    continue
                st = GPU_COMPARE_STYLE[cfg["kernel"]]
                lbl = f"{name} · {KERNEL_NAME[cfg['kernel']]}"
                if cfg["kernel"] == "array":
                    lbl += " · " + m_meta.get("array_force_sum",
                                              ARRAY_FORCE_SUM_OLD)
                ax.plot([a for a, _ in pts], [t for _, t in pts], color=colour,
                        ls=st["ls"], marker=st["m"], ms=5, label=lbl)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(SYSTEM_NAME[system])
        ax.grid(True, which="both", ls=":", alpha=0.4)
        ax.set_xlabel("number of atoms")
        ax.set_ylabel("time per step (ms)")
        if ax.has_data():
            ax.legend(fontsize=8)
    fig.suptitle("GPU comparison: array (CuPy) and JAX kernels on the "
                 "matscipy-neighbours list", fontsize=12)
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


def write_doc_page(path, plot_names, meta, kernel_plot_name, gpu_plot_name,
                   machines):
    """Write the documentation page. ``plot_names`` maps system -> image file
    name (relative to the page); ``kernel_plot_name`` is the kernel-comparison
    figure and ``gpu_plot_name`` the GPU comparison across ``machines`` (name ->
    {meta, systems}) and the main test machine."""
    if meta.get("array_force_sum", ARRAY_FORCE_SUM_OLD) == ARRAY_FORCE_SUM_OLD:
        array_sum_text = ("scatters the forces with a weighted `bincount` per "
                          "component (an atomic scatter)")
    else:
        array_sum_text = ("sums the forces of each atom's contiguous segment of "
                          "pairs (`first_neighbours`, then one thread per atom "
                          "on the GPU and `add.reduceat` on the CPU; no "
                          "atomics)")
    gpu_list = "\n".join(
        f"- **{name}** (host CPU: {m['meta']['cpu']})"
        for name, m in [(meta["gpu"], dict(meta=meta))] + list(machines.items()))
    sizes = meta["sizes"]
    cpu_cap = meta.get("max_atoms_cpu")
    cpu_range = (f"\nCPU runs stop at {cpu_cap:,} atoms; the GPU runs cover the "
                 f"full range." if cpu_cap else "")
    sections = {
        "droplet": f"""## Non-periodic: droplet in vacuum

A self-bound liquid droplet in a **non-periodic**, generously padded box: most
of the cell grid is empty, so this exercises the sparse (hashed) grid and a
list without periodic images.

![Droplet: time vs. number of atoms]({plot_names.get("droplet", "")})

How to read it (on the test machine):

- matscipy-neighbours scales linearly on the GPU up to the largest size. Its
  neighbour matrix (indices only, as the non-periodic droplet needs no
  shifts) is the fastest GPU list at every size, and its pair list is faster
  than ALCHEMI's pair list at every size.
- ALCHEMI's list build grows super-linearly beyond a few million atoms on this
  sparse geometry: from 3×10⁶ to 10⁷ atoms its build time rises about
  twentyfold for 3.3× the atoms, and its curves end well above
  matscipy-neighbours'.
- vesin's GPU path grows super-linearly and falls far behind for these large,
  low-density droplets.
- On the CPU, single-threaded matscipy-neighbours `(1t)` is faster than the
  classic matscipy 1.2.0 and vesin CPU lists from about 10⁵ atoms on; the
  multi-threaded `(mt)` curves pull away as the system grows (at small sizes
  the thread start-up cost dominates).
""",
        "liquid": f"""## Periodic: bulk liquid

A homogeneous liquid at reduced number density {meta.get("density", 0.8442):g}
in a **fully periodic** cubic box: every grid cell is occupied (dense grid), and
pairs across the boundary carry a non-zero cell shift. At temperature 0.7 the
default density is the Verlet (1967) state point (also the LAMMPS LJ
benchmark's), a liquid at positive pressure well away from coexistence. This is the geometry a
bulk molecular-dynamics or structure-analysis workload sees, and it removes the
vacuum that favours cell lists in the droplet case, so it is the fairer
comparison between the list implementations.

![Liquid: time vs. number of atoms]({plot_names.get("liquid", "")})

How to read it (on the test machine):

- On the GPU the list build dominates the step for the Warp and C++ kernels,
  which avoid per-pair arrays, and the array kernel adds a roughly constant
  factor on top. The **list** choice therefore drives the scaling. JAX, which
  consumes the neighbour matrix (the faster list to build), is on par with
  the fused kernels.
- In the **matrix** format (the Warp `(matrix)` curves, where both libraries
  supply indices and cell shifts to the same kernel) ALCHEMI is ahead from 10⁶
  to 3×10⁶ atoms (by 10–25%), matscipy-neighbours below 10⁶ atoms and at 10⁷
  atoms (there by almost a factor of two). At 3×10⁷ atoms
  matscipy-neighbours' matrix runs out of GPU memory: its int64 indices and
  shifts take twice the space of ALCHEMI's int32 ones.
- On the **pair list** matscipy-neighbours is ahead below 10⁶ atoms, the two
  are within about 5% from 10⁶ to 3×10⁶, matscipy-neighbours is ahead at 10⁷
  atoms, and only matscipy-neighbours reaches 3×10⁷ atoms, where its pair
  list is also slightly faster than ALCHEMI's matrix.
- In the **JAX** panel matscipy-neighbours' `neighbour_matrix` is faster than
  ALCHEMI's jit-compiled cell list from 10⁵ atoms on and reaches 10⁷ atoms,
  while ALCHEMI runs out of GPU memory above 3×10⁶.
- vesin's GPU path is more than an order of magnitude slower than both at 10⁶
  atoms and grows super-linearly.
- On the CPU, single-threaded matscipy-neighbours `(1t)` is faster than both
  the classic matscipy 1.2.0 and the vesin CPU lists at every size measured.
- Per atom the liquid has more neighbours within the cutoff than the droplet,
  whose surface atoms have fewer, so absolute times are somewhat higher.
- The periodic shifts are folded into the distance vectors by the list (array
  and JAX kernels) or applied from the shift array in the fused kernels (Warp
  and C++), so no kernel wraps positions itself; atoms are free to drift out
  of the box.
""",
    }
    body = f"""# Benchmark

Per-step wall time of the [Lennard-Jones Langevin](examples.md) example for two
systems — a **bulk liquid in a periodic box** and a **droplet in vacuum**
(non-periodic) — across system sizes, for the full cross-product of **device**
(CPU / GPU), **neighbour list** and **kernels** (Warp / array (NumPy/CuPy) /
JAX / C++). Lower is better. The neighbour-list backends are:

- **matscipy-neighbours** — this library (`matscipy_neighbours`), CPU + GPU;
- **matscipy 1.2.0** — the classic [`matscipy`](https://pypi.org/project/matscipy/)
  package's `neighbour_list`, the CPU reference this library descends from;
- **vesin** — [`vesin`](https://github.com/luthaf/vesin), CPU + GPU;
- **ALCHEMI** — NVIDIA ALCHEMI's
  [`nvalchemiops`](https://github.com/NVIDIA/nvalchemi-toolkit-ops) cell list
  (`method="cell_list"`), GPU only, through its PyTorch interface (Warp and
  array kernels) and its JAX interface (JAX kernel).

!!! warning "What is being measured"
    Every step rebuilds the neighbour list from scratch, and the list is the
    dominant cost of a step here. That is deliberate: these runs are set up to
    expose the neighbour-list performance as much as possible. In a real
    application one would build the list with a Verlet shell (a skin added to
    the cutoff) and reuse it for many steps until an atom has moved half the
    skin, so the total time spent constructing neighbour lists plays a much
    smaller role in the overall cost of a simulation than it does below.

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

!!! note "Pair list or neighbour matrix"
    A neighbour list comes in two layouts. The **pair list** is three flat
    arrays `i`, `j`, `S` with one entry per pair; every backend returns it, and
    the array, Warp and C++ kernels run one thread (or array element) per
    pair. The **neighbour matrix** gives each atom a row of fixed capacity
    (here 96 slots) plus a neighbour count; its shapes are static, so the JAX
    kernel uses it, and it is ALCHEMI's native layout (its pair list is
    compacted from the matrix). The Warp panels therefore show the GPU curves
    of matscipy-neighbours and ALCHEMI twice: on the pair list, and on the
    matrix (`(matrix)`, one thread per atom summing its own row). In the matrix
    form both libraries supply the same data — neighbour indices, plus the
    cell shifts in the periodic box (`neighbour_matrix(...,
    quantities="S")`, or `""` for the droplet) — to the same kernel.

!!! note "Missing curves"
    vesin and matscipy 1.2.0 only feed the Warp and array kernels, and ALCHEMI
    the Warp, array and JAX kernels: JAX needs a fixed-capacity neighbour
    matrix, and the C++ example uses the in-tree C++ core. matscipy 1.2.0 is
    CPU-only, and ALCHEMI is benchmarked on the GPU only. A curve that ends
    before the largest size either ran **out of GPU memory** at the next size
    (e.g. the JAX dense neighbour matrix, or the per-pair arrays of the array
    kernels) or was stopped because its next run was predicted to take longer
    than {meta.get("max_run_seconds", 60):g} s.

Run configuration: reduced LJ units, cutoff 2.5, dt 0.005, friction 1.0,
temperature 0.7; sizes from {sizes[0]:,} to {sizes[-1]:,} atoms.{cpu_range}
Up to {meta["steps"]} steps per point (fewer for the largest systems; JAX and
Warp are compiled once during an untimed warm-up; JAX runs without its
default up-front reservation of GPU memory, which would leave too little for
the neighbour list). Both systems start from an
FCC lattice; the droplet uses lattice constant 1.6, the liquid fills its box at
the stated density.

"""
    # Periodic first, whatever order the systems were run (or saved) in.
    body += "\n".join(sections[sy] for sy in SYSTEM_ORDER
                       if sy in meta["systems"] and sy in sections)
    body += f"""
## Kernel comparison: the cost of the Lennard-Jones implementation

The figure below keeps the neighbour list fixed — **matscipy-neighbours** on
the GPU, and on the CPU **single-threaded** — and varies only the kernels that
consume it: array (NumPy/CuPy), JAX, Warp and C++/CUDA. Within one device the
array, Warp and C++ curves share the same pair-list build, so the vertical
spread between them is the cost of the potential, not of the list; JAX
consumes `neighbour_matrix` instead, which on the GPU builds faster than the
pair list (see the matrix curves in the Warp panels), so part of JAX's lead
there is the list. This section therefore does
**not** primarily benchmark the neighbour list but the implementation of the
Lennard-Jones potential on top of it:

- the **array** path materialises the per-pair distance vectors returned by the
  list and {array_sum_text};
- **JAX** `jit`-compiles a dense masked sum over the fixed-capacity
  `neighbour_matrix` (no scatter, but padded rows);
- **Warp** and **C++/CUDA** run one fused pass over the `ij` pairs (plus the
  shift `S` in the periodic box) that recomputes each distance and never
  materialises per-pair arrays; on the GPU the potential energy is reduced
  within each thread block and added with one atomic per block.

![Kernel comparison]({kernel_plot_name})

The C++ force loop is OpenMP-parallel and honours the single-thread setting;
the NumPy, Warp-CPU and JAX-CPU kernels use their own threading and are not
pinned to one core, so on the CPU the comparison is indicative rather than
strict.

## GPU comparison: array and JAX kernels across GPUs

The array (CuPy) and JAX kernels run unchanged on NVIDIA (CUDA) and AMD (ROCm)
GPUs, so they can compare GPUs directly. The figure below shows both kernels on
the matscipy-neighbours GPU list (pair list for the array kernels,
`neighbour_matrix` for JAX) on each GPU measured so far:

{gpu_list}

![GPU comparison]({gpu_plot_name})

How to read it:

- The legend says how each run's array kernel sums the pair forces per atom:
  a weighted `bincount` per component (an atomic scatter; the H200 curves) or
  a segment sum over the pairs of each atom (`first_neighbours`, then a small
  CuPy kernel with one thread per atom, no atomics; the MI300A curves). On the
  MI300A the scatter took about 440 ms at 10⁶ atoms and the segment sum takes
  about 1.5 ms.
- On the **AMD Instinct MI300A** (ROCm 6.4, JAX 0.4.35 from AMD's ROCm wheels,
  CuPy 13.6 built from source) the JAX kernel is on par with the H200 up to
  10⁴ atoms, where both are dominated by launch latency, and about 2.5× slower
  from 10⁶ atoms on (26 ms against 11 ms per step at 10⁶ atoms).
- The array kernel on the MI300A is about 9× slower than on the H200 at 10⁶
  atoms (185–190 ms against 22 ms). Most of the step goes into the two full reductions
  (`.sum()` for the energy and the virial), about 80 ms each: CuPy accelerates
  reductions with CUB by default on CUDA but not on ROCm, and its generic
  reduction is slow there. The neighbour list takes about 30 ms of the step.
- On the MI300A, JAX runs out of memory at 10⁷ atoms and the array kernel at
  3×10⁷: plain device allocations reach only the ~63 GiB coarse-grained window
  of the 128 GB of HBM the APU shares with its CPUs, and the neighbour matrix
  alone needs 31 GB at 10⁷ atoms. The H200 (141 GB) runs JAX at 10⁷ atoms.

## Performance portability: NVIDIA and AMD

The same source runs on NVIDIA (CUDA) and AMD (ROCm) GPUs, but code that is
fast on one is not automatically fast on the other. Bringing the MI300A runs
in line with the H200 ones surfaced the following, roughly in order of impact;
the numbers are for the periodic liquid at 10⁶ atoms (5.4×10⁷ pairs) on the
MI300A.

**Caching allocators behave differently.** The library allocates its outputs
afresh on every call, so it relies on a caching allocator. On CUDA the
stream-ordered memory pool (`cudaMallocFromPoolAsync` with an unlimited
release threshold) splits and reuses freed blocks as intended. ROCm 6.4's pool
did not reliably hand freed blocks back: small requests took the large cached
blocks, and whether a block was reused varied from run to run. Every miss maps
fresh memory, at 40–150 ms for the gigabyte-sized outputs, and the pair list
took 160 ms to build instead of 20 ms. On HIP the library therefore keeps its
own cache of `hipMalloc` blocks, in free lists by size, with requests rounded
up to one of eight sizes per power of two.

**Floating-point atomics are not free.** A per-atom sum over pairs written as
a scatter — a weighted `bincount`, or `jax.ops.segment_sum` — lowers to
float64 atomic adds. Those are fast on NVIDIA but slow on the MI300A when many
threads hit the same atom, as they do for pairs sorted by atom: the weighted
`bincount` took about 440 ms (three components), JAX's `segment_sum` (even with
`indices_are_sorted=True`) 1.8 s. The list is a *full* list, so no scatter is
needed: the pairs of an atom are one contiguous segment, `first_neighbours`
gives the segment starts (on the GPU for device input), and one thread per
atom sums its segment in about 1.5 ms, deterministically. The neighbour-matrix
format avoids the scatter by construction, since each atom sums its own row;
this is how the JAX kernel works.

**Library defaults differ between backends.** CuPy accelerates reductions with
CUB by default on CUDA, but not on ROCm (`CUPY_ACCELERATORS` defaults to
`cub` and to empty, respectively). Without it a full reduction such as
`x.sum()` over 5.4×10⁷ elements takes about 80 ms on the MI300A, and these two
reductions (energy and virial) are now most of the array kernel's step.
Enabling CUB on ROCm is not a fix: in the CuPy 13.6 build used here,
`sum(axis=1)` then returned wrong results for large arrays. CuPy's
`add.reduceat` is built from a cumulative sum over all elements, which costs a
temporary of the input's size (13 GB at 10⁷ atoms) and some accuracy.

**Unified memory on an APU has its own limits.** The MI300A's CPUs and GPU
share 128 GB of HBM, but plain device allocations (`hipMalloc`, and the pools
built on it by the library, CuPy and JAX) reach only a coarse-grained window
of about 63 GiB. Runs that fit easily on the 141 GB H200 run out of memory
there. Managed memory (`hipMallocManaged`) reaches the whole HBM, but the HIP
runtime then performs device-to-device `hipMemcpy` on the host CPU, so copies
should be done with kernels instead. The CPU and the GPU also share the
memory bandwidth, so load from other processes on the node can show up in the
timings.

**Getting the software stack right takes care.** The JAX ROCm plugins on PyPI
are built for other ROCm versions (`libamd_comgr.so.2` versus ROCm 6.4's
`.so.3`); AMD's wheels from `repo.radeon.com` for the installed ROCm release
work. There are no CuPy wheels for ROCm 6.x, so CuPy is built from source
(`CUPY_INSTALL_USE_HIP=1`, `HCC_AMDGPU_TARGET=gfx942`). CuPy compiles each
kernel on first use and caches it on disk, which on ROCm took long enough to
inflate a first run several-fold, so the cache should be warmed before timing.

This page is generated by `examples/lj_langevin/benchmark.py`. Regenerate it on
your own hardware with:

```bash
python examples/lj_langevin/benchmark.py --build build --doc-out docs/benchmark.md
```

The raw timings are written to `--results-out` (JSON); pass that file to
`--replot` to redraw the plots and this page without re-running the benchmark
(the kernel-comparison figure is drawn from the same results). `--systems
liquid` or `--systems droplet` restricts the run to one system. For
the C++ curves, build with `-DBUILD_EXAMPLES=ON` (and `-DENABLE_CUDA=ON` for
the GPU binary); the others need `pip install jax warp-lang vesin muTimer
matscipy==1.2.0 matplotlib nvalchemi-toolkit-ops` and a CUDA build of PyTorch
(for ALCHEMI) in the interpreter that runs this driver.

To add another GPU to the GPU comparison, run only the array and JAX kernels on
it and store them in the existing results file (its main results stay as they
are; the plots and this page are redrawn from the merged file):

```bash
python examples/lj_langevin/benchmark.py --build build \\
    --devices gpu --lists matscipy --kernels array jax \\
    --sizes 100 1000 10000 100000 1000000 3000000 10000000 30000000 \\
    --add-machine docs/benchmark_results.json --doc-out docs/benchmark.md
```

On an AMD GPU this needs a HIP build (`-DENABLE_HIP=ON`) and the ROCm builds of
CuPy and JAX.
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
    ap.add_argument("--thrash-growth", type=float, default=THRASH_GROWTH,
                    help="drop a point, and stop its configuration, when the "
                         "time per step grows this many times faster than the "
                         "atom count (a WSL memory-spill artifact); 0 disables")
    ap.add_argument("--timeout", type=int, default=300,
                    help="per-run timeout in seconds (safety net)")
    ap.add_argument("--plot-dir", default=os.path.join(HERE, "..", "..", "docs"),
                    help="directory for the plots: one benchmark_<system>.png "
                         "per system plus benchmark_kernels.png")
    ap.add_argument("--doc-out", default=None,
                    help="write a documentation page (with hardware info) here")
    ap.add_argument("--results-out", default=None,
                    help="write the raw timings and machine info (JSON) here")
    ap.add_argument("--replot", default=None, metavar="JSON",
                    help="skip the runs; redraw plot/page from a --results-out "
                         "file")
    ap.add_argument("--resume", default=None, metavar="JSON",
                    help="continue an interrupted run: configurations that a "
                         "--results-out file records as finished are taken "
                         "from it instead of being re-run (use the same "
                         "--sizes and settings)")
    ap.add_argument("--kernels", choices=KERNEL_ORDER, nargs="+",
                    default=list(KERNEL_ORDER),
                    help="run only these kernels (default: all)")
    ap.add_argument("--lists", choices=NL_ORDER, nargs="+",
                    default=list(NL_ORDER),
                    help="run only these neighbour-list backends (default: all)")
    ap.add_argument("--devices", choices=["gpu", "cpu"], nargs="+",
                    default=["gpu", "cpu"],
                    help="run only on these devices (default: both)")
    ap.add_argument("--add-machine", default=None, metavar="JSON",
                    help="store this run as an additional machine in an "
                         "existing --results-out file (under 'machines', keyed "
                         "by --machine) instead of writing a results file of "
                         "its own; the file's main results are kept, and the "
                         "plots and page are redrawn from the merged file. "
                         "Typically combined with --devices gpu --lists "
                         "matscipy --kernels array jax for the GPU comparison")
    ap.add_argument("--machine", default=None,
                    help="name of this machine for --add-machine (default: the "
                         "detected GPU model)")
    args = ap.parse_args()

    if args.replot:
        with open(args.replot) as fh:
            saved = json.load(fh)
        if "systems" in saved:
            results = saved["systems"]
        else:   # results file from before the liquid system was added
            results = {"droplet": saved["configs"]}
        saved["meta"].setdefault("systems", list(results))
        finish(results, saved["meta"], args, saved.get("machines", {}))
        return

    build = os.path.abspath(args.build)
    pkg = os.path.join(HERE, "..", "..", "language_bindings", "python")
    base_env = dict(os.environ,
                    PYTHONPATH=os.pathsep.join(
                        [build, pkg, os.environ.get("PYTHONPATH", "")]))

    meta = dict(cpu=detect_cpu(), gpu=detect_gpu(), ncores=usable_cores(),
                systems=args.systems, density=args.density,
                sizes=args.sizes, steps=args.steps,
                timeout=args.timeout, max_run_seconds=args.max_run_seconds,
                max_atoms_cpu=args.max_atoms_cpu,
                thrash_growth=args.thrash_growth,
                kernels=args.kernels, lists=args.lists, devices=args.devices,
                array_force_sum=ARRAY_FORCE_SUM)
    machine = args.machine or meta["gpu"]
    merged = None
    if args.add_machine:
        with open(args.add_machine) as fh:
            merged = json.load(fh)
    finished = {}
    if args.resume:
        with open(args.resume) as fh:
            saved = json.load(fh)
        if args.add_machine and machine in saved.get("machines", {}):
            saved = saved["machines"][machine]
        for system, cfgs in saved["systems"].items():
            finished[system] = {label(c): c for c in cfgs if c.get("done")}

    def save(results):
        """Write the results so far; runs are long, and an interrupted one can
        be continued with --resume."""
        if merged is not None:
            merged.setdefault("machines", {})[machine] = dict(meta=meta,
                                                             systems=results)
            with open(args.add_machine, "w") as fh:
                json.dump(merged, fh, indent=1)
        elif args.results_out:
            with open(args.results_out, "w") as fh:
                json.dump(dict(meta=meta, systems=results), fh, indent=1)

    results = {}
    for system in args.systems:
        print(f"=== {SYSTEM_NAME[system]}", file=sys.stderr)
        cfgs = make_configs(args.kernels, args.lists, args.devices)
        results[system] = cfgs
        for cfg in cfgs:
            if not cfg["supported"]:
                continue
            done = finished.get(system, {}).get(label(cfg))
            if done is not None:
                cfg["points"], cfg["done"] = done["points"], True
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
                    if (args.thrash_growth > 0
                            and (ms / pm) / (atoms / pa) > args.thrash_growth):
                        print(f"  {label(cfg):44s} atoms={atoms} -> {ms:.0f} "
                              f"ms/step looks like a memory-thrash artifact "
                              f"(super-linear blow-up); dropping and stopping",
                              file=sys.stderr)
                        break
                cfg["points"].append((atoms, ms))
                print(f"  {label(cfg):44s} atoms={atoms:>8d} -> {ms:.2f} ms/step",
                      file=sys.stderr)
            cfg["done"] = True
            save(results)

    save(results)
    if merged is not None:
        print(f"wrote {args.add_machine} (machine '{machine}')", file=sys.stderr)
        for system in args.systems:
            print(f"\n### {machine}: {SYSTEM_NAME[system]}\n\n"
                  + table_markdown(results[system], meta["sizes"])
                  + "\n\n(values are ms/step)")
        finish(merged["systems"], merged["meta"], args, merged["machines"])
        return
    if args.results_out:
        print(f"wrote {args.results_out}", file=sys.stderr)
    finish(results, meta, args)


def finish(results, meta, args, machines=None):
    """Plot and print the console table per system, and optionally write the
    doc page. ``machines`` maps the name of each additional machine to its
    {meta, systems}; they only enter the GPU-comparison plot."""
    machines = machines or {}
    plot_dir = os.path.abspath(args.plot_dir)
    os.makedirs(plot_dir, exist_ok=True)
    plot_names = {}
    for system in [sy for sy in SYSTEM_ORDER if sy in results]:
        cfgs = results[system]
        plot_names[system] = f"benchmark_{system}.png"
        make_plot(cfgs, meta["sizes"], os.path.join(plot_dir, plot_names[system]),
                  system)
        print(f"\n### {SYSTEM_NAME[system]}\n\n"
              + table_markdown(cfgs, meta["sizes"]) + "\n\n(values are ms/step)")
    kernel_plot_name = "benchmark_kernels.png"
    make_kernel_plot(results, meta["sizes"],
                     os.path.join(plot_dir, kernel_plot_name))
    gpu_plot_name = "benchmark_gpus.png"
    make_gpu_plot([(meta["gpu"], meta, results)]
                  + [(name, m["meta"], m["systems"])
                     for name, m in machines.items()],
                  os.path.join(plot_dir, gpu_plot_name))
    if args.doc_out:
        write_doc_page(os.path.abspath(args.doc_out), plot_names, meta,
                       kernel_plot_name, gpu_plot_name, machines)
        print(f"\nwrote {args.doc_out}", file=sys.stderr)


if __name__ == "__main__":
    main()
