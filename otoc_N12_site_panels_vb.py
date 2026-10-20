"""Per-site OTOC panels for the 2x3 Kitaev-Heisenberg truncation series, with the
Vb (Eq 14) correction added next to the corrections already in the data.

    python otoc_N12_site_panels_vb.py
    $env:N12_DATA_DIR = "D:\\some\\folder"; python otoc_N12_site_panels_vb.py

Reproduces the layout of the "OTOC at site j ... raw vs corrected vs exact
Trotter" figures (one figure per site, one column-pair per cutoff, an upper
OTOC panel and a lower run - exact strip) and adds one curve:

    Vb = raw + 2 l(t) Theta(v_B t - |j - j0|),   l = 1 - w_norm2

v_B is fitted per cutoff to the RAW run's crossings of Csq > 0.2 (the note's
0.1, doubled for this project's Csq = 4 sum_anti |c|^2), using only crossings
with l <= 0.05, through the origin -- the same procedure as
otoc_vb_front_correction.jl and otoc_N12_correction_compare.jl.

"corrected" is the otoc_cor_j column already in the CSV: raw plus the OTOC
weight removed at every truncation, accumulated over the run.
"exact" is the cut 1e-12 run, which keeps all 4^11 reachable strings.

Output: plots/N12_site<j>_vb.png, one per site.
"""
import os
from collections import deque

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.lines import Line2D

# --- configuration -----------------------------------------------------------
DATA_DIR = os.environ.get(
    "N12_DATA_DIR",
    r"C:\Users\maria\Downloads\data-trunc_series_N12\data-trunc_series_N12")
PREFIX = "kh_honeycomb_N12_J0.1_cut"
REF_CUT = "1e-12"
CUTS = ["1e-3", "5e-4", "2e-4", "1e-4", "5e-5", "1e-5", "1e-6"]
THR_NOTE = 0.1          # note's threshold on C_S; Csq threshold is twice this
ELL_FIT_MAX = 0.05      # use a crossing in the v_B fit only while l is this small
# -----------------------------------------------------------------------------

THR_CSQ = 2 * THR_NOTE
OUTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots")
os.makedirs(OUTDIR, exist_ok=True)

C_EXACT, C_RAW, C_COR, C_VB = "#0b0b0b", "#eb6834", "#2a78d6", "#1baf7a"
INK2, GRID, SURFACE = "#52514e", "#e1e0d9", "#fcfcfb"


def read_run(cut):
    path = os.path.join(DATA_DIR, f"{PREFIX}{cut}.csv")
    with open(path) as f:
        hdr = f.readline().strip().split(",")
    M = np.loadtxt(path, delimiter=",", skiprows=1)
    col = {h: M[:, i] for i, h in enumerate(hdr)}
    sites = [int(h[5:]) for h in hdr if h.startswith("otoc_") and not h.startswith("otoc_cor_")]
    return dict(time=col["time"], norm2=col["w_norm2"], sites=sites,
                raw={j: col[f"otoc_{j}"] for j in sites},
                cor={j: col[f"otoc_cor_{j}"] for j in sites})


def distances(cut, origin, N):
    """Bond distances from the run's own Hamiltonian file (two-body strings = bonds)."""
    adj = {q: set() for q in range(1, N + 1)}
    with open(os.path.join(DATA_DIR, f"{PREFIX}{cut}_hamiltonian.txt")) as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            q = [i + 1 for i, ch in enumerate(line.split()[1]) if ch != "I"]
            if len(q) == 2:
                adj[q[0]].add(q[1]); adj[q[1]].add(q[0])
    d = {origin: 0}; queue = deque([origin])
    while queue:
        u = queue.popleft()
        for v in adj[u]:
            if v not in d:
                d[v] = d[u] + 1; queue.append(v)
    return d


def fit_vB(run, dist):
    ell = 1 - run["norm2"]
    pts = []
    for j in run["sites"]:
        k = np.argmax(run["raw"][j] > THR_CSQ)
        if run["raw"][j][k] <= THR_CSQ or ell[k] > ELL_FIT_MAX:
            continue
        pts.append((dist[j], run["time"][k]))
    d, t = np.array(pts).T
    return float((d * t).sum() / (t * t).sum())


def style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#898781")
    ax.tick_params(colors=INK2, labelsize=10)
    ax.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)


ref = read_run(REF_CUT)
N = max(ref["sites"])
origin = (set(range(1, N + 1)) - set(ref["sites"])).pop()
dist = distances(REF_CUT, origin, N)

runs = {}
for cut in CUTS:
    r = read_run(cut)
    n = min(len(r["time"]), len(ref["time"]))
    assert np.allclose(r["time"][:n], ref["time"][:n]), f"{cut}: time grids differ"
    vB = fit_vB(r, dist)
    ell = 1 - r["norm2"][:n]
    r["vb"] = {j: r["raw"][j][:n] + 2 * ell * (vB * r["time"][:n] > dist[j])
               for j in r["sites"]}
    r["n"], r["vB"] = n, vB
    runs[cut] = r
    print(f"cut {cut:>5}: v_B = {vB:.3f} bonds/time, l_end = {ell[-1]:.3e}")

plt.rcParams.update({"font.family": "DejaVu Sans", "axes.titlesize": 11.5})

for j in ref["sites"]:
    fig = plt.figure(figsize=(19, 12.4), dpi=100, facecolor=SURFACE)
    fig.suptitle(f"OTOC at site {j} (distance {dist[j]} from W = Z$_{{{origin}}}$): "
                 "raw vs corrected vs Vb vs exact Trotter \u2014 2x3 periodic honeycomb, "
                 "K = 1, J = 0.1, dt = 0.05",
                 x=0.007, y=0.995, ha="left", fontsize=13.5)
    outer = GridSpec(2, 4, figure=fig, left=0.125, right=0.92, top=0.945, bottom=0.04,
                     wspace=0.28, hspace=0.22)
    upper_axes = []
    for i, cut in enumerate(CUTS):
        r = runs[cut]; n = r["n"]
        t = r["time"][:n]; ex = ref["raw"][j][:n]
        raw, cor, vb = r["raw"][j][:n], r["cor"][j][:n], r["vb"][j]
        cell = GridSpecFromSubplotSpec(2, 1, subplot_spec=outer[i // 4, i % 4],
                                       height_ratios=[2, 1], hspace=0.17)
        ax = fig.add_subplot(cell[0]); axs = fig.add_subplot(cell[1], sharex=ax)
        style(ax); style(axs)

        ax.plot(t, ex, color=C_EXACT, lw=3.0, zorder=3)
        ax.plot(t, raw, color=C_RAW, lw=1.8, ls="--", zorder=4)
        ax.plot(t, cor, color=C_COR, lw=1.8, zorder=5)
        ax.plot(t, vb, color=C_VB, lw=1.8, ls="-.", zorder=6)
        ax.set_title(f"cutoff {cut}", loc="left", pad=6)
        ax.tick_params(labelbottom=False)
        upper_axes.append(ax)

        axs.axhline(0, color=C_EXACT, lw=1.2)
        axs.plot(t, raw - ex, color=C_RAW, lw=1.8, ls="--")
        axs.plot(t, cor - ex, color=C_COR, lw=1.8)
        axs.plot(t, vb - ex, color=C_VB, lw=1.8, ls="-.")
        axs.set_xlim(0, t[-1]); axs.set_xlabel("t", color=INK2, fontsize=11)
        axs.ticklabel_format(axis="y", style="sci", scilimits=(-2, 3), useMathText=False)
        axs.yaxis.get_offset_text().set_fontsize(9)
        axs.yaxis.get_offset_text().set_color(INK2)
        if i % 4 == 0:
            ax.set_ylabel(f"OTOC$_{{{j}}}$(t)", color=INK2, fontsize=11)
            axs.set_ylabel("run \u2212 exact", color=INK2, fontsize=11)

    # shared y-range on the upper panels, as in the original figures
    lo = min(a.get_ylim()[0] for a in upper_axes); hi = max(a.get_ylim()[1] for a in upper_axes)
    for a in upper_axes:
        a.set_ylim(lo, hi)
        a.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.5))

    # legend + notes in the empty eighth slot
    slot = fig.add_subplot(outer[1, 3]); slot.axis("off")
    handles = [Line2D([], [], color=C_EXACT, lw=3.0),
               Line2D([], [], color=C_RAW, lw=1.8, ls="--"),
               Line2D([], [], color=C_COR, lw=1.8),
               Line2D([], [], color=C_VB, lw=1.8, ls="-.")]
    labels = ["exact Trotter (no truncation)", "raw (truncated)", "corrected",
              "Vb corrected (Eq 14)"]
    slot.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.0, 1.0),
                frameon=False, fontsize=12.5, handlelength=2.6)
    vbs = ", ".join(f"{runs[c]['vB']:.2f}" for c in CUTS)
    notes = ("Upper panels: OTOC(t) = ||[W(t), Z$_j$]||$^2$ / 2$^N$, W = Z$_1$.\n"
             "Lower strips: run minus exact Trotter, each on\n"
             "its own scale (read the scale before comparing).\n\n"
             "Exact Trotter = same dt = 0.05, 2nd order, but no\n"
             "truncation (all 4$^{11}$ reachable strings kept), so\n"
             "the strips show truncation error and nothing else.\n\n"
             "Corrected = raw + the OTOC weight removed at every\n"
             "truncation, accumulated over the run.\n\n"
             "Vb = raw + 2 l(t) \u0398(v$_B$ t \u2212 d$_j$), l = 1 \u2212 ||W||$^2$,\n"
             f"v$_B$ fitted per cutoff to raw Csq > {THR_CSQ:g} crossings\n"
             f"(v$_B$ = {vbs}).")
    slot.text(0.0, 0.66, notes, transform=slot.transAxes, va="top", ha="left",
              fontsize=10.2, color=INK2, linespacing=1.2)

    out = os.path.join(OUTDIR, f"N12_site{j}_vb.png")
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)
    print("wrote", out)
