"""Create the corrected eight-mesh Figure 4 from saved v1.6 stage timings."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")  # Reproducible file output; no display or GUI backend is needed.
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter


ROOT = Path(__file__).resolve().parent
INPUT = ROOT / "data" / "final_v16_stage_timings.csv"
OUTPUT = ROOT / "figures" / "V6_absolutetimings"
ORDERS = (1, 2, 3, 4)
MESHES = tuple(range(1, 9))

# CSV column, printed legend label, and bar color.
STAGES = (
    ("mesh_loading", "Mesh loading", "#2166ac"),
    ("mesh_processing", "Mesh processing", "#4393c3"),
    ("reference_element", "Reference element", "#92c5de"),
    ("move_to_gpu", "Move to GPU", "#d1e5f0"),
    ("assembly_matrix", "Matrix assembly", "#238b45"),
    ("project_faces", "Project faces", "#41ab5d"),
    ("reduce_system", "Reduce system", "#a1d99b"),
    ("solve_reduced_system", "Solve reduced system", "#d7301f"),
    ("reconstruct_system", "Reconstruct system", "#756bb1"),
    ("reconstruct_elemental_solution", "Reconstruct elemental solution", "#9e9ac8"),
    ("reference_element_post", "Ref. element (post)", "#e6550d"),
    ("data_preparation_post", "Data preparation (post)", "#fd8d3c"),
    ("gpu_postprocess", "GPU postprocess", "#fdae6b"),
    ("move_to_cpu", "Move to CPU", "#fdd0a2"),
    ("compute_errors", "Compute errors", "#fdd49e"),
)

LEGEND_GROUPS = (
    ("Initialisation", STAGES[0:4]),
    ("Assembly", STAGES[4:7]),
    ("Solve", STAGES[7:8]),
    ("Reconstruction", STAGES[8:10]),
    ("Post-processing", STAGES[10:13]),
    ("", STAGES[13:15]),
)


def load_timings() -> dict[tuple[int, int], dict]:
    """Index each CSV row by (mesh, p)."""
    with INPUT.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    timings = {(int(row["mesh_number"]), int(row["order"])): row for row in rows}
    expected = {(mesh, p) for mesh in MESHES for p in ORDERS}
    if set(timings) != expected:
        raise ValueError("Expected p=1,...,4 and meshes 1,...,8")
    return timings


def configure_style() -> None:
    # Start from a clean style even if another figure was made in this process.
    mpl.rcdefaults()
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["STIX Two Text", "Times New Roman", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 12,
            "axes.labelsize": 14,
            "axes.titlesize": 17,
            "xtick.labelsize": 11,
            "ytick.labelsize": 12,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def trace_dof_factor(timings, mesh) -> int:
    """Recover the factor in N_trace = factor * (p+1)."""
    factors = [round(float(timings[mesh, p]["ndofs"]) / (p + 1)) for p in ORDERS]
    if len(set(factors)) != 1:
        raise ValueError(f"Inconsistent trace-DoF factor on mesh {mesh}: {factors}")
    return factors[0]


def add_legend(figure) -> None:
    """Place the six algorithmic groups below the bars."""
    anchors = (0.075, 0.245, 0.405, 0.535, 0.735, 0.895)
    for x, (title, stages) in zip(anchors, LEGEND_GROUPS):
        handles = [
            Patch(facecolor=color, edgecolor="white", linewidth=0.35, label=label)
            for _, label, color in stages
        ]
        legend = figure.legend(
            handles=handles, loc="lower left", bbox_to_anchor=(x, 0.018),
            frameon=False, fontsize=10.5, title=title or None,
            title_fontproperties={"weight": "bold", "size": 10.5},
            handlelength=1.1, handleheight=1.0, labelspacing=0.35,
            borderaxespad=0,
        )
        figure.add_artist(legend)


def main() -> None:
    configure_style()
    timings = load_timings()

    figure, axis = plt.subplots(figsize=(19.96, 7.88))
    group_spacing = 1.22
    bar_width = 0.155
    centers = np.arange(8) * group_spacing
    offsets = (np.arange(4) - 1.5) * bar_width

    bars = []
    for mesh_index, mesh in enumerate(MESHES):
        for order_index, p in enumerate(ORDERS):
            row = timings[mesh, p]
            x = centers[mesh_index] + offsets[order_index]
            bottom = 0.0
            for column, _, color in STAGES:
                value = float(row[column])
                axis.bar(
                    x, value, width=0.90 * bar_width, bottom=bottom,
                    color=color, edgecolor="white", linewidth=0.35,
                )
                bottom += value
            bars.append((x, float(row["total"]), p))

    maximum = max(total for _, total, _ in bars)
    for x, total, p in bars:
        axis.text(
            x, total + 0.014 * maximum, f"{1000 * total:.0f} ms",
            ha="center", va="bottom", rotation=90, fontsize=8,
        )
        axis.text(
            x, -0.030, fr"$p={p}$", transform=axis.get_xaxis_transform(),
            ha="center", va="top", fontsize=9,
        )

    labels = [
        f"Mesh {mesh}\n({trace_dof_factor(timings, mesh):,} · $(p+1)$ DOFs)"
        for mesh in MESHES
    ]
    axis.set_xticks(centers)
    axis.set_xticklabels(labels, fontsize=12)
    axis.tick_params(axis="x", pad=52, length=0)
    axis.set_ylabel("Execution time (s)")
    axis.set_title("HDG GPU Solver — Absolute Execution Time", pad=12)
    axis.set_ylim(0, 1.085 * maximum)
    axis.yaxis.set_major_formatter(
        FuncFormatter(lambda value, _: "0 ms" if value == 0 else f"{value:.2f} s")
    )
    axis.grid(axis="y", linestyle="--", linewidth=0.65, color="#d5d5d5")
    axis.set_axisbelow(True)
    axis.spines[["top", "right"]].set_visible(False)

    # Dotted lines visually separate the eight mesh groups.
    for left, right in zip(centers[:-1], centers[1:]):
        separator = 0.5 * (left + right)
        axis.plot(
            [separator, separator], [-0.14, 1],
            transform=axis.get_xaxis_transform(), color="#d3d3d3",
            linestyle=":", linewidth=0.8, clip_on=False, zorder=0,
        )

    add_legend(figure)
    figure.add_artist(
        Line2D([0.055, 0.985], [0.205, 0.205], transform=figure.transFigure,
               color="black", linewidth=0.75)
    )
    figure.subplots_adjust(left=0.055, right=0.985, top=0.92, bottom=0.36)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT.with_suffix(".pdf"))
    figure.savefig(OUTPUT.with_suffix(".svg"))
    figure.savefig(OUTPUT.with_suffix(".png"), dpi=100)
    plt.close(figure)

    stage_sum = np.array([
        sum(float(row[column]) for column, _, _ in STAGES)
        for row in timings.values()
    ])
    total = np.array([float(row["total"]) for row in timings.values()])
    mismatch = np.max(np.abs(stage_sum - total) / total)
    print(f"Created Figure 4 in {OUTPUT.parent}")
    print(f"Maximum |stage sum - total| / total: {mismatch:.3%}")


if __name__ == "__main__":
    main()
