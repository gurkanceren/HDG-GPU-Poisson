"""Create manuscript Figures 6--8 from the unified 32-case summary CSV."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")  # Reproducible file output; no display or GUI backend is needed.
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
INPUT = ROOT / "data" / "final_unified_summary.csv"
OUTPUT = ROOT / "figures"

ORDERS = (1, 2, 3, 4)
COLORS = {1: "#0072B2", 2: "#E69F00", 3: "#009E73", 4: "#CC79A7"}
MARKERS = {1: "o", 2: "s", 3: "^", 4: "D"}


def load_results() -> list[dict]:
    """Read only the CSV columns that appear in Figures 6--8."""
    rows = []
    with INPUT.open(newline="") as stream:
        for source in csv.DictReader(stream):
            rows.append(
                {
                    "p": int(source["order"]),
                    "mesh": int(source["mesh"]),
                    "ndof": int(source["ndof_trace"]),
                    "time": float(source["time_median"]),
                    "q25": float(source["time_q25"]),
                    "q75": float(source["time_q75"]),
                    "error_u": float(source["error_u_median"]),
                    "error_post": float(source["error_post_median"]),
                }
            )

    expected = {(p, mesh) for p in ORDERS for mesh in range(1, 9)}
    if {(row["p"], row["mesh"]) for row in rows} != expected:
        raise ValueError("Expected p=1,...,4 and meshes 1,...,8")
    return rows


def configure_style() -> None:
    # Matplotlib settings are global. Reset first so Figure 4 cannot alter this
    # figure when both modules are run by make_all_figures.py.
    mpl.rcdefaults()
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["STIX Two Text", "Times New Roman", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 9.2,
            "axes.labelsize": 10,
            "legend.fontsize": 8,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "axes.linewidth": 0.8,
            "lines.linewidth": 1.65,
            "lines.markersize": 5.8,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.08,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save(figure: plt.Figure, name: str) -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT / f"{name}.pdf")
    figure.savefig(OUTPUT / f"{name}.svg")
    figure.savefig(OUTPUT / f"{name}.png", dpi=600)
    plt.close(figure)


def format_axis(axis: plt.Axes) -> None:
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.grid(True, which="major", color="#d9d9d9", linewidth=0.55)
    axis.grid(True, which="minor", color="#eeeeee", linewidth=0.35)
    axis.tick_params(which="both", direction="in", top=True, right=True)


def one_degree(rows: list[dict], p: int) -> list[dict]:
    """Return the eight meshes for degree p, from coarse to fine."""
    return sorted((row for row in rows if row["p"] == p), key=lambda row: row["mesh"])


def draw_degree_curves(axis, rows, x_name, error_name) -> None:
    """Draw the four curves shared by Figures 6 and 7."""
    for p in ORDERS:
        series = one_degree(rows, p)
        x = np.array([row[x_name] for row in series])
        error = np.array([row[error_name] for row in series])

        # Markers through the minimum are filled; later meshes are open.
        minimum_index = int(np.argmin(error))
        open_marker = np.arange(len(series)) > minimum_index

        axis.plot(x, error, color=COLORS[p], label=fr"$p={p}$")
        if x_name == "time":
            q25 = np.array([row["q25"] for row in series])
            q75 = np.array([row["q75"] for row in series])
            axis.errorbar(
                x,
                error,
                xerr=np.vstack([x - q25, q75 - x]),
                fmt="none",
                ecolor=COLORS[p],
                elinewidth=0.75,
                capsize=1.8,
                alpha=0.75,
            )
        axis.scatter(
            x[~open_marker], error[~open_marker],
            marker=MARKERS[p], facecolor=COLORS[p], edgecolor="white",
            linewidth=0.55, zorder=3,
        )
        axis.scatter(
            x[open_marker], error[open_marker],
            marker=MARKERS[p], facecolor="white", edgecolor=COLORS[p],
            linewidth=1.25, zorder=4,
        )
    format_axis(axis)


def make_curve_figure(rows, x_name, x_label, output_name) -> None:
    """Make Figure 6 when x=time or Figure 7 when x=ndof."""
    figure, axes = plt.subplots(1, 2, figsize=(7.15, 3.05), constrained_layout=True)
    panels = (
        ("error_u", r"(a) HDG solution $u_h$"),
        ("error_post", r"(b) Postprocessed solution $u_h^\star$"),
    )
    for axis, (error_name, title) in zip(axes, panels):
        draw_degree_curves(axis, rows, x_name, error_name)
        axis.set_xlabel(x_label)
        axis.set_ylabel(r"$L^2(\Omega)$ error")
        axis.set_title(title, pad=5)
    axes[0].legend(ncol=2, frameon=False, loc="lower left")

    if x_name == "time":
        axes[1].text(
            0.98, 0.98,
            "filled: decreasing-error branch\nopen: beyond the minimum",
            transform=axes[1].transAxes, ha="right", va="top", fontsize=7.4,
            bbox={"facecolor": "white", "edgecolor": "#bdbdbd", "pad": 3.0},
        )
    save(figure, output_name)


def pareto_frontier(rows, error_name) -> list[dict]:
    """Keep a point only when it improves on every faster point."""
    frontier = []
    best_error = np.inf
    for row in sorted(rows, key=lambda item: (item["time"], item[error_name])):
        if row[error_name] < best_error:
            frontier.append(row)
            best_error = row[error_name]
    return frontier


def make_pareto_figure(rows) -> None:
    """Make Figure 8."""
    figure, axes = plt.subplots(1, 2, figsize=(7.15, 3.05), constrained_layout=True)
    panels = (
        ("error_u", r"(a) HDG solution $u_h$"),
        ("error_post", r"(b) Postprocessed solution $u_h^\star$"),
    )
    for axis, (error_name, title) in zip(axes, panels):
        for p in ORDERS:
            series = one_degree(rows, p)
            axis.scatter(
                [row["time"] for row in series],
                [row[error_name] for row in series],
                color=COLORS[p], marker=MARKERS[p], alpha=0.5,
                edgecolor="none", label=fr"$p={p}$",
            )
        frontier = pareto_frontier(rows, error_name)
        axis.plot(
            [row["time"] for row in frontier],
            [row[error_name] for row in frontier],
            color="#202020", linewidth=2.1, marker="o", markersize=3.6,
            markerfacecolor="white", markeredgecolor="#202020",
            label="Pareto frontier", zorder=5,
        )
        axis.set_xlabel(r"End-to-end GPU pipeline time, $T$ [s]")
        axis.set_ylabel(r"$L^2(\Omega)$ error")
        axis.set_title(title, pad=5)
        format_axis(axis)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, ncol=5, frameon=False, loc="outside lower center")
    save(figure, "time_accuracy_pareto_fp32")


def main() -> None:
    configure_style()
    rows = load_results()
    make_curve_figure(
        rows, "time", r"End-to-end GPU pipeline time, $T$ [s]",
        "time_to_accuracy_fp32",
    )
    make_curve_figure(
        rows, "ndof", r"Trace degrees of freedom, $N_{\widehat{u}}$",
        "dof_to_accuracy_fp32",
    )
    make_pareto_figure(rows)
    print(f"Created Figures 6--8 in {OUTPUT}")


if __name__ == "__main__":
    main()
