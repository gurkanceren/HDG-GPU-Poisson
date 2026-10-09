"""Recreate manuscript Figures 4 and 6--8 from the saved CSV files."""

from plot_accuracy import main as make_figures_6_to_8
from plot_timing_breakdown import main as make_figure_4


if __name__ == "__main__":
    make_figure_4()
    make_figures_6_to_8()
