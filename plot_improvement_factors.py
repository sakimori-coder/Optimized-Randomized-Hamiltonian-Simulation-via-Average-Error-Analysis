"""Plot the chemistry and SYK CSVs as PDFs in the paper's figure style."""

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


plt.rcParams.update({
    "font.family": "STIXGeneral",
    "mathtext.fontset": "stix",
    "axes.labelsize": 24,
    "xtick.labelsize": 20,
    "ytick.labelsize": 20,
})
RESULTS_DIR = Path(__file__).resolve().parent / "results"
METRICS = (
    ("I_M", r"$I_M$: Error upper bound", "#1f77b4"),
    ("I_r", r"$I_r$: Average infidelity", "#d95f02"),
    ("I_sig", r"$I_{\mathrm{sig}}$: Average signal error", "#2ca02c"),
)
MOLECULE_LABELS = {
    "h2_sto3g_jw": r"$\mathrm{H}_2$",
    "lih_sto3g_full_jw": r"$\mathrm{LiH}$",
    "beh2_sto3g_full_jw": r"$\mathrm{BeH}_2$",
    "h2o_sto3g_full_jw": r"$\mathrm{H}_2\mathrm{O}$",
    "nh3_sto3g_full_jw": r"$\mathrm{NH}_3$",
    "ch4_sto3g_full_jw": r"$\mathrm{CH}_4$",
    "n2_sto3g_full_jw": r"$\mathrm{N}_2$",
    "h2o_ccpvdz_full_jw": r"$\mathrm{H}_2\mathrm{O}$" + "\ncc-pVDZ",
    "ch4_ccpvdz_full_jw": r"$\mathrm{CH}_4$" + "\ncc-pVDZ",
    "femoco_reiher_54e_54o_jw": r"$\mathrm{FeMoco}$",
}


def save_figure(fig, ax, name):
    ax.axhline(1, color="black", linestyle="--", linewidth=1)
    ax.set_ylabel("Improvement factor")
    ax.legend(loc="upper left", fontsize=20, frameon=False)
    path = RESULTS_DIR / f"{name}.pdf"
    fig.savefig(path)
    print(f"Saved {path}")
    plt.close(fig)


def plot_chemistry():
    with (RESULTS_DIR / "chemistry_improvement_factors.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))

    fig, ax = plt.subplots(figsize=(12, 6.4), layout="constrained")
    positions = np.arange(len(rows))
    for index, (key, label, color) in enumerate(METRICS):
        # A blank cell means the metric was not calculated, not zero.
        values = np.array([float(row[key]) if row[key] else np.nan for row in rows])
        finite = np.isfinite(values)
        if finite.any():
            ax.bar(positions[finite] + (index - 1) * 0.25, values[finite], width=0.25,
                   color=color, edgecolor="black", linewidth=0.5, label=label)

    labels = [MOLECULE_LABELS[row["molecule"]] + f"\n$n = {row['num_qubits']}$" for row in rows]
    ax.set_xticks(positions, labels)
    ax.set_xlabel("Molecular system")
    ax.grid(axis="y", alpha=0.25)
    save_figure(fig, ax, "molecular_improvement_factors")


def plot_syk():
    with (RESULTS_DIR / "syk_improvement_factors.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))

    # Match the paper's panel widths: SYK 0.455, molecules 0.525 of textwidth.
    fig, ax = plt.subplots(figsize=(12 * 0.455 / 0.525, 6.4), layout="constrained")
    for key, label, color in METRICS:
        samples = {}
        for row in rows:
            if row[key]:
                samples.setdefault(int(row["num_qubits"]), []).append(float(row[key]))
        if not samples:
            continue
        qubits = sorted(samples)
        # Take the median of the individual realizations' improvement ratios.
        median = [np.median(samples[n]) for n in qubits]
        ax.plot(qubits, median, linewidth=1.7, color=color, label=label)

    ax.set_xlabel("Number of qubits")
    ax.grid(which="major", alpha=0.25)
    save_figure(fig, ax, "syk_improvement_factors")


if __name__ == "__main__":
    plot_chemistry()
    plot_syk()
