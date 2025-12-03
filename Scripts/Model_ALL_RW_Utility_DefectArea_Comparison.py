#!/usr/bin/env python3
"""
Written By: Giovanni Zattoni

RW 'Utility' Figure — Average defect area % vs. eliminated error source
=======================================================================

Changes in this version:
- Programmed shift is a CONSTANT per scenario (set in SCENARIO_SHIFT_MM).
- For the "No LLS_A" scenario ONLY, we impose:
      compaction_error(x) = min( −(LLS_B(x) − LLS_A(x)), 0 )
  and use:
      w_NoLLSA(x) = NOMINAL + compaction_error(x)
- For the "No LLS_B" scenario, width is FIXED at the nominal value:
      w_NoLLSB(x) = NOMINAL  (constant over x)

Elimination semantics (FINAL; no LB > LA anywhere):
- "No <SOURCE>" means that source is set to ZERO in the model (perfect world).
- Width construction rules (LA & LB aligned at the same x indices):
    • All errors / No LT / No CAM: w = NOMINAL + LB
    • No LLS_A:                    w = NOMINAL + min( −(LB − LA), 0 )
    • No LLS_B:                    w = NOMINAL

Programmed shift rule (constant per scenario):
----------------------------------------------
For every scenario, in every run:
    programmed_shift = SCENARIO_SHIFT_MM[scenario_key]
where scenario_key ∈ {None, "CAM", "LT", "LLS_A", "LLS_B"}.

Outputs generated:
------------------
- rw_defect_barchart.png — bar chart comparing average Gap% and Overlap%
- rw_defect_summary.csv — CSV summary with Gap%, Overlap%, and shift stats (std=0 here)
"""

##############################################################################################################

# External imports
import os
import sys
import csv
import textwrap
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter

REPO_ROOT = os.path.dirname(__file__)
SCRIPTS_DIR = os.path.join(REPO_ROOT, "Scripts")
sys.path.insert(0, SCRIPTS_DIR)

# Internal imports
from Model_ALL_RandomWalk import fit_random_walk, generate_random_walk

##############################################################################################################
""""Functions and constants"""

# ---------------------------------------------------------------------
# Simulation configuration
# ---------------------------------------------------------------------
N_RUNS = 2                 # Number of simulations per scenario (for averaging)
NUM_TOWS = 31               # Number of parallel tows across the lane
NOMINAL_WIDTH_MM = 6.35     # Nominal tow width [mm]
PROPOSAL_TYPE = "RWM"       # Proposal type for random walk sampling
RANDOM_SEED = 59            # For reproducibility; set to None for random

# >>>>>> Constant programmed shifts per scenario (mm)
SCENARIO_SHIFT_MM = {
    None:     6.35,   # "All four variations" (baseline)
    "CAM":    6.35,   # "No tape lateral movement"
    "LT":     6.35,   # "No robot inaccuracy"
    "LLS_A":  6.35,   # "No width variation"
    "LLS_B":  6.4321, # "No compaction variation"
}

# Output file locations
FIG_PATH = os.path.join(REPO_ROOT, "rw_defect_barchart.pdf")
CSV_PATH = os.path.join(REPO_ROOT, "rw_defect_summary.csv")

# Y-axis formatting parameters (major step only; minors off)
Y_MAJOR_STEP = 1.0
Y_DECIMALS = 2

# ---------------------------------------------------------------------
# Global plotting style (Times New Roman, journal-style)
# ---------------------------------------------------------------------
plt.rcParams.update({
    # Fonts
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "Nimbus Roman No9 L"],
    "mathtext.fontset": "stix",

    # Axes & layout
    "axes.grid": False,
    "axes.edgecolor": "black",
    "axes.linewidth": 1.0,
    "figure.figsize": (9.5, 4.8),
    "axes.labelsize": 12,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,

    # Tick appearance
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.major.size": 4.5,
    "ytick.major.size": 4.5,
    "xtick.minor.size": 0.0,  # not used (minors off)
    "ytick.minor.size": 0.0,  # not used (minors off)

    # Legend styling
    "legend.frameon": True,
    "legend.edgecolor": "black",
})

# Color constants for Gap and Overlap bars
PASTEL_BLUE = "#A7C7E7"   # Gap
PASTEL_RED  = "#F4A6A6"   # Overlap
# Thin reference line colors (darker)
LINE_BLUE = "#1f77b4"
LINE_RED  = "#d62728"

# ---------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------
def _interp_to(arr: np.ndarray, m: int) -> np.ndarray:
    """Linearly interpolate a 1D array to have 'm' uniform samples."""
    x_old = np.linspace(0, 1, len(arr))
    x_new = np.linspace(0, 1, m)
    return np.interp(x_new, x_old, arr)

def _gap_overlap_percent_envelope(centerlines, widths, dx=1.0):
    """
    Compute average Gap% and Overlap% across the layup width envelope.

    For each adjacent tow pair i and i+1:
      diff(x) = bottom_{i+1}(x) - top_i(x)
      gap(x)  = max(diff, 0)
      ovl(x)  = max(-diff, 0)

    Then normalize by the total width envelope area:
      ∫ (max(top_edges) - min(bottom_edges)) dx
    """
    n_tows = len(centerlines)
    assert n_tows == len(widths)

    # Top/bottom tow edge profiles
    top_edges = [c + w / 2 for c, w in zip(centerlines, widths)]
    bot_edges = [c - w / 2 for c, w in zip(centerlines, widths)]

    # Initialize total integrated areas
    total_gap_area = 0.0
    total_ovl_area = 0.0

    # Integrate pairwise differences between adjacent tows
    for i in range(n_tows - 1):
        diff = bot_edges[i + 1] - top_edges[i]
        gap = np.clip(diff, 0, None)
        ovl = np.clip(-diff, 0, None)
        total_gap_area += float(np.trapezoid(gap, dx=dx))
        total_ovl_area += float(np.trapezoid(ovl, dx=dx))

    # Envelope normalization
    highest_top = np.maximum.reduce(top_edges)
    lowest_bot  = np.minimum.reduce(bot_edges)
    total_layout_area = float(np.trapezoid(highest_top - lowest_bot, dx=dx))

    # Convert to percentage
    gap_pct = 100.0 * total_gap_area / total_layout_area if total_layout_area > 0 else 0.0
    ovl_pct = 100.0 * total_ovl_area / total_layout_area if total_layout_area > 0 else 0.0
    return gap_pct, ovl_pct

def _wrap_labels(labels, width=16):
    """Wrap long x-axis labels to prevent crowding."""
    return [textwrap.fill(s, width=width) for s in labels]

# ---------------------------------------------------------------------
# Core simulation per layout (uses constant programmed_shift)
# ---------------------------------------------------------------------
def simulate_layout(num_tows=6, eliminate=None, programmed_shift_mm=6.35, log_shift=False):
    """
    Simulate a single multi-tow layup for a given elimination scenario.

    Each error source (LT, CAM, LLS_A, LLS_B) is modeled as a random walk.
    Widths and centerlines are constructed according to the elimination rules.
    Programmed shift is a CONSTANT provided via 'programmed_shift_mm'.

    Special rules:
      • No LLS_A: compaction_error = min( −(LB − LA), 0 ), width = NOMINAL + compaction_error
      • No LLS_B: width = NOMINAL (constant over x)
    """
    # ---- Step 1: Fit random-walk distributions for each error source ----
    LT_steps, LT_prop, LT_tgt, LT_dist, LT_params = fit_random_walk("LT")
    CAM_steps, CAM_prop, CAM_tgt, CAM_dist, CAM_params = fit_random_walk("CAM")
    LA_steps, LA_prop, LA_tgt, LA_dist, LA_params = fit_random_walk("LLS_A")
    LB_steps, LB_prop, LB_tgt, LB_dist, LB_params = fit_random_walk("LLS_B")

    # Interpolate all to same number of steps
    n_steps = min(LT_steps, CAM_steps, LA_steps, LB_steps)

    # Sampling functions
    def sample_LT():  return generate_random_walk("LT",  LT_steps,  LT_prop,  LT_tgt,  LT_dist,  LT_params,  proposal_type=PROPOSAL_TYPE)
    def sample_CAM(): return generate_random_walk("CAM", CAM_steps, CAM_prop, CAM_tgt, CAM_dist, CAM_params, proposal_type=PROPOSAL_TYPE)
    def sample_LA():  return generate_random_walk("LLS_A", LA_steps, LA_prop, LA_tgt, LA_dist, LA_params, proposal_type=PROPOSAL_TYPE)
    def sample_LB():  return generate_random_walk("LLS_B", LB_steps, LB_prop, LB_tgt, LB_dist, LB_params, proposal_type=PROPOSAL_TYPE)

    # ---- Step 2: Generate realizations for all tows ----
    LT_arr, CAM_arr, LA_arr, LB_arr = [], [], [], []
    for _ in range(num_tows):
        LT_arr.append(_interp_to(sample_LT(),  n_steps))
        CAM_arr.append(_interp_to(sample_CAM(), n_steps))
        LA_arr.append(_interp_to(sample_LA(),  n_steps))
        LB_arr.append(_interp_to(sample_LB(),  n_steps))

    LT_arr, CAM_arr, LA_arr, LB_arr = map(np.stack, [LT_arr, CAM_arr, LA_arr, LB_arr])

    # ---- Step 3: Apply elimination logic ----
    if eliminate == "LT":
        LT_arr[:] = 0.0
    if eliminate == "CAM":
        CAM_arr[:] = 0.0

    # ---- Step 4: CONSTANT programmed shift (provided) ----
    programmed_shift = float(programmed_shift_mm)
    if log_shift:
        print(f"[shift] Scenario={eliminate or 'None'}, programmed_shift={programmed_shift:.6f} mm")

    # ---- Step 5: Build tow centerlines and widths ----
    centerlines, widths = [], []
    for i in range(num_tows):
        base_offset = i * programmed_shift
        center = base_offset + LT_arr[i] + CAM_arr[i]

        # Final widths per scenario
        if eliminate == "LLS_B":
            # No compaction variation: width strictly equals the nominal width (constant over x)
            width = np.full(n_steps, NOMINAL_WIDTH_MM, dtype=float)

        elif eliminate == "LLS_A":
            # No width-before variation: use rectified negative compaction error
            # (i.e., if LB - LA > 0 -> 0; if LB - LA < 0 -> -(LB - LA))
            compaction_error = np.minimum(-(LB_arr[i] - LA_arr[i]), 0.0)
            width = NOMINAL_WIDTH_MM + compaction_error

        else:
            # All / No LT / No CAM: use width after compaction (LB)
            width = NOMINAL_WIDTH_MM + LB_arr[i]

        # Safety floor to avoid zero/negative widths
        pre_clip = width.copy()
        width = np.clip(width, 1e-6, None)
        if np.any(width != pre_clip):
            clipped_pts = int(np.count_nonzero(width != pre_clip))
            pct = 100.0 * clipped_pts / n_steps
            print(f"[clip] Scenario={eliminate or 'None'}, Tow={i}, clipped={clipped_pts}/{n_steps} ({pct:.2f}%)")

        centerlines.append(center)
        widths.append(width)

    # ---- Step 6: Evaluate total defect area ----
    gap_pct, ovl_pct = _gap_overlap_percent_envelope(centerlines, widths, dx=1.0)
    return gap_pct, ovl_pct, programmed_shift

# ---------------------------------------------------------------------
# Batch experiment (averages multiple runs per scenario)
# ---------------------------------------------------------------------
def run_experiment(n_runs=100, num_tows=6):
    """
    Run multiple simulations per elimination scenario and average results.
    Programmed shift is a constant per scenario (SCENARIO_SHIFT_MM).
    """
    scenarios = [
        ("All four variations",        None),    # baseline
        ("No tape lateral movement",   "CAM"),   # CAM off
        ("No robot inaccuracy",        "LT"),    # LT off
        # ("No width variation",       "LLS_A"), # width before off (uses rectified compaction error)
        ("No width variation or compaction variation", "LLS_B"),  # width fixed to NOMINAL
    ]

    results = {label: {"gap": [], "ovl": [], "shift": []} for label, _ in scenarios}
    total_iters = n_runs * len(scenarios)
    done = 0

    print(f"Running {total_iters} simulations...\n")
    for label, elim in scenarios:
        const_shift = SCENARIO_SHIFT_MM[elim]  # look up constant shift for this scenario
        for _ in range(n_runs):
            log_shift = (elim in ("LLS_A", "LLS_B"))  # keep earlier logging behavior
            g, o, shift = simulate_layout(num_tows, eliminate=elim, programmed_shift_mm=const_shift, log_shift=log_shift)
            results[label]["gap"].append(g)
            results[label]["ovl"].append(o)
            results[label]["shift"].append(shift)

            done += 1
            if done % 10 == 0 or done == total_iters:
                print(f"Progress: {done}/{total_iters} runs complete")

    print("Simulation complete.\n")

    # Compute mean ± std per scenario (std of shift is zero now)
    summary = {}
    for label, _ in scenarios:
        v = results[label]
        gap_avg = np.mean(v["gap"]) if v["gap"] else 0.0
        ovl_avg = np.mean(v["ovl"]) if v["ovl"] else 0.0
        shift_avg = np.mean(v["shift"]) if v["shift"] else SCENARIO_SHIFT_MM[None]
        shift_std = 0.0  # constant per run
        summary[label] = (gap_avg, ovl_avg, shift_avg, shift_std)
    return scenarios, summary

# ---------------------------------------------------------------------
# Plotting function (adds thin reference lines at tallest Gap & Overlap bars)
# ---------------------------------------------------------------------
def plot_barchart(scenarios, summary, save_path):
    labels = [s[0] for s in scenarios]
    gap_means = [summary[l][0] for l in labels]
    ovl_means = [summary[l][1] for l in labels]
    x = np.arange(len(labels))
    width = 0.30  # bar width

    # --------- Pre-compute reference line positions ----------
    gap_line = None
    if gap_means:
        i_max_g = int(np.argmax(gap_means))
        y_g = float(gap_means[i_max_g])
        # GAP bars: centers at (x - width/2) → span [x - width, x]
        x_left_edge_tallest_gap = x[i_max_g] - width      # left edge of tallest GAP bar
        x_right_edge_last_gap   = x[-1]                   # right edge of last GAP bar
        gap_line = (y_g, x_left_edge_tallest_gap, x_right_edge_last_gap)

    ovl_line = None
    if ovl_means:
        i_max_o = int(np.argmax(ovl_means))
        y_o = float(ovl_means[i_max_o])
        # OVERLAP bars: centers at (x + width/2) → span [x, x + width]
        x_left_edge_tallest_ovl = x[i_max_o]              # left edge of tallest OVL bar
        x_right_edge_last_ovl   = x[-1] + width           # right edge of last OVL bar
        ovl_line = (y_o, x_left_edge_tallest_ovl, x_right_edge_last_ovl)
    # --------------------------------------------------------

    fig, ax = plt.subplots()

    # ------- Draw reference lines FIRST (behind the bars) -------
    if gap_line is not None:
        y_g, xg_min, xg_max = gap_line
        if xg_max > xg_min:
            ax.hlines(
                y_g,
                xmin=xg_min,
                xmax=xg_max,
                colors=LINE_BLUE,
                linestyles='-',
                linewidth=0.9,
                zorder=1,      # behind bars
            )

    if ovl_line is not None:
        y_o, xo_min, xo_max = ovl_line
        if xo_max > xo_min:
            ax.hlines(
                y_o,
                xmin=xo_min,
                xmax=xo_max,
                colors=LINE_RED,
                linestyles='-',
                linewidth=0.9,
                zorder=1,      # behind bars
            )
    # ------------------------------------------------------------

    # Bars (give them higher zorder so they’re in front of lines)
    ax.bar(x - width/2, gap_means, width,
           label="Gap", color=PASTEL_BLUE, zorder=2)
    ax.bar(x + width/2, ovl_means, width,
           label="Overlap", color=PASTEL_RED, zorder=2)

    # Axis labels / ticks
    ax.set_ylabel("Defect area %")
    wrapped = _wrap_labels(labels, width=18)
    ax.set_xticks(x, wrapped, rotation=0, ha="center")
    ax.set_ylim(0, 5)
    ax.yaxis.set_major_locator(MultipleLocator(Y_MAJOR_STEP))
    ax.yaxis.set_major_formatter(FormatStrFormatter(f"%.{Y_DECIMALS}f"))
    ax.minorticks_off()
    ax.tick_params(axis='y',which='both',direction='in',
                   length=8,width=1.2,left=True,right=True,)


    ax.legend(loc="best", frameon=False)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.22)
    fig.savefig(save_path, bbox_inches="tight", format="pdf")
    print(f"[✓] Saved figure: {save_path}")




# ---------------------------------------------------------------------
# CSV saving
# ---------------------------------------------------------------------
def save_csv(scenarios, summary, save_path):
    """Save results to CSV file for reproducibility or supplemental material."""
    with open(save_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Scenario",
            "Gap % (avg)",
            "Overlap % (avg)",
            "Programmed shift (constant) [mm]",
            "Programmed shift std (mm)"
        ])
        for label, _ in scenarios:
            gap, ovl, shift_avg, shift_std = summary[label]
            writer.writerow([label, f"{gap:.6f}", f"{ovl:.6f}", f"{shift_avg:.6f}", f"{shift_std:.6f}"])
    print(f"[✓] Saved CSV: {save_path}")

##############################################################################################################
""""Run this file"""

def main():
    # Ensure reproducibility
    if RANDOM_SEED is not None:
        np.random.seed(RANDOM_SEED)

    # Run experiment and generate figure
    scenarios, summary = run_experiment(N_RUNS, NUM_TOWS)
    plot_barchart(scenarios, summary, FIG_PATH)
    save_csv(scenarios, summary, CSV_PATH)

if __name__ == "__main__":
    main()
