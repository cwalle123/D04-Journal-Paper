import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

from constants import tow_width_specified
from Model_ALL_RandomWalk import generate_RW_multitow
from Data_ALL_traverse import traverse_tow_constructor

def traverse_tow_gaps_and_overlaps_selected(
        tows=None,
        plot=True,
        tow_spacing=None,
        print_statement=True):
    """
    Compute gap/overlap percentages for selected real traverse tows.

    Example:
        tows = list(range(16, 31))

    Important:
        traverse_tow_constructor only works for tows 2...30.
    """

    if tows is None:
        tows = list(range(16, 31))

    tows = list(tows)

    if tow_spacing is None:
        tow_spacing = tow_width_specified

    for tow in tows:
        if tow not in range(2, 31):
            raise ValueError(
                f"Tow {tow} cannot be reconstructed from traverse data. "
                f"Use tows between 2 and 30."
            )

    top_edge_paths = []
    bottom_edge_paths = []
    x_vals_list = []

    for local_idx, tow in enumerate(tows):

        traverse_tow = traverse_tow_constructor(tow, normalize=True)

        if traverse_tow is None:
            continue

        offset_mm = local_idx * tow_spacing

        x_vals_list.append(traverse_tow["x_centerline"].to_numpy())

        top_edge_paths.append(
            traverse_tow["y_left"].to_numpy() + offset_mm
        )

        bottom_edge_paths.append(
            traverse_tow["y_right"].to_numpy() + offset_mm
        )

    min_len = min(len(arr) for arr in x_vals_list)

    x_vals = x_vals_list[0][:min_len]

    top_edge_paths = [arr[:min_len] for arr in top_edge_paths]
    bottom_edge_paths = [arr[:min_len] for arr in bottom_edge_paths]

    gap_overlap_dict = {
        f"Gap/overlap_Tow{tows[tow_idx]}_Tow{tows[tow_idx + 1]}":
            bottom_edge_paths[tow_idx + 1] - top_edge_paths[tow_idx]
        for tow_idx in range(len(top_edge_paths) - 1)
    }

    gap_overlap_df = pd.DataFrame(gap_overlap_dict, index=x_vals)

    gap_df = gap_overlap_df.where(gap_overlap_df > 0)
    overlap_df = gap_overlap_df.where(gap_overlap_df < 0)

    highest_tow_edge = top_edge_paths[-1]
    lowest_tow_edge = bottom_edge_paths[0]

    total_layout_area = np.trapezoid(highest_tow_edge - lowest_tow_edge, x_vals)

    total_gap_area = sum(
        np.trapezoid(np.clip(values, 0, None), x_vals)
        for values in gap_overlap_df.values.T
    )

    total_overlap_area = sum(
        np.trapezoid(np.clip(-values, 0, None), x_vals)
        for values in gap_overlap_df.values.T
    )

    gap_percent = (total_gap_area / total_layout_area) * 100 if total_layout_area > 0 else 0
    overlap_percent = (total_overlap_area / total_layout_area) * 100 if total_layout_area > 0 else 0

    if print_statement:
        print("\nExperimental traverse defect percentages")
        print(f"Tows used: {tows}")
        print(f"Total layout area: {total_layout_area:.2f} mm²")
        print(f"Gap area: {total_gap_area:.2f} mm² ({gap_percent:.2f}%)")
        print(f"Overlap area: {total_overlap_area:.2f} mm² ({overlap_percent:.2f}%)")

    if plot:
        plt.figure(figsize=(10, 6))

        for i, (top, bottom) in enumerate(zip(top_edge_paths, bottom_edge_paths)):
            color = plt.get_cmap("tab10")(i % 10)
            tow_number = tows[i]

            plt.plot(
                x_vals,
                (top + bottom) / 2,
                "--",
                color=color,
                label=f"Tow {tow_number} centerline"
            )

            plt.plot(x_vals, top, "-", color=color)
            plt.plot(x_vals, bottom, "-", color=color)

        plt.xlabel("Tow length (mm)")
        plt.ylabel("Tow position (mm)")
        plt.title(f"Traverse Tow Layout for Tows {tows[0]}-{tows[-1]}")
        plt.legend(loc="best", fontsize=8)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    return gap_overlap_df, gap_df, overlap_df, gap_percent, overlap_percent


def calculate_RW_first15_and_exp_last15_defect_percentages(
        n_RW_simulations: int = 100,
        tow_spacing_mm: float = tow_width_specified,
        tow_width_mm: float = tow_width_specified,
        tow_length_mm: float = 1000,
        proposal_type: str = "RWM",
        save_csv: bool = True,
        plot_experimental: bool = False):
    """
    Calculates:

    1. RW/MCMC defect percentages:
       - RW trained on first 15 sensor tows: 1...15
       - Then used to generate a 15-tow virtual laminate

    2. Experimental defect percentages:
       - Real traverse data from last 15 reconstructable tows: 16...30

    Returns:
        rw_results_df
        summary_df
    """

    # RW training tows
    RW_TRAINING_TOWS = list(range(1, 16))

    # Experimental validation tows
    EXPERIMENTAL_TOWS = list(range(16, 31))

    num_tows = len(EXPERIMENTAL_TOWS)

    print("\n===================================================")
    print("RW TRAINING")
    print("===================================================")
    print(f"RW trained on sensor tows: {RW_TRAINING_TOWS}")
    print(f"RW virtual laminate has {num_tows} tows")

    print("\n===================================================")
    print("EXPERIMENTAL VALIDATION")
    print("===================================================")
    print(f"Experimental traverse tows used: {EXPERIMENTAL_TOWS}")

    # -------------------------------------------------------------------------
    # Experimental last 15 tows
    # -------------------------------------------------------------------------
    (
        exp_gap_overlap_df,
        exp_gap_df,
        exp_overlap_df,
        exp_gap_percent,
        exp_overlap_percent
    ) = traverse_tow_gaps_and_overlaps_selected(
        tows=EXPERIMENTAL_TOWS,
        plot=plot_experimental,
        tow_spacing=tow_spacing_mm,
        print_statement=True
    )

    # -------------------------------------------------------------------------
    # RW trained on first 15 tows
    # -------------------------------------------------------------------------
    rw_rows = []

    for sim in tqdm(range(n_RW_simulations), desc="Running RW simulations"):

        (
            rw_gap_overlap_df,
            rw_gap_df,
            rw_overlap_df,
            rw_gap_percent,
            rw_overlap_percent,
            rw_all_tows_data
        ) = generate_RW_multitow(
            num_tows=num_tows,
            tow_spacing_mm=tow_spacing_mm,
            tow_width_mm=tow_width_mm,
            tow_length_mm=tow_length_mm,
            proposal_type=proposal_type,
            print_statement=False,
            training_tows=RW_TRAINING_TOWS
        )

        rw_rows.append({
            "Simulation": sim + 1,
            "RW Gap (%)": rw_gap_percent,
            "RW Overlap (%)": rw_overlap_percent
        })

    rw_results_df = pd.DataFrame(rw_rows)

    rw_gap_mean = rw_results_df["RW Gap (%)"].mean()
    rw_gap_std = rw_results_df["RW Gap (%)"].std()

    rw_overlap_mean = rw_results_df["RW Overlap (%)"].mean()
    rw_overlap_std = rw_results_df["RW Overlap (%)"].std()

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    summary_df = pd.DataFrame({
        "Case": [
            "RW trained on first 15 tows",
            "Experimental last 15 tows"
        ],
        "Tows used": [
            str(RW_TRAINING_TOWS),
            str(EXPERIMENTAL_TOWS)
        ],
        "Gap mean (%)": [
            rw_gap_mean,
            exp_gap_percent
        ],
        "Gap std (%)": [
            rw_gap_std,
            0.0
        ],
        "Overlap mean (%)": [
            rw_overlap_mean,
            exp_overlap_percent
        ],
        "Overlap std (%)": [
            rw_overlap_std,
            0.0
        ]
    })

    print("\n===================================================")
    print("FINAL RESULTS")
    print("===================================================")

    print("\nRW trained on first 15 tows:")
    print(f"Gap     = {rw_gap_mean:.3f}% ± {rw_gap_std:.3f}%")
    print(f"Overlap = {rw_overlap_mean:.3f}% ± {rw_overlap_std:.3f}%")

    print("\nExperimental last 15 tows:")
    print(f"Gap     = {exp_gap_percent:.3f}%")
    print(f"Overlap = {exp_overlap_percent:.3f}%")

    # -------------------------------------------------------------------------
    # Save results
    # -------------------------------------------------------------------------
    if save_csv:
        os.makedirs("Cached Data", exist_ok=True)

        rw_csv_path = "Cached Data/RW_first15_training_defect_percentages.csv"
        summary_csv_path = "Cached Data/RW_first15_vs_Experimental_last15_summary.csv"

        rw_results_df.to_csv(rw_csv_path, index=False)
        summary_df.to_csv(summary_csv_path, index=False)

        print(f"\nSaved RW simulation results to: {rw_csv_path}")
        print(f"Saved summary to: {summary_csv_path}")

    return rw_results_df, summary_df


if __name__ == "__main__":

    rw_results_df, summary_df = calculate_RW_first15_and_exp_last15_defect_percentages(
        n_RW_simulations=100,
        tow_spacing_mm=tow_width_specified,
        tow_width_mm=tow_width_specified,
        tow_length_mm=1000,
        proposal_type="RWM",
        save_csv=True,
        plot_experimental=False
    )