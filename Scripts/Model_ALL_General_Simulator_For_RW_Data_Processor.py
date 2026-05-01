# External imports
from concurrent.futures import ProcessPoolExecutor
import matplotlib.pyplot as plt
from tqdm import tqdm
import seaborn as sns
import pandas as pd
import numpy as np
import time
import sys
import os

#Internal imports
from Model_ALL_RandomWalk import generate_random_walk, get_n_steps, get_proposal_distribution

############################################################################################################################################
"""Functions"""

baseline_file="Cached Data/Normal Distribution Variations/baseline_spacing_data.csv"

baseline_params = {
        "LT": (0.00, 0.06),
        "CAM": (0.00, 0.06),
        "LLSB": (0.00, 0.06),
        "LLSA": (0.00, 0.06)}

RW_PROPOSAL_STD = {
    "LT": get_proposal_distribution("LT"),
    "CAM": get_proposal_distribution("CAM"),
    "LLS_B": get_proposal_distribution("LLS_B"),
    "LLS_A": get_proposal_distribution("LLS_A")}

STEP_CACHE = {
    "LT": get_n_steps("LT") or 400,
    "CAM": get_n_steps("CAM") or 400,
    "LLS_B": get_n_steps("LLS_B") or 400,
    "LLS_A": get_n_steps("LLS_A") or 400}

def NORMAL_distribution(mu, sigma):
    from scipy.stats import norm
    return lambda x: norm.pdf(x, loc=mu, scale=sigma)

def compute_spacing_distribution(params, num_tows=31, tow_length_mm=1000):
    """
    Returns:
        spacing_data: all local spacing values (mm)
        gap_lengths: lengths of gap regions (mm)
        overlap_lengths: lengths of overlap regions (mm)
    """

    from scipy.stats import norm

    LT_steps = STEP_CACHE["LT"]
    CAM_steps = STEP_CACHE["CAM"]
    LLSB_steps = STEP_CACHE["LLS_B"]
    LLSA_steps = STEP_CACHE["LLS_A"]

    tow_offset = 0
    top_paths = []
    bottom_paths = []

    x = np.linspace(0, tow_length_mm, min(LT_steps, CAM_steps, LLSB_steps, LLSA_steps))
    dx = x[1] - x[0]

    for _ in range(num_tows):

        LT = generate_random_walk("LT", LT_steps, RW_PROPOSAL_STD["LT"],
                                 NORMAL_distribution(*params["LT"]), norm, params["LT"])

        CAM = generate_random_walk("CAM", CAM_steps, RW_PROPOSAL_STD["CAM"],
                                  NORMAL_distribution(*params["CAM"]), norm, params["CAM"])

        LLS_B = generate_random_walk("LLS_B", LLSB_steps, RW_PROPOSAL_STD["LLS_B"],
                                    NORMAL_distribution(*params["LLSB"]), norm, params["LLSB"])

        LLS_A = generate_random_walk("LLS_A", LLSA_steps, RW_PROPOSAL_STD["LLS_A"],
                                    NORMAL_distribution(*params["LLSA"]), norm, params["LLSA"])

        def interp(arr):
            return np.interp(
                np.linspace(0, len(arr)-1, len(x)),
                np.arange(len(arr)),
                arr
            )

        LT, CAM, LLS_B, LLS_A = map(interp, (LT, CAM, LLS_B, LLS_A))

        center = tow_offset + CAM + LT
        width = 6.35 + LLS_B

        top = center + 0.5 * width
        bottom = center - 0.5 * width

        top_paths.append(top)
        bottom_paths.append(bottom)

        tow_offset += 6.35

    # -----------------------------
    # COLLECT SPACING + SEGMENTS
    # -----------------------------
    spacing_data = []
    gap_lengths = []
    overlap_lengths = []

    for i in range(len(top_paths) - 1):

        diff = bottom_paths[i + 1] - top_paths[i]
        spacing_data.extend(diff)

        # --- zero-crossing tracking ---
        current_sign = np.sign(diff[0])
        segment_start_idx = 0

        for j in range(1, len(diff)):
            new_sign = np.sign(diff[j])

            # detect crossing (ignore exact zeros edge-case simply)
            if new_sign != current_sign and new_sign != 0:

                length = (j - segment_start_idx) * dx

                if current_sign > 0:
                    gap_lengths.append(length)
                elif current_sign < 0:
                    overlap_lengths.append(length)

                segment_start_idx = j
                current_sign = new_sign

        # handle final segment
        length = (len(diff) - segment_start_idx) * dx
        if current_sign > 0:
            gap_lengths.append(length)
        elif current_sign < 0:
            overlap_lengths.append(length)

    return (
        np.array(spacing_data),
        np.array(gap_lengths),
        np.array(overlap_lengths)
    )

def compute_baseline_spacing(runs=100, tows=31):
    global normal_mode
    normal_mode = True

    spacing_data = []
    gap_data = []
    overlap_data = []

    def print_progress(i, total, start_time):
        progress = (i + 1) / total
        bar_len = 30
        filled = int(bar_len * progress)

        bar = "█" * filled + "-" * (bar_len - filled)
        percent = progress * 100

        elapsed = time.time() - start_time
        remaining = (elapsed / progress - elapsed) if progress > 0 else 0

        sys.stdout.write(
            f"\rBaseline |{bar}| {percent:6.2f}% "
            f"Elapsed: {elapsed:5.1f}s "
            f"Remaining: {remaining:5.1f}s"
        )
        sys.stdout.flush()

    start_time = time.time()

    for i in range(runs):
        spacing, gaps, overlaps = compute_spacing_distribution(baseline_params, num_tows=tows)

        spacing_data.extend(spacing)
        gap_data.extend(gaps)
        overlap_data.extend(overlaps)

        print_progress(i, runs, start_time)

    print("\nBaseline ready.\n")

    baseline_df = pd.DataFrame({
        "value": np.concatenate([spacing_data, gap_data, overlap_data]),
        "metric": (["spacing"] * len(spacing_data) +
                   ["gap_length"] * len(gap_data) +
                   ["overlap_length"] * len(overlap_data)),
        "type": "baseline"
    })

    save_path = "Cached Data/Normal Distribution Variations/baseline_spacing_data.csv"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    baseline_df.to_csv(save_path, index=False)
    print(f"Saved baseline data to: {save_path}")

    return baseline_df

def run_single_spacing_worker(args):
    custom_params, tows = args
    return compute_spacing_distribution(custom_params, num_tows=tows)

def KDE_spacing_from_normals(
    custom_params=None,
    runs=100,
    tows=31,
    sensor_type=None,
    distribution_parameter=None,
    save_csv=True):

    global normal_mode
    normal_mode = True

    valid_sensors = ["LT", "CAM", "LLSB", "LLSA", "LT_CAM", "LLSB_LLSA", "ALL"]
    valid_params = ["mu", "sigma", "both"]

    if sensor_type not in valid_sensors:
        raise ValueError(f"sensor_type must be one of {valid_sensors}")
    if distribution_parameter not in valid_params:
        raise ValueError(f"distribution_parameter must be one of {valid_params}")

    baseline_df = pd.read_csv(baseline_file)

    if custom_params is None:
        raise ValueError("Provide custom_params.")

    args = [(custom_params, tows) for _ in range(runs)]

    with ProcessPoolExecutor() as executor:
        results = list(tqdm(
            executor.map(run_single_spacing_worker, args),
            total=runs,
            desc="Generating spacing data"
        ))

    # unpack results
    spacing_all = []
    gaps_all = []
    overlaps_all = []

    for spacing, gaps, overlaps in results:
        spacing_all.extend(spacing)
        gaps_all.extend(gaps)
        overlaps_all.extend(overlaps)

    customdata_df = pd.DataFrame({
        "value": np.concatenate([spacing_all, gaps_all, overlaps_all]),
        "metric": (["spacing"] * len(spacing_all) +
                   ["gap_length"] * len(gaps_all) +
                   ["overlap_length"] * len(overlaps_all))
    })

    if save_csv:
        filename = f"Cached Data/Normal Distribution Variations/{sensor_type}_shifted_{distribution_parameter}_spacing_data.csv"
        if os.path.exists(filename):
            print(f"Overwriting existing file: {filename}")
        customdata_df.to_csv(filename, index=False, mode="w")
        print(f"Saved: {filename}")

    return customdata_df, baseline_df

def run_spacing_multiple_simulations(runs=100, tows=31):

    experiments = [
        ("LT", "sigma", ["LT"]),
        ("LT", "mu", ["LT"]),
        ("CAM", "sigma", ["CAM"]),
        ("CAM", "mu", ["CAM"]),
        ("LT_CAM", "sigma", ["LT", "CAM"]),
        ("LT_CAM", "mu", ["LT", "CAM"]),
        ("LLSB", "sigma", ["LLSB"]),
        ("LLSB", "mu", ["LLSB"]),
        ("LLSA", "sigma", ["LLSA"]),
        ("LLSA", "mu", ["LLSA"]),
        ("LLSB_LLSA", "sigma", ["LLSB", "LLSA"]),
        ("LLSB_LLSA", "mu", ["LLSB", "LLSA"]),
        ("ALL", "both", ["LT", "CAM", "LLSB", "LLSA"])]
    

def generate_LT_CAM_opposite_spacing_data(runs=100, tows=31):
    """
    Generates the two new LT+CAM opposite-direction cases and saves them to CSV.

    Opposite convention:
    - sigma opposite: LT sigma increased, CAM sigma decreased
    - mu opposite:    LT mean increased, CAM mean decreased
    """

    save_dir = "Cached Data/Normal Distribution Variations"
    os.makedirs(save_dir, exist_ok=True)

    # --------------------------------------------------
    # LT+CAM sigma opposite
    # Baseline: LT sigma = 0.06, CAM sigma = 0.06
    # Opposite: LT sigma = 0.12, CAM sigma = 0.00
    # --------------------------------------------------
    sigma_opposite_params = baseline_params.copy()
    sigma_opposite_params["LT"] = (0.00, 0.1)
    sigma_opposite_params["CAM"] = (0.00, 0.03)

    df_sigma_opposite, _ = KDE_spacing_from_normals(
        custom_params=sigma_opposite_params,
        runs=runs,
        tows=tows,
        sensor_type="LT_CAM",
        distribution_parameter="sigma",
        save_csv=False
    )

    sigma_path = os.path.join(
        save_dir,
        "LT_CAM_opposite_shifted_sigma_spacing_data.csv"
    )

    df_sigma_opposite.to_csv(sigma_path, index=False)
    print(f"Saved: {sigma_path}")

    # --------------------------------------------------
    # LT+CAM mu opposite
    # Baseline: LT mu = 0.00, CAM mu = 0.00
    # Opposite: LT mu = +0.08, CAM mu = -0.08
    # --------------------------------------------------
    mu_opposite_params = baseline_params.copy()
    mu_opposite_params["LT"] = (0.08, 0.06)
    mu_opposite_params["CAM"] = (-0.08, 0.06)

    df_mu_opposite, _ = KDE_spacing_from_normals(
        custom_params=mu_opposite_params,
        runs=runs,
        tows=tows,
        sensor_type="LT_CAM",
        distribution_parameter="mu",
        save_csv=False
    )

    mu_path = os.path.join(
        save_dir,
        "LT_CAM_opposite_shifted_mu_spacing_data.csv"
    )

    df_mu_opposite.to_csv(mu_path, index=False)
    print(f"Saved: {mu_path}")

def plot_spacing_distribution(custom_file, title="", bins=100):

    df_custom = pd.read_csv(custom_file)
    df_baseline = pd.read_csv(baseline_file)

    # extract safely
    custom_data = df_custom[df_custom["metric"] == "spacing"]["value"].values \
        if "metric" in df_custom.columns else df_custom["spacing"].values

    baseline_data = df_baseline[df_baseline["metric"] == "spacing"]["value"].values \
        if "metric" in df_baseline.columns else df_baseline["spacing"].values

    # 🔥 shared bin range
    all_data = np.concatenate([baseline_data, custom_data])
    bin_edges = np.linspace(np.min(all_data), np.max(all_data), bins + 1)

    plt.figure(figsize=(10, 6))

    plt.hist(baseline_data, bins=bin_edges, density=True, alpha=0.5, label="Baseline")
    plt.hist(custom_data, bins=bin_edges, density=True, alpha=0.5, label="Custom")

    try:
        sns.kdeplot(baseline_data, linewidth=2)
        sns.kdeplot(custom_data, linewidth=2)
    except:
        pass

    plt.axvline(0, color="black", linestyle=":", label="Ideal (0 gap)")
    plt.xlabel("Spacing (mm)")
    plt.ylabel("Density")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_gap_length_distribution(custom_file, title="", bins=30):

    df_custom = pd.read_csv(custom_file)
    df_baseline = pd.read_csv(baseline_file)

    if "metric" in df_custom.columns:
        custom_gaps = df_custom[df_custom["metric"] == "gap_length"]["value"].values
        baseline_gaps = df_baseline[df_baseline["metric"] == "gap_length"]["value"].values
    else:
        custom_gaps = df_custom["gap_length"].values
        baseline_gaps = df_baseline["gap_length"].values

    # 🔥 FIX: shared bin edges over fixed range
    bin_edges = np.linspace(0, 150, bins + 1)

    plt.figure(figsize=(10, 6))

    plt.hist(baseline_gaps, bins=bin_edges, alpha=0.5, label="Baseline gaps")
    plt.hist(custom_gaps, bins=bin_edges, alpha=0.5, label="Custom gaps")

    plt.xlabel("Gap length (mm)")
    plt.ylabel("Frequency")
    plt.xlim(0, 150)
    plt.ylim(0, 25000)
    plt.title(title + " - Gap Distribution")
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_all_spacing_variations(bins=100):

    figsize = (18, 18)
    baseline_df = pd.read_csv(baseline_file)

    # extract baseline spacing safely
    if "metric" in baseline_df.columns:
        baseline = baseline_df[baseline_df["metric"] == "spacing"]["value"].values
    else:
        baseline = baseline_df["spacing"].values

    cases = [
        ("LT_std", "LT_shifted_sigma_spacing_data.csv"),
        ("LT_mu", "LT_shifted_mu_spacing_data.csv"),
        ("CAM_std", "CAM_shifted_sigma_spacing_data.csv"),
        ("CAM_mu", "CAM_shifted_mu_spacing_data.csv"),
        ("LT+CAM_std", "LT_CAM_shifted_sigma_spacing_data.csv"),
        ("LT+CAM_mu", "LT_CAM_shifted_mu_spacing_data.csv"),
        ("LLSB_std", "LLSB_shifted_sigma_spacing_data.csv"),
        ("LLSB_mu", "LLSB_shifted_mu_spacing_data.csv"),
        ("LLSA_std", "LLSA_shifted_sigma_spacing_data.csv"),
        ("LLSA_mu", "LLSA_shifted_mu_spacing_data.csv"),
        ("LLSB+LLSA_std", "LLSB_LLSA_shifted_sigma_spacing_data.csv"),
        ("LLSB+LLSA_mu", "LLSB_LLSA_shifted_mu_spacing_data.csv"),
        ("ALL_both", "ALL_shifted_both_spacing_data.csv"),
    ]

    fig, axes = plt.subplots(4, 4, figsize=figsize)
    axes = axes.flatten()

def plot_LT_CAM_spacing_variations_ordered(bins=100):
    """
    Plot only LT and CAM spacing variations in the requested order.

    Layout:
    - Row 1: Baseline across both columns
    - Row 2: LT_sigma | LT_mu
    - Row 3: CAM_sigma | CAM_mu
    - Row 4: LT_CAM_sigma | LT_CAM_mu
    - Row 5: LT_CAM_sigma_opposite | LT_CAM_mu_opposite

    Legend in each plot contains:
    - plot name
    - mean of custom
    - SD of custom

    No plot titles.
    No KDE lines.
    """

    from matplotlib.lines import Line2D

    def extract_spacing(df):
        if "metric" in df.columns:
            return df[df["metric"] == "spacing"]["value"].values
        return df["spacing"].values

    save_dir = "Cached Data/Normal Distribution Variations"

    baseline_df = pd.read_csv(baseline_file)
    baseline = extract_spacing(baseline_df)

    cases = [
        ("LT_sigma", os.path.join(save_dir, "LT_shifted_sigma_spacing_data.csv")),
        ("LT_mu", os.path.join(save_dir, "LT_shifted_mu_spacing_data.csv")),

        ("CAM_sigma", os.path.join(save_dir, "CAM_shifted_sigma_spacing_data.csv")),
        ("CAM_mu", os.path.join(save_dir, "CAM_shifted_mu_spacing_data.csv")),

        ("LT_CAM_sigma", os.path.join(save_dir, "LT_CAM_shifted_sigma_spacing_data.csv")),
        ("LT_CAM_mu", os.path.join(save_dir, "LT_CAM_shifted_mu_spacing_data.csv")),

        ("LT_CAM_sigma_opposite", os.path.join(save_dir, "LT_CAM_opposite_shifted_sigma_spacing_data.csv")),
        ("LT_CAM_mu_opposite", os.path.join(save_dir, "LT_CAM_opposite_shifted_mu_spacing_data.csv")),
    ]

    loaded_cases = []
    all_data = [baseline]

    for name, file in cases:
        if not os.path.exists(file):
            raise FileNotFoundError(
                f"Missing file: {file}\n"
                f"Generate it first before plotting."
            )

        df = pd.read_csv(file)
        data = extract_spacing(df)

        loaded_cases.append((name, data))
        all_data.append(data)

    all_data = np.concatenate(all_data)

    x_min = np.min(all_data)
    x_max = np.max(all_data)
    bin_edges = np.linspace(x_min, x_max, bins + 1)

    fig = plt.figure(figsize=(18, 18))
    gs = fig.add_gridspec(5, 2)

    axes = []

    # Baseline plot spans both columns
    axes.append(fig.add_subplot(gs[0, :]))

    # Remaining 8 plots
    for r in range(1, 5):
        for c in range(2):
            axes.append(fig.add_subplot(gs[r, c]))

    baseline_mean = np.mean(baseline)
    baseline_std = np.std(baseline, ddof=1)

    def make_text_legend(ax, line1, line2, line3):
        handles = [
            Line2D([], [], linestyle="none", label=line1),
            Line2D([], [], linestyle="none", label=line2),
            Line2D([], [], linestyle="none", label=line3),
        ]
        ax.legend(handles=handles, loc="best", handlelength=0, handletextpad=0)

    def plot(ax, data, name, show_custom=True):
        # Histograms only
        ax.hist(baseline, bins=bin_edges, density=True, alpha=0.4)

        if show_custom:
            ax.hist(data, bins=bin_edges, density=True, alpha=0.4)

        # Keep this if you still want the ideal spacing reference
        ax.axvline(0, color="black", linestyle=":")

        ax.set_xlim(x_min, x_max)

        # No title
        # ax.set_title(name)

        if show_custom:
            custom_mean = np.mean(data)
            custom_std = np.std(data, ddof=1)

            make_text_legend(
                ax,
                f"{name}",
                f"Mean = {custom_mean:.4f}",
                f"SD = {custom_std:.4f}"
            )
        else:
            make_text_legend(
                ax,
                "Baseline",
                f"Mean = {baseline_mean:.4f}",
                f"SD = {baseline_std:.4f}"
            )

    # Baseline plot
    plot(axes[0], baseline, "Baseline", show_custom=False)

    # Remaining plots
    for ax, (name, data) in zip(axes[1:], loaded_cases):
        plot(ax, data, name, show_custom=True)

    plt.tight_layout()

    plt.savefig(
        "LT_CAM_spacing_variations_ordered.png",
        dpi=300,
        bbox_inches="tight"
    )

plt.show()

def plot_all_gap_length_variations(bins=30):

    figsize = (18, 18)

    cases = [
        ("LT_std", "LT_shifted_sigma_spacing_data.csv"),
        ("LT_mu", "LT_shifted_mu_spacing_data.csv"),
        ("CAM_std", "CAM_shifted_sigma_spacing_data.csv"),
        ("CAM_mu", "CAM_shifted_mu_spacing_data.csv"),
        ("LT+CAM_std", "LT_CAM_shifted_sigma_spacing_data.csv"),
        ("LT+CAM_mu", "LT_CAM_shifted_mu_spacing_data.csv"),
        ("LLSB_std", "LLSB_shifted_sigma_spacing_data.csv"),
        ("LLSB_mu", "LLSB_shifted_mu_spacing_data.csv"),
        ("LLSA_std", "LLSA_shifted_sigma_spacing_data.csv"),
        ("LLSA_mu", "LLSA_shifted_mu_spacing_data.csv"),
        ("LLSB+LLSA_std", "LLSB_LLSA_shifted_sigma_spacing_data.csv"),
        ("LLSB+LLSA_mu", "LLSB_LLSA_shifted_mu_spacing_data.csv"),
        ("ALL_both", "ALL_shifted_both_spacing_data.csv"),
    ]

    fig, axes = plt.subplots(4, 4, figsize=figsize)
    axes = axes.flatten()

    baseline_df = pd.read_csv(baseline_file)

    if "metric" in baseline_df.columns:
        baseline_gaps = baseline_df[baseline_df["metric"] == "gap_length"]["value"].values
    else:
        baseline_gaps = baseline_df["gap_length"].values

    # 🔥 FIX: shared bins for ALL plots
    bin_edges = np.linspace(0, 150, bins + 1)

    def plot(ax, data, title):

        ax.hist(baseline_gaps, bins=bin_edges, alpha=0.4, label="Baseline gaps")
        ax.hist(data, bins=bin_edges, alpha=0.4, label="Custom gaps")

        ax.set_xlim(0, 150)
        ax.set_ylim(0, 25000)

        ax.set_title(title)
        ax.set_ylabel("Frequency")
        ax.legend()

    plot(axes[0], baseline_gaps, "Baseline")

    for i, (title, file) in enumerate(cases, start=1):

        df = pd.read_csv(f"Cached Data/Normal Distribution Variations/{file}")

        if "metric" in df.columns:
            data = df[df["metric"] == "gap_length"]["value"].values
        else:
            data = df["gap_length"].values

        plot(axes[i], data, title)

    for j in range(len(cases) + 1, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.show()

#TODO: do this plot for the four scenarios 
#TODO: Find where to change the means and SD of the increased and decreased metrics. change increased mean to 0.1 and mu=0.

############################################################################################################################################
"""Generate Data"""
############################################################################################################################################
"""Generate Data / Graphs"""

if __name__ == "__main__":

    from multiprocessing import freeze_support, set_start_method

    freeze_support()

    try:
        set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    # --------------------------------------------------
    # Run this ONCE if the opposite LT+CAM files do not exist yet
    # --------------------------------------------------
    #generate_LT_CAM_opposite_spacing_data(runs=100, tows=31)

    # --------------------------------------------------
    # Plot LT/CAM spacing variations
    # --------------------------------------------------
    plot_LT_CAM_spacing_variations_ordered(bins=100)

    # compute_baseline_spacing()
    # KDE_spacing_from_normals(params_test, runs=100, sensor_type="ALL", distribution_parameter="both") # GENERATES ONE DATA SET BASED ON PARAMS_TEST
    # run_spacing_multiple_simulations() # GENERATES A LOT OF DATA. Takes 25 min!

"""Generate Graphs"""
#plot_spacing_distribution("Cached Data/Normal Distribution Variations/LT_shifted_mu_spacing_data.csv")
#plot_gap_length_distribution("Cached Data/Normal Distribution Variations/LT_shifted_mu_spacing_data.csv")
#plot_all_spacing_variations(bins=100) # Takes a minute
#plot_all_gap_length_variations(bins=30) # Takes a minute