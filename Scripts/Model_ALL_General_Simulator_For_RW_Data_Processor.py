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
        "LT": (0.00, 0.1),
        "CAM": (0.00, 0.1),
        "LLSB": (0.00, 0.1),
        "LLSA": (0.00, 0.1)}

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

    def modify(params, sensors=None, change=None):
        new_params = params.copy()

        for s in sensors:
            mu, sigma = new_params[s]

            if change == "mean":
                mu = 0.5
            elif change == "std":
                sigma = 0.15
            elif change == "both":
                mu = 0.5
                sigma = 0.15

            new_params[s] = (mu, sigma)

        return new_params

    start_time = time.time()

    for i, (name, param_type, sensors) in enumerate(experiments):

        print(f"\n===== Running {i+1}/{len(experiments)}: {name} | {param_type} =====")

        change = {
            "mu": "mean",
            "sigma": "std",
            "both": "both"
        }.get(param_type)

        custom_params = modify(baseline_params, sensors=sensors, change=change)

        KDE_spacing_from_normals(
            custom_params=custom_params,
            runs=runs,
            tows=tows,
            sensor_type=name,
            distribution_parameter=param_type,
            save_csv=True)

    elapsed = time.time() - start_time
    print(f"\nAll simulations complete in {elapsed:.1f}s\n")   

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
    sigma_opposite_params["LT"] = (0.00, 0.05)
    sigma_opposite_params["CAM"] = (0.00, 0.15)

    df_sigma_opposite, _ = KDE_spacing_from_normals(
        custom_params=sigma_opposite_params,
        runs=runs,
        tows=tows,
        sensor_type="LT_CAM",
        distribution_parameter="sigma",
        save_csv=False)

    sigma_path = os.path.join(save_dir,"LT_CAM_opposite_shifted_sigma_spacing_data.csv")

    df_sigma_opposite.to_csv(sigma_path, index=False)
    print(f"Saved: {sigma_path}")

    # --------------------------------------------------
    # LT+CAM mu opposite
    # Baseline: LT mu = 0.00, CAM mu = 0.00
    # Opposite: LT mu = +0.08, CAM mu = -0.08
    # --------------------------------------------------
    mu_opposite_params = baseline_params.copy()
    mu_opposite_params["LT"] = (-0.5, 0.1)
    mu_opposite_params["CAM"] = (0.5, 0.1)

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

def plot_LT_CAM_spacing_variations_ordered(bins=100):
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

    # Outer grid (only vertical control)
    outer = fig.add_gridspec(5, 1, hspace=0.3)

    axes = []

    # Top row (centered, same width as others)
    top = outer[0].subgridspec(1, 3, width_ratios=[1, 2, 1])
    axes.append(fig.add_subplot(top[0, 1]))

    # Remaining rows (normal 2-column layout)
    for r in range(1, 5):
        inner = outer[r].subgridspec(1, 2, wspace=0.25)
        axes.append(fig.add_subplot(inner[0, 0]))
        axes.append(fig.add_subplot(inner[0, 1]))

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
        ax.hist(baseline, bins=bin_edges, density=True, alpha=0.4)

        if show_custom:
            ax.hist(data, bins=bin_edges, density=True, alpha=0.4)

        ax.axvline(0, color="black", linestyle=":")
        ax.set_xlim(-1.2, 1.2)

        if show_custom:
            make_text_legend(
                ax,
                f"{name}",
                f"Mean = {np.mean(data):.4f}",
                f"SD = {np.std(data, ddof=1):.4f}"
            )
        else:
            make_text_legend(
                ax,
                "Baseline",
                f"Mean = {baseline_mean:.4f}",
                f"SD = {baseline_std:.4f}"
            )

    # Baseline
    plot(axes[0], baseline, "Baseline", show_custom=False)

    # Remaining plots
    for ax, (name, data) in zip(axes[1:], loaded_cases):
        plot(ax, data, name, show_custom=True)

    plt.savefig(
        "LT_CAM_spacing_variations_ordered.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

def generate_LLSB_LLSA_opposite_spacing_data(runs=100, tows=31):
    """
    Generates the two new LLSA+LLSB opposite-direction cases and saves them to CSV.

    Opposite convention:
    - sigma opposite: LLSA sigma increased, LLSB sigma decreased
    - mu opposite:    LLSA mean increased, LLSB mean decreased
    """

    save_dir = "Cached Data/Normal Distribution Variations"
    os.makedirs(save_dir, exist_ok=True)

    # --------------------------------------------------
    # LLSA+LLSB sigma opposite
    # Baseline: LLSA sigma = 0.06, LLSB sigma = 0.06
    # Opposite: LLSA sigma = 0.12, LLSB sigma = 0.00
    # --------------------------------------------------
    sigma_opposite_params = baseline_params.copy()
    sigma_opposite_params["LLSA"] = (0.00, 0.05)
    sigma_opposite_params["LLSB"] = (0.00, 0.15)

    df_sigma_opposite, _ = KDE_spacing_from_normals(
        custom_params=sigma_opposite_params,
        runs=runs,
        tows=tows,
        sensor_type="LLSB_LLSA",
        distribution_parameter="sigma",
        save_csv=False
    )

    sigma_path = os.path.join(
        save_dir,
        "LLSB_LLSA_opposite_shifted_sigma_spacing_data.csv"
    )

    df_sigma_opposite.to_csv(sigma_path, index=False)
    print(f"Saved: {sigma_path}")

    # --------------------------------------------------
    # LLSA+LLSB mu opposite
    # Baseline: LLSA mu = 0.00, LLSB mu = 0.00
    # Opposite: LLSA mu = +0.08, LLSB mu = -0.08
    # --------------------------------------------------
    mu_opposite_params = baseline_params.copy()
    mu_opposite_params["LLSA"] = (-0.5, 0.1)
    mu_opposite_params["LLSB"] = (0.5, 0.1)

    df_mu_opposite, _ = KDE_spacing_from_normals(
        custom_params=mu_opposite_params,
        runs=runs,
        tows=tows,
        sensor_type="LLSB_LLSA",
        distribution_parameter="mu",
        save_csv=False
    )

    mu_path = os.path.join(
        save_dir,
        "LLSB_LLSA_opposite_shifted_mu_spacing_data.csv"
    )

    df_mu_opposite.to_csv(mu_path, index=False)
    print(f"Saved: {mu_path}")

def plot_LLSB_LLSA_spacing_variations_ordered(bins=100):
    from matplotlib.lines import Line2D

    def extract_spacing(df):
        if "metric" in df.columns:
            return df[df["metric"] == "spacing"]["value"].values
        return df["spacing"].values

    save_dir = "Cached Data/Normal Distribution Variations"

    baseline_df = pd.read_csv(baseline_file)
    baseline = extract_spacing(baseline_df)

    cases = [
        ("LLSA_sigma", os.path.join(save_dir, "LLSA_shifted_sigma_spacing_data.csv")),
        ("LLSA_mu", os.path.join(save_dir, "LLSA_shifted_mu_spacing_data.csv")),

        ("LLSB_sigma", os.path.join(save_dir, "LLSB_shifted_sigma_spacing_data.csv")),
        ("LLSB_mu", os.path.join(save_dir, "LLSB_shifted_mu_spacing_data.csv")),

        ("LLSB_LLSA_sigma", os.path.join(save_dir, "LLSB_LLSA_shifted_sigma_spacing_data.csv")),
        ("LLSB_LLSA_mu", os.path.join(save_dir, "LLSB_LLSA_shifted_mu_spacing_data.csv")),

        ("LLSB_LLSA_sigma_opposite", os.path.join(save_dir, "LLSB_LLSA_opposite_shifted_sigma_spacing_data.csv")),
        ("LLSB_LLSA_mu_opposite", os.path.join(save_dir, "LLSB_LLSA_opposite_shifted_mu_spacing_data.csv")),
    ]

    loaded_cases = []
    all_data = [baseline]

    for name, file in cases:
        if not os.path.exists(file):
            raise FileNotFoundError(f"Missing file: {file}")

        df = pd.read_csv(file)
        data = extract_spacing(df)

        loaded_cases.append((name, data))
        all_data.append(data)

    all_data = np.concatenate(all_data)

    x_min = np.min(all_data)
    x_max = np.max(all_data)
    bin_edges = np.linspace(x_min, x_max, bins + 1)

    fig = plt.figure(figsize=(18, 18))

    # Outer grid: ONLY controls vertical stacking
    outer = fig.add_gridspec(5, 1, hspace=0.3)

    axes = []

    # ---- Top row (centered but same width as others) ----
    top = outer[0].subgridspec(1, 3, width_ratios=[1, 2, 1])
    axes.append(fig.add_subplot(top[0, 1]))

    # ---- Remaining rows (normal 2-column layout) ----
    for r in range(1, 5):
        inner = outer[r].subgridspec(1, 2, wspace=0.25)
        axes.append(fig.add_subplot(inner[0, 0]))
        axes.append(fig.add_subplot(inner[0, 1]))

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
        ax.hist(baseline, bins=bin_edges, density=True, alpha=0.4)

        if show_custom:
            ax.hist(data, bins=bin_edges, density=True, alpha=0.4)

        ax.axvline(0, color="black", linestyle=":")
        ax.set_xlim(-1.2, 1.2)

        if show_custom:
            make_text_legend(
                ax,
                f"{name}",
                f"Mean = {np.mean(data):.4f}",
                f"SD = {np.std(data, ddof=1):.4f}"
            )
        else:
            make_text_legend(
                ax,
                "Baseline",
                f"Mean = {baseline_mean:.4f}",
                f"SD = {baseline_std:.4f}"
            )

    plot(axes[0], baseline, "Baseline", show_custom=False)

    for ax, (name, data) in zip(axes[1:], loaded_cases):
        plot(ax, data, name, show_custom=True)

    plt.savefig(
        "LLSB_LLSA_spacing_variations_ordered.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

def plot_baseline_vs_all_spacing_variations_ordered(bins=100):
    from matplotlib.lines import Line2D

    def extract_spacing(df):
        if "metric" in df.columns:
            return df[df["metric"] == "spacing"]["value"].values
        return df["spacing"].values

    save_dir = "Cached Data/Normal Distribution Variations"

    # ---- Load baseline ----
    baseline_df = pd.read_csv(baseline_file)
    baseline = extract_spacing(baseline_df)

    # ---- Load ALL combined data ----
    all_file = os.path.join(save_dir, "ALL_shifted_both_spacing_data.csv")
    if not os.path.exists(all_file):
        raise FileNotFoundError(f"Missing file: {all_file}")

    all_df = pd.read_csv(all_file)
    all_data = extract_spacing(all_df)

    # ---- Shared bins ----
    combined = np.concatenate([baseline, all_data])
    x_min = np.min(combined)
    x_max = np.max(combined)
    bin_edges = np.linspace(x_min, x_max, bins + 1)

    # ---- Stats ----
    baseline_mean = np.mean(baseline)
    baseline_std = np.std(baseline, ddof=1)

    all_mean = np.mean(all_data)
    all_std = np.std(all_data, ddof=1)

    # ---- Figure with 2 rows ----
    fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    def make_text_legend(ax, lines):
        handles = [Line2D([], [], linestyle="none", label=line) for line in lines]
        ax.legend(handles=handles, loc="best", handlelength=0, handletextpad=0)

    # ---- Top: Baseline only ----
    ax = axes[0]
    ax.hist(baseline, bins=bin_edges, density=True, alpha=0.5)
    ax.axvline(0, color="black", linestyle=":")
    ax.set_xlim(-1.2, 1.2)

    make_text_legend(
        ax,
        [
            "Baseline",
            f"Mean = {baseline_mean:.4f}",
            f"SD = {baseline_std:.4f}",
        ]
    )

    # ---- Bottom: Baseline + ALL overlay ----
    ax = axes[1]
    ax.hist(baseline, bins=bin_edges, density=True, alpha=0.4)
    ax.hist(all_data, bins=bin_edges, density=True, alpha=0.4)

    ax.axvline(0, color="black", linestyle=":")
    ax.set_xlim(-1.2, 1.2)

    make_text_legend(
        ax,
        [
            "Baseline vs ALL",
            f"Baseline μ={baseline_mean:.4f}, σ={baseline_std:.4f}",
            f"ALL μ={all_mean:.4f}, σ={all_std:.4f}",
        ]
    )

    plt.tight_layout()

    plt.savefig(
        "baseline_vs_all_overlay.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

def plot_LT_CAM_gap_length_variations_ordered(bins=100):
    from matplotlib.lines import Line2D

    def extract_gap(df):
        if "metric" in df.columns:
            return df[df["metric"] == "gap_length"]["value"].values
        return df["gap_length"].values

    save_dir = "Cached Data/Normal Distribution Variations"

    baseline_df = pd.read_csv(baseline_file)
    baseline = extract_gap(baseline_df)

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

    loaded_cases, all_data = [], [baseline]

    for name, file in cases:
        df = pd.read_csv(file)
        data = extract_gap(df)
        loaded_cases.append((name, data))
        all_data.append(data)

    all_data = np.concatenate(all_data)

    x_min, x_max = np.min(all_data), np.max(all_data)
    bin_edges = np.linspace(x_min, x_max, bins + 1)

    fig = plt.figure(figsize=(18, 18))
    outer = fig.add_gridspec(5, 1, hspace=0.3)

    axes = []

    top = outer[0].subgridspec(1, 3, width_ratios=[1, 2, 1])
    axes.append(fig.add_subplot(top[0, 1]))

    for r in range(1, 5):
        inner = outer[r].subgridspec(1, 2, wspace=0.25)
        axes.append(fig.add_subplot(inner[0, 0]))
        axes.append(fig.add_subplot(inner[0, 1]))

    baseline_mean = np.mean(baseline)
    baseline_std = np.std(baseline, ddof=1)

    def legend(ax, lines):
        handles = [Line2D([], [], linestyle="none", label=l) for l in lines]
        ax.legend(handles=handles, loc="best", handlelength=0)

    def plot(ax, data, name, show_custom=True):
        ax.hist(baseline, bins=bin_edges, density=False, alpha=0.4)

        if show_custom:
            ax.hist(data, bins=bin_edges, density=False, alpha=0.4)

        ax.set_xlim(0, 80)

        if show_custom:
            legend(ax, [name,
                        f"Mean = {np.mean(data):.4f}",
                        f"SD = {np.std(data, ddof=1):.4f}"])
        else:
            legend(ax, ["Baseline",
                        f"Mean = {baseline_mean:.4f}",
                        f"SD = {baseline_std:.4f}"])

    plot(axes[0], baseline, "Baseline", False)

    for ax, (name, data) in zip(axes[1:], loaded_cases):
        plot(ax, data, name)

    plt.savefig("LT_CAM_gap_length_variations.png", dpi=300, bbox_inches="tight")
    plt.show()

def plot_LLSB_LLSA_gap_length_variations_ordered(bins=100):
    from matplotlib.lines import Line2D

    def extract_gap(df):
        if "metric" in df.columns:
            return df[df["metric"] == "gap_length"]["value"].values
        return df["gap_length"].values

    save_dir = "Cached Data/Normal Distribution Variations"

    baseline = extract_gap(pd.read_csv(baseline_file))

    cases = [
        ("LLSA_sigma", os.path.join(save_dir, "LLSA_shifted_sigma_spacing_data.csv")),
        ("LLSA_mu", os.path.join(save_dir, "LLSA_shifted_mu_spacing_data.csv")),

        ("LLSB_sigma", os.path.join(save_dir, "LLSB_shifted_sigma_spacing_data.csv")),
        ("LLSB_mu", os.path.join(save_dir, "LLSB_shifted_mu_spacing_data.csv")),

        ("LLSB_LLSA_sigma", os.path.join(save_dir, "LLSB_LLSA_shifted_sigma_spacing_data.csv")),
        ("LLSB_LLSA_mu", os.path.join(save_dir, "LLSB_LLSA_shifted_mu_spacing_data.csv")),

        ("LLSB_LLSA_sigma_opposite", os.path.join(save_dir, "LLSB_LLSA_opposite_shifted_sigma_spacing_data.csv")),
        ("LLSB_LLSA_mu_opposite", os.path.join(save_dir, "LLSB_LLSA_opposite_shifted_mu_spacing_data.csv")),
    ]

    loaded_cases, all_data = [], [baseline]

    for name, file in cases:
        data = extract_gap(pd.read_csv(file))
        loaded_cases.append((name, data))
        all_data.append(data)

    all_data = np.concatenate(all_data)
    bin_edges = np.linspace(np.min(all_data), np.max(all_data), bins + 1)

    fig = plt.figure(figsize=(18, 18))
    outer = fig.add_gridspec(5, 1, hspace=0.3)

    axes = []

    top = outer[0].subgridspec(1, 3, width_ratios=[1, 2, 1])
    axes.append(fig.add_subplot(top[0, 1]))

    for r in range(1, 5):
        inner = outer[r].subgridspec(1, 2, wspace=0.25)
        axes.append(fig.add_subplot(inner[0, 0]))
        axes.append(fig.add_subplot(inner[0, 1]))

    def legend(ax, lines):
        handles = [Line2D([], [], linestyle="none", label=l) for l in lines]
        ax.legend(handles=handles)

    def plot(ax, data, name, show_custom=True):
        ax.hist(baseline, bins=bin_edges, density=False, alpha=0.4)

        if show_custom:
            ax.hist(data, bins=bin_edges, density=False, alpha=0.4)

        ax.set_xlim(0, 80)

        if show_custom:
            legend(ax, [name,
                        f"Mean = {np.mean(data):.4f}",
                        f"SD = {np.std(data, ddof=1):.4f}"])
        else:
            legend(ax, ["Baseline",
                        f"Mean = {np.mean(baseline):.4f}",
                        f"SD = {np.std(baseline, ddof=1):.4f}"])

    plot(axes[0], baseline, "Baseline", False)

    for ax, (name, data) in zip(axes[1:], loaded_cases):
        plot(ax, data, name)

    plt.savefig("LLSB_LLSA_gap_length_variations.png", dpi=300, bbox_inches="tight")
    plt.show()

def plot_baseline_vs_all_gap_length_variations_ordered(bins=100):
    from matplotlib.lines import Line2D

    def extract_gap(df):
        if "metric" in df.columns:
            return df[df["metric"] == "gap_length"]["value"].values
        return df["gap_length"].values

    save_dir = "Cached Data/Normal Distribution Variations"

    baseline = extract_gap(pd.read_csv(baseline_file))
    all_data = extract_gap(pd.read_csv(
        os.path.join(save_dir, "ALL_shifted_both_spacing_data.csv")
    ))

    combined = np.concatenate([baseline, all_data])
    bin_edges = np.linspace(np.min(combined), np.max(combined), bins + 1)

    fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    def legend(ax, lines):
        handles = [Line2D([], [], linestyle="none", label=l) for l in lines]
        ax.legend(handles=handles)

    # Baseline
    axes[0].hist(baseline, bins=bin_edges, density=False, alpha=0.5)
    axes[0].set_xlim(0, 80)

    legend(axes[0], [
        "Baseline",
        f"Mean = {np.mean(baseline):.4f}",
        f"SD = {np.std(baseline, ddof=1):.4f}"
    ])

    # Overlay
    axes[1].hist(baseline, bins=bin_edges, density=False, alpha=0.4)
    axes[1].hist(all_data, bins=bin_edges, density=False, alpha=0.4)
    axes[1].set_xlim(0, 80)

    legend(axes[1], [
        "Baseline vs ALL",
        f"Baseline μ={np.mean(baseline):.4f}, σ={np.std(baseline, ddof=1):.4f}",
        f"ALL μ={np.mean(all_data):.4f}, σ={np.std(all_data, ddof=1):.4f}",
    ])

    plt.tight_layout()
    plt.savefig("baseline_vs_all_gap_length.png", dpi=300, bbox_inches="tight")
    plt.show()

# Test functions NOT FOR PAPER!

def plot_LT_CAM_gap_length_variations_weighted_ordered(bins=50):
    from matplotlib.lines import Line2D

    def extract_gap(df):
        if "metric" in df.columns:
            return df[df["metric"] == "gap_length"]["value"].values
        return df["gap_length"].values

    save_dir = "Cached Data/Normal Distribution Variations"

    baseline_df = pd.read_csv(baseline_file)
    baseline = extract_gap(baseline_df)

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

    loaded_cases, all_data = [], [baseline]

    for name, file in cases:
        df = pd.read_csv(file)
        data = extract_gap(df)
        loaded_cases.append((name, data))
        all_data.append(data)

    all_data = np.concatenate(all_data)

    bin_edges = np.linspace(0, 1000, bins + 1)

    fig = plt.figure(figsize=(18, 18))
    outer = fig.add_gridspec(5, 1, hspace=0.3)

    axes = []

    top = outer[0].subgridspec(1, 3, width_ratios=[1, 2, 1])
    axes.append(fig.add_subplot(top[0, 1]))

    for r in range(1, 5):
        inner = outer[r].subgridspec(1, 2, wspace=0.25)
        axes.append(fig.add_subplot(inner[0, 0]))
        axes.append(fig.add_subplot(inner[0, 1]))

    baseline_mean = np.mean(baseline)
    baseline_std = np.std(baseline, ddof=1)

    def legend(ax, lines):
        handles = [Line2D([], [], linestyle="none", label=l) for l in lines]
        ax.legend(handles=handles, loc="best", handlelength=0)

    def pretty_name(name):
        return (
            name
            .replace("_", " + ")
            .replace("sigma", "SD Shift")
            .replace("mu", "Mean Shift")
        )

    def plot(ax, data, name, show_custom=True):

        ax.hist(
            baseline,
            bins=bin_edges,
            weights=baseline,
            density=False,
            alpha=0.4
        )

        if show_custom:
            ax.hist(
                data,
                bins=bin_edges,
                weights=data,
                density=False,
                alpha=0.4
            )

        ax.set_xlim(0, 1000)

        if show_custom:
            legend(ax, [
                pretty_name(name),
                f"Mean = {np.mean(data):.4f}",
                f"SD = {np.std(data, ddof=1):.4f}"
            ])
        else:
            legend(ax, [
                "Baseline",
                f"Mean = {baseline_mean:.4f}",
                f"SD = {baseline_std:.4f}"
            ])

        ax.set_ylabel("Total Gap Length")

    plot(axes[0], baseline, "Baseline", False)

    for ax, (name, data) in zip(axes[1:], loaded_cases):
        plot(ax, data, name)

    plt.savefig(
        "LT_CAM_gap_length_variations_weighted.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

def plot_LLSB_LLSA_gap_length_variations_weighted_ordered(bins=50):
    from matplotlib.lines import Line2D

    def extract_gap(df):
        if "metric" in df.columns:
            return df[df["metric"] == "gap_length"]["value"].values
        return df["gap_length"].values

    save_dir = "Cached Data/Normal Distribution Variations"

    baseline = extract_gap(pd.read_csv(baseline_file))

    cases = [
        ("LLSA_sigma", os.path.join(save_dir, "LLSA_shifted_sigma_spacing_data.csv")),
        ("LLSA_mu", os.path.join(save_dir, "LLSA_shifted_mu_spacing_data.csv")),

        ("LLSB_sigma", os.path.join(save_dir, "LLSB_shifted_sigma_spacing_data.csv")),
        ("LLSB_mu", os.path.join(save_dir, "LLSB_shifted_mu_spacing_data.csv")),

        ("LLSB_LLSA_sigma", os.path.join(save_dir, "LLSB_LLSA_shifted_sigma_spacing_data.csv")),
        ("LLSB_LLSA_mu", os.path.join(save_dir, "LLSB_LLSA_shifted_mu_spacing_data.csv")),

        ("LLSB_LLSA_sigma_opposite", os.path.join(save_dir, "LLSB_LLSA_opposite_shifted_sigma_spacing_data.csv")),
        ("LLSB_LLSA_mu_opposite", os.path.join(save_dir, "LLSB_LLSA_opposite_shifted_mu_spacing_data.csv")),
    ]

    loaded_cases, all_data = [], [baseline]

    for name, file in cases:
        data = extract_gap(pd.read_csv(file))
        loaded_cases.append((name, data))
        all_data.append(data)

    all_data = np.concatenate(all_data)

    bin_edges = np.linspace(0, 1000, bins + 1)

    fig = plt.figure(figsize=(18, 18))
    outer = fig.add_gridspec(5, 1, hspace=0.3)

    axes = []

    top = outer[0].subgridspec(1, 3, width_ratios=[1, 2, 1])
    axes.append(fig.add_subplot(top[0, 1]))

    for r in range(1, 5):
        inner = outer[r].subgridspec(1, 2, wspace=0.25)
        axes.append(fig.add_subplot(inner[0, 0]))
        axes.append(fig.add_subplot(inner[0, 1]))

    def legend(ax, lines):
        handles = [Line2D([], [], linestyle="none", label=l) for l in lines]
        ax.legend(handles=handles)

    def pretty_name(name):
        return (
            name
            .replace("_", " + ")
            .replace("sigma", "SD Shift")
            .replace("mu", "Mean Shift")
        )

    def plot(ax, data, name, show_custom=True):

        ax.hist(
            baseline,
            bins=bin_edges,
            weights=baseline,
            density=False,
            alpha=0.4
        )

        if show_custom:
            ax.hist(
                data,
                bins=bin_edges,
                weights=data,
                density=False,
                alpha=0.4
            )

        ax.set_xlim(0, 1000)

        if show_custom:
            legend(ax, [
                pretty_name(name),
                f"Mean = {np.mean(data):.4f}",
                f"SD = {np.std(data, ddof=1):.4f}"
            ])
        else:
            legend(ax, [
                "Baseline",
                f"Mean = {np.mean(baseline):.4f}",
                f"SD = {np.std(baseline, ddof=1):.4f}"
            ])

        ax.set_ylabel("Total Gap Length")

    plot(axes[0], baseline, "Baseline", False)

    for ax, (name, data) in zip(axes[1:], loaded_cases):
        plot(ax, data, name)

    plt.savefig(
        "LLSB_LLSA_gap_length_variations_weighted.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

def plot_baseline_vs_all_gap_length_variations_weighted_ordered(bins=50):
    from matplotlib.lines import Line2D

    def extract_gap(df):
        if "metric" in df.columns:
            return df[df["metric"] == "gap_length"]["value"].values
        return df["gap_length"].values

    save_dir = "Cached Data/Normal Distribution Variations"

    baseline = extract_gap(pd.read_csv(baseline_file))

    all_data = extract_gap(pd.read_csv(
        os.path.join(save_dir, "ALL_shifted_both_spacing_data.csv")
    ))

    combined = np.concatenate([baseline, all_data])

    bin_edges = np.linspace(0, 1000, bins + 1)

    fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    def legend(ax, lines):
        handles = [Line2D([], [], linestyle="none", label=l) for l in lines]
        ax.legend(handles=handles)

    # -----------------------------------------
    # Baseline
    # -----------------------------------------
    axes[0].hist(
        baseline,
        bins=bin_edges,
        weights=baseline,
        density=False,
        alpha=0.5
    )

    axes[0].set_xlim(0, 1000)
    axes[0].set_ylabel("Total Gap Length")

    legend(axes[0], [
        "Baseline",
        f"Mean = {np.mean(baseline):.4f}",
        f"SD = {np.std(baseline, ddof=1):.4f}"
    ])

    # -----------------------------------------
    # Overlay
    # -----------------------------------------
    axes[1].hist(
        baseline,
        bins=bin_edges,
        weights=baseline,
        density=False,
        alpha=0.4
    )

    axes[1].hist(
        all_data,
        bins=bin_edges,
        weights=all_data,
        density=False,
        alpha=0.4
    )

    axes[1].set_xlim(0, 1000)
    axes[1].set_ylabel("Total Gap Length")
    axes[1].set_xlabel("Gap Length (mm)")

    legend(axes[1], [
        "Baseline vs ALL",
        f"Baseline μ={np.mean(baseline):.4f}, σ={np.std(baseline, ddof=1):.4f}",
        f"ALL μ={np.mean(all_data):.4f}, σ={np.std(all_data, ddof=1):.4f}",
    ])

    plt.tight_layout()

    plt.savefig(
        "baseline_vs_all_gap_length_weighted.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

def plot_LT_CAM_overlap_length_variations_weighted_ordered(bins=50):
    from matplotlib.lines import Line2D

    def extract_overlap(df):
        if "metric" in df.columns:
            return df[df["metric"] == "overlap_length"]["value"].values
        return df["overlap_length"].values

    save_dir = "Cached Data/Normal Distribution Variations"

    baseline_df = pd.read_csv(baseline_file)
    baseline = extract_overlap(baseline_df)

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

    for name, file in cases:
        df = pd.read_csv(file)
        data = extract_overlap(df)
        loaded_cases.append((name, data))

    bin_edges = np.linspace(0, 1000, bins + 1)

    fig = plt.figure(figsize=(18, 18))
    outer = fig.add_gridspec(5, 1, hspace=0.3)

    axes = []

    top = outer[0].subgridspec(1, 3, width_ratios=[1, 2, 1])
    axes.append(fig.add_subplot(top[0, 1]))

    for r in range(1, 5):
        inner = outer[r].subgridspec(1, 2, wspace=0.25)
        axes.append(fig.add_subplot(inner[0, 0]))
        axes.append(fig.add_subplot(inner[0, 1]))

    baseline_mean = np.mean(baseline)
    baseline_std = np.std(baseline, ddof=1)

    def legend(ax, lines):
        handles = [Line2D([], [], linestyle="none", label=l) for l in lines]
        ax.legend(handles=handles, loc="best", handlelength=0)

    def pretty_name(name):
        return (
            name
            .replace("_", " + ")
            .replace("sigma", "SD Shift")
            .replace("mu", "Mean Shift")
        )

    def plot(ax, data, name, show_custom=True):

        ax.hist(
            baseline,
            bins=bin_edges,
            weights=baseline,
            alpha=0.4
        )

        if show_custom:
            ax.hist(
                data,
                bins=bin_edges,
                weights=data,
                alpha=0.4
            )

        ax.set_xlim(0, 1000)

        if show_custom:
            legend(ax, [
                pretty_name(name),
                f"Mean = {np.mean(data):.4f}",
                f"SD = {np.std(data, ddof=1):.4f}"
            ])
        else:
            legend(ax, [
                "Baseline",
                f"Mean = {baseline_mean:.4f}",
                f"SD = {baseline_std:.4f}"
            ])

        ax.set_ylabel("Total Overlap Length")

    plot(axes[0], baseline, "Baseline", False)

    for ax, (name, data) in zip(axes[1:], loaded_cases):
        plot(ax, data, name)

    plt.savefig(
        "LT_CAM_overlap_length_variations_weighted.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

def plot_LLSB_LLSA_overlap_length_variations_weighted_ordered(bins=50):
    from matplotlib.lines import Line2D

    def extract_overlap(df):
        if "metric" in df.columns:
            return df[df["metric"] == "overlap_length"]["value"].values
        return df["overlap_length"].values

    save_dir = "Cached Data/Normal Distribution Variations"

    baseline = extract_overlap(pd.read_csv(baseline_file))

    cases = [
        ("LLSA_sigma", os.path.join(save_dir, "LLSA_shifted_sigma_spacing_data.csv")),
        ("LLSA_mu", os.path.join(save_dir, "LLSA_shifted_mu_spacing_data.csv")),

        ("LLSB_sigma", os.path.join(save_dir, "LLSB_shifted_sigma_spacing_data.csv")),
        ("LLSB_mu", os.path.join(save_dir, "LLSB_shifted_mu_spacing_data.csv")),

        ("LLSB_LLSA_sigma", os.path.join(save_dir, "LLSB_LLSA_shifted_sigma_spacing_data.csv")),
        ("LLSB_LLSA_mu", os.path.join(save_dir, "LLSB_LLSA_shifted_mu_spacing_data.csv")),

        ("LLSB_LLSA_sigma_opposite", os.path.join(save_dir, "LLSB_LLSA_opposite_shifted_sigma_spacing_data.csv")),
        ("LLSB_LLSA_mu_opposite", os.path.join(save_dir, "LLSB_LLSA_opposite_shifted_mu_spacing_data.csv")),
    ]

    loaded_cases = []

    for name, file in cases:
        data = extract_overlap(pd.read_csv(file))
        loaded_cases.append((name, data))

    bin_edges = np.linspace(0, 1000, bins + 1)

    fig = plt.figure(figsize=(18, 18))
    outer = fig.add_gridspec(5, 1, hspace=0.3)

    axes = []

    top = outer[0].subgridspec(1, 3, width_ratios=[1, 2, 1])
    axes.append(fig.add_subplot(top[0, 1]))

    for r in range(1, 5):
        inner = outer[r].subgridspec(1, 2, wspace=0.25)
        axes.append(fig.add_subplot(inner[0, 0]))
        axes.append(fig.add_subplot(inner[0, 1]))

    def legend(ax, lines):
        handles = [Line2D([], [], linestyle="none", label=l) for l in lines]
        ax.legend(handles=handles)

    def pretty_name(name):
        return (
            name
            .replace("_", " + ")
            .replace("sigma", "SD Shift")
            .replace("mu", "Mean Shift")
        )

    def plot(ax, data, name, show_custom=True):

        ax.hist(
            baseline,
            bins=bin_edges,
            weights=baseline,
            alpha=0.4
        )

        if show_custom:
            ax.hist(
                data,
                bins=bin_edges,
                weights=data,
                alpha=0.4
            )

        ax.set_xlim(0, 1000)

        if show_custom:
            legend(ax, [
                pretty_name(name),
                f"Mean = {np.mean(data):.4f}",
                f"SD = {np.std(data, ddof=1):.4f}"
            ])
        else:
            legend(ax, [
                "Baseline",
                f"Mean = {np.mean(baseline):.4f}",
                f"SD = {np.std(baseline, ddof=1):.4f}"
            ])

        ax.set_ylabel("Total Overlap Length")

    plot(axes[0], baseline, "Baseline", False)

    for ax, (name, data) in zip(axes[1:], loaded_cases):
        plot(ax, data, name)

    plt.savefig(
        "LLSB_LLSA_overlap_length_variations_weighted.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

def plot_baseline_vs_all_overlap_length_variations_weighted_ordered(bins=50):
    from matplotlib.lines import Line2D

    def extract_overlap(df):
        if "metric" in df.columns:
            return df[df["metric"] == "overlap_length"]["value"].values
        return df["overlap_length"].values

    save_dir = "Cached Data/Normal Distribution Variations"

    baseline = extract_overlap(pd.read_csv(baseline_file))

    all_data = extract_overlap(pd.read_csv(
        os.path.join(save_dir, "ALL_shifted_both_spacing_data.csv")
    ))

    bin_edges = np.linspace(0, 1000, bins + 1)

    fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    def legend(ax, lines):
        handles = [Line2D([], [], linestyle="none", label=l) for l in lines]
        ax.legend(handles=handles)

    # -----------------------------------------
    # Baseline
    # -----------------------------------------
    axes[0].hist(
        baseline,
        bins=bin_edges,
        weights=baseline,
        alpha=0.5
    )

    axes[0].set_xlim(0, 1000)
    axes[0].set_ylabel("Total Overlap Length")

    legend(axes[0], [
        "Baseline",
        f"Mean = {np.mean(baseline):.4f}",
        f"SD = {np.std(baseline, ddof=1):.4f}"
    ])

    # -----------------------------------------
    # Overlay
    # -----------------------------------------
    axes[1].hist(
        baseline,
        bins=bin_edges,
        weights=baseline,
        alpha=0.4
    )

    axes[1].hist(
        all_data,
        bins=bin_edges,
        weights=all_data,
        alpha=0.4
    )

    axes[1].set_xlim(0, 1000)
    axes[1].set_ylabel("Total Overlap Length")
    axes[1].set_xlabel("Overlap Length (mm)")

    legend(axes[1], [
        "Baseline vs ALL",
        f"Baseline μ={np.mean(baseline):.4f}, σ={np.std(baseline, ddof=1):.4f}",
        f"ALL μ={np.mean(all_data):.4f}, σ={np.std(all_data, ddof=1):.4f}",
    ])

    plt.tight_layout()

    plt.savefig(
        "baseline_vs_all_overlap_length_weighted.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

def plot_LT_CAM_combined_gap_overlap_weighted_ordered(bins=50):
    from matplotlib.lines import Line2D

    def extract_combined(df):

        if "metric" in df.columns:
            gaps = df[df["metric"] == "gap_length"]["value"].values
            overlaps = -df[df["metric"] == "overlap_length"]["value"].values
        else:
            gaps = df["gap_length"].values
            overlaps = -df["overlap_length"].values

        return np.concatenate([gaps, overlaps])

    save_dir = "Cached Data/Normal Distribution Variations"

    baseline = extract_combined(pd.read_csv(baseline_file))

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

    for name, file in cases:
        data = extract_combined(pd.read_csv(file))
        loaded_cases.append((name, data))

    bin_edges = np.linspace(-1000, 1000, bins + 1)

    fig = plt.figure(figsize=(18, 18))
    outer = fig.add_gridspec(5, 1, hspace=0.3)

    axes = []

    top = outer[0].subgridspec(1, 3, width_ratios=[1, 2, 1])
    axes.append(fig.add_subplot(top[0, 1]))

    for r in range(1, 5):
        inner = outer[r].subgridspec(1, 2, wspace=0.25)
        axes.append(fig.add_subplot(inner[0, 0]))
        axes.append(fig.add_subplot(inner[0, 1]))

    def legend(ax, lines):
        handles = [Line2D([], [], linestyle="none", label=l) for l in lines]
        ax.legend(handles=handles, loc="best", handlelength=0)

    def pretty_name(name):
        return (
            name
            .replace("_", " + ")
            .replace("sigma", "SD Shift")
            .replace("mu", "Mean Shift")
        )

    def plot(ax, data, name, show_custom=True):

        ax.hist(
            baseline,
            bins=bin_edges,
            weights=np.abs(baseline),
            alpha=0.4
        )

        if show_custom:
            ax.hist(
                data,
                bins=bin_edges,
                weights=np.abs(data),
                alpha=0.4
            )

        ax.axvline(0, color="black", linestyle=":")
        ax.set_xlim(-1000, 1000)

        if show_custom:
            legend(ax, [
                pretty_name(name),
                f"Mean = {np.mean(data):.4f}",
                f"SD = {np.std(data, ddof=1):.4f}"
            ])
        else:
            legend(ax, [
                "Baseline",
                f"Mean = {np.mean(baseline):.4f}",
                f"SD = {np.std(baseline, ddof=1):.4f}"
            ])

        ax.set_ylabel("Total Defect Length")

    plot(axes[0], baseline, "Baseline", False)

    for ax, (name, data) in zip(axes[1:], loaded_cases):
        plot(ax, data, name)

    plt.savefig(
        "LT_CAM_combined_gap_overlap_weighted.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

def plot_LLSB_LLSA_combined_gap_overlap_weighted_ordered(bins=50):
    from matplotlib.lines import Line2D

    def extract_combined(df):

        if "metric" in df.columns:
            gaps = df[df["metric"] == "gap_length"]["value"].values
            overlaps = -df[df["metric"] == "overlap_length"]["value"].values
        else:
            gaps = df["gap_length"].values
            overlaps = -df["overlap_length"].values

        return np.concatenate([gaps, overlaps])

    save_dir = "Cached Data/Normal Distribution Variations"

    baseline = extract_combined(pd.read_csv(baseline_file))

    cases = [
        ("LLSA_sigma", os.path.join(save_dir, "LLSA_shifted_sigma_spacing_data.csv")),
        ("LLSA_mu", os.path.join(save_dir, "LLSA_shifted_mu_spacing_data.csv")),

        ("LLSB_sigma", os.path.join(save_dir, "LLSB_shifted_sigma_spacing_data.csv")),
        ("LLSB_mu", os.path.join(save_dir, "LLSB_shifted_mu_spacing_data.csv")),

        ("LLSB_LLSA_sigma", os.path.join(save_dir, "LLSB_LLSA_shifted_sigma_spacing_data.csv")),
        ("LLSB_LLSA_mu", os.path.join(save_dir, "LLSB_LLSA_shifted_mu_spacing_data.csv")),

        ("LLSB_LLSA_sigma_opposite", os.path.join(save_dir, "LLSB_LLSA_opposite_shifted_sigma_spacing_data.csv")),
        ("LLSB_LLSA_mu_opposite", os.path.join(save_dir, "LLSB_LLSA_opposite_shifted_mu_spacing_data.csv")),
    ]

    loaded_cases = []

    for name, file in cases:
        data = extract_combined(pd.read_csv(file))
        loaded_cases.append((name, data))

    bin_edges = np.linspace(-1000, 1000, bins + 1)

    fig = plt.figure(figsize=(18, 18))
    outer = fig.add_gridspec(5, 1, hspace=0.3)

    axes = []

    top = outer[0].subgridspec(1, 3, width_ratios=[1, 2, 1])
    axes.append(fig.add_subplot(top[0, 1]))

    for r in range(1, 5):
        inner = outer[r].subgridspec(1, 2, wspace=0.25)
        axes.append(fig.add_subplot(inner[0, 0]))
        axes.append(fig.add_subplot(inner[0, 1]))

    def legend(ax, lines):
        handles = [Line2D([], [], linestyle="none", label=l) for l in lines]
        ax.legend(handles=handles)

    def pretty_name(name):
        return (
            name
            .replace("_", " + ")
            .replace("sigma", "SD Shift")
            .replace("mu", "Mean Shift")
        )

    def plot(ax, data, name, show_custom=True):

        ax.hist(
            baseline,
            bins=bin_edges,
            weights=np.abs(baseline),
            alpha=0.4
        )

        if show_custom:
            ax.hist(
                data,
                bins=bin_edges,
                weights=np.abs(data),
                alpha=0.4
            )

        ax.axvline(0, color="black", linestyle=":")
        ax.set_xlim(-1000, 1000)

        if show_custom:
            legend(ax, [
                pretty_name(name),
                f"Mean = {np.mean(data):.4f}",
                f"SD = {np.std(data, ddof=1):.4f}"
            ])
        else:
            legend(ax, [
                "Baseline",
                f"Mean = {np.mean(baseline):.4f}",
                f"SD = {np.std(baseline, ddof=1):.4f}"
            ])

        ax.set_ylabel("Total Defect Length")

    plot(axes[0], baseline, "Baseline", False)

    for ax, (name, data) in zip(axes[1:], loaded_cases):
        plot(ax, data, name)

    plt.savefig(
        "LLSB_LLSA_combined_gap_overlap_weighted.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

def plot_baseline_vs_all_combined_gap_overlap_weighted_ordered(bins=50):
    from matplotlib.lines import Line2D

    def extract_combined(df):

        if "metric" in df.columns:
            gaps = df[df["metric"] == "gap_length"]["value"].values
            overlaps = -df[df["metric"] == "overlap_length"]["value"].values
        else:
            gaps = df["gap_length"].values
            overlaps = -df["overlap_length"].values

        return np.concatenate([gaps, overlaps])

    save_dir = "Cached Data/Normal Distribution Variations"

    baseline = extract_combined(pd.read_csv(baseline_file))

    all_data = extract_combined(pd.read_csv(
        os.path.join(save_dir, "ALL_shifted_both_spacing_data.csv")
    ))

    bin_edges = np.linspace(-1000, 1000, bins + 1)

    fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    def legend(ax, lines):
        handles = [Line2D([], [], linestyle="none", label=l) for l in lines]
        ax.legend(handles=handles)

    # -----------------------------------------
    # Baseline
    # -----------------------------------------
    axes[0].hist(
        baseline,
        bins=bin_edges,
        weights=np.abs(baseline),
        alpha=0.5
    )

    axes[0].axvline(0, color="black", linestyle=":")
    axes[0].set_xlim(-1000, 1000)
    axes[0].set_ylabel("Total Defect Length")

    legend(axes[0], [
        "Baseline",
        f"Mean = {np.mean(baseline):.4f}",
        f"SD = {np.std(baseline, ddof=1):.4f}"
    ])

    # -----------------------------------------
    # Overlay
    # -----------------------------------------
    axes[1].hist(
        baseline,
        bins=bin_edges,
        weights=np.abs(baseline),
        alpha=0.4
    )

    axes[1].hist(
        all_data,
        bins=bin_edges,
        weights=np.abs(all_data),
        alpha=0.4
    )

    axes[1].axvline(0, color="black", linestyle=":")
    axes[1].set_xlim(-1000, 1000)

    axes[1].set_ylabel("Total Defect Length")
    axes[1].set_xlabel("Overlap ← Length → Gap")

    legend(axes[1], [
        "Baseline vs ALL",
        f"Baseline μ={np.mean(baseline):.4f}, σ={np.std(baseline, ddof=1):.4f}",
        f"ALL μ={np.mean(all_data):.4f}, σ={np.std(all_data, ddof=1):.4f}",
    ])

    plt.tight_layout()

    plt.savefig(
        "baseline_vs_all_combined_gap_overlap_weighted.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

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
    # Run this ONCE if the opposite LT+CAM or LLSA/LLSB files do not exist yet
    # --------------------------------------------------
    # generate_LT_CAM_opposite_spacing_data(runs=100, tows=31)
    # generate_LLSB_LLSA_opposite_spacing_data(runs=100, tows=31)

    # --------------------------------------------------
    # Plot spacing variations or gap length variations
    # --------------------------------------------------
    # plot_LT_CAM_spacing_variations_ordered(bins=100)
    # plot_LLSB_LLSA_spacing_variations_ordered(bins=100)
    # plot_baseline_vs_all_spacing_variations_ordered(bins=100)
    # plot_LT_CAM_gap_length_variations_ordered(bins=150)
    # plot_LLSB_LLSA_gap_length_variations_ordered(bins=150)
    # plot_baseline_vs_all_gap_length_variations_ordered(bins=150)

    # compute_baseline_spacing()
    # params_test = {
    #     "LT": (0.00, 0.1),
    #     "CAM": (0.00, 0.1),
    #     "LLSB": (0.05, 0.1),
    #     "LLSA": (0.00, 0.1)}

    # KDE_spacing_from_normals(params_test, runs=100, sensor_type="LLSB", distribution_parameter="mu") # GENERATES ONE DATA SET BASED ON PARAMS_TEST
    # run_spacing_multiple_simulations() # GENERATES A LOT OF DATA. Takes 25 min!

    # plot_spacing_distribution("Cached Data/Normal Distribution Variations/LT_shifted_mu_spacing_data.csv")
    # plot_gap_length_distribution("Cached Data/Normal Distribution Variations/LT_shifted_mu_spacing_data.csv")

    # Test functions NOT FOR PAPER:
    # plot_LT_CAM_combined_gap_overlap_weighted_ordered()
    # plot_LLSB_LLSA_combined_gap_overlap_weighted_ordered()
    # plot_baseline_vs_all_combined_gap_overlap_weighted_ordered()