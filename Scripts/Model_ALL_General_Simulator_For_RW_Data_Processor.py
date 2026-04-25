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

RW_PROPOSAL_STD = {
    "LT": get_proposal_distribution("LT"),
    "CAM": get_proposal_distribution("CAM"),
    "LLS_B": get_proposal_distribution("LLS_B"),
    "LLS_A": get_proposal_distribution("LLS_A")}

STEP_CACHE = {
    "LT": get_n_steps("LT"),
    "CAM": get_n_steps("CAM"),
    "LLS_B": get_n_steps("LLS_B"),
    "LLS_A": get_n_steps("LLS_A")}

def NORMAL_distribution(mu, sigma):
    from scipy.stats import norm
    return lambda x: norm.pdf(x, loc=mu, scale=sigma)

def compute_spacing_distribution(params, num_tows=31, tow_length_mm=1000):
    """
    Returns ALL local spacing values (mm) across all tow pairs and x positions.
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

    for _ in range(num_tows):

        LT = generate_random_walk(
            "LT", LT_steps, RW_PROPOSAL_STD["LT"],
            NORMAL_distribution(*params["LT"]),
            norm,
            params["LT"]
        )

        CAM = generate_random_walk(
            "CAM", CAM_steps, RW_PROPOSAL_STD["CAM"],
            NORMAL_distribution(*params["CAM"]),
            norm,
            params["CAM"]
        )

        LLS_B = generate_random_walk(
            "LLS_B", LLSB_steps, RW_PROPOSAL_STD["LLS_B"],
            NORMAL_distribution(*params["LLSB"]),
            norm,
            params["LLSB"]
        )

        LLS_A = generate_random_walk(
            "LLS_A", LLSA_steps, RW_PROPOSAL_STD["LLS_A"],
            NORMAL_distribution(*params["LLSA"]),
            norm,
            params["LLSA"]
        )

        def interp(arr):
            return np.interp(
                np.linspace(0, len(arr)-1, len(x)),
                np.arange(len(arr)),
                arr
            )

        LT = interp(LT)
        CAM = interp(CAM)
        LLS_B = interp(LLS_B)
        LLS_A = interp(LLS_A)

        center = tow_offset + CAM + LT
        width = 6.35 + LLS_B

        top = center + 0.5 * width
        bottom = center - 0.5 * width

        top_paths.append(top)
        bottom_paths.append(bottom)

        tow_offset += 6.35

    # -----------------------------
    # COLLECT SPACING (KEY PART)
    # -----------------------------
    spacing_data = []

    for i in range(len(top_paths) - 1):
        diff = bottom_paths[i + 1] - top_paths[i]
        spacing_data.extend(diff)   # keep ALL values (not integrated)

    return np.array(spacing_data)

def compute_baseline_spacing(runs=100, tows=31):
    """
    Compute baseline spacing distribution ONCE, store in DataFrame,
    and always save to CSV.
    """

    baseline_params = {
        "LT": (-0.08, 0.06),
        "CAM": (-0.08, 0.06),
        "LLSB": (-0.08, 0.06),
        "LLSA": (-0.08, 0.06)}

    global normal_mode
    normal_mode = True

    baseline_data = []

    # progress bar
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
        baseline_data.extend(
            compute_spacing_distribution(baseline_params, num_tows=tows)
        )
        print_progress(i, runs, start_time)

    print("\nBaseline ready.\n")

    baseline_df = pd.DataFrame({
        "spacing": baseline_data,
        "type": "baseline"
    })

    # -----------------------------
    # SAVE TO CSV (ALWAYS)
    # -----------------------------
    save_path = "Cached Data/Normal Distribution Variations/baseline_spacing_data.csv"

    # ensure directory exists
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

    if sensor_type not in valid_sensors: raise ValueError(f"sensor_type must be one of {valid_sensors}") 
    if distribution_parameter not in valid_params: raise ValueError(f"distribution_parameter must be one of {valid_params}")

    baseline_df = pd.read_csv(baseline_file)

    # -----------------------------
    # GENERATE DATA
    # -----------------------------
    if custom_params is None:
        raise ValueError("Provide custom_params.")

    args = [(custom_params, tows) for _ in range(runs)]

    with ProcessPoolExecutor() as executor:
        results = list(tqdm(
            executor.map(run_single_spacing_worker, args),
            total=runs,
            desc="Generating spacing data"
        ))

    customdata_df = pd.DataFrame({"spacing": np.concatenate(results)})

    if save_csv:

        base = f"Cached Data/Normal Distribution Variations/{sensor_type}_shifted_{distribution_parameter}_spacing_data"
        filename = base + ".csv"

        i = 1
        while os.path.exists(filename):
            filename = f"{base}_{i}.csv"
            i += 1

        customdata_df.to_csv(filename, index=False)
        print(f"Saved: {filename}")

    return customdata_df, baseline_df

def run_spacing_multiple_simulations(runs=100, tows=31):
    
    base = {
        "LT": (-0.08, 0.06),
        "CAM": (-0.08, 0.06),
        "LLSB": (-0.08, 0.06),
        "LLSA": (-0.08, 0.06)}

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
                mu = 0
            elif change == "std":
                sigma = 0.12
            elif change == "both":
                mu = 0
                sigma = 0.12

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

        custom_params = modify(base, sensors=sensors, change=change)

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
    custom_data = pd.read_csv(custom_file)["spacing"].values
    baseline_data = pd.read_csv(baseline_file)["spacing"].values
    
    plt.figure(figsize=(10, 6))

    plt.hist(baseline_data, bins=bins, density=True, alpha=0.5, label="Baseline")
    plt.hist(custom_data, bins=bins, density=True, alpha=0.5, label="Custom")

    try:
        sns.kdeplot(baseline_data, linewidth=2)
        sns.kdeplot(custom_data, linewidth=2)
    except:
        pass

    plt.axvline(0, color="black", linestyle=":", label="Ideal (0 gap)")
    plt.xlabel("Gap (mm)")
    plt.ylabel("Density")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_all_spacing_variations(bins=100):
    
    figsize=(18, 18)
    baseline = pd.read_csv(baseline_file)["spacing"].values

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

    def plot(ax, data, title):
        ax.hist(baseline, bins=bins, density=True, alpha=0.4)
        ax.hist(data, bins=bins, density=True, alpha=0.4)

        try:
            sns.kdeplot(baseline, ax=ax)
            sns.kdeplot(data, ax=ax)
        except:
            pass

        ax.set_title(title)

    plot(axes[0], baseline, "Baseline")

    for i, (title, file) in enumerate(cases, start=1):
        df = pd.read_csv(f"Cached Data/Normal Distribution Variations/{file}")
        plot(axes[i], df["spacing"].values, title)

    for j in range(len(cases)+1, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.show()

############################################################################################################################################
"""Generate Data"""
if __name__ == "__main__": # REQUIRED for multiprocessing (especially on Windows/macOS)
    from multiprocessing import freeze_support, set_start_method

    freeze_support()

    try:
        set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    params_test = {
        "LT": (-0.08, 0.06),
        "CAM": (-0.08, 0.06),
        "LLSB": (-0.08, 0.06),
        "LLSA": (-0.08, 0.06)}
    
    # compute_baseline_spacing()
    # KDE_spacing_from_normals(params_test, runs=100, sensor_type="LT_CAM", distribution_parameter="mu") # GENERATES ONE DATA SET BASED ON PARAMS_TEST
    # run_spacing_multiple_simulations() # GENERATES A LOT OF DATA. Takes 25 min!

"""Generate Graphs"""
plot_spacing_distribution("Cached Data/Normal Distribution Variations/LT_shifted_mu_spacing_data.csv")
# plot_all_spacing_variations(bins=100) # Takes a minute