import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from mpl_toolkits.mplot3d import Axes3D
from Model_ALL_Simulation import generate_multitow_layout
from Handling_ALL_Functions import get_synced_data
import Model_ALL_Simulation as sim_model

def compute_fft(signal, sampling_rate):
    n = len(signal)
    fft_result = np.fft.fft(signal)
    freq = np.fft.fftfreq(n, d=1 / sampling_rate)
    magnitude = np.abs(fft_result) / n
    positive = freq > 0
    return freq[positive], magnitude[positive]

def find_best_nsteps_and_bins(tow_range=range(2, 8), nsteps_candidates=None, bin_candidates=None, n_repeats=10):
    if nsteps_candidates is None:
        nsteps_candidates = list(range(100, 600, 10))
    if bin_candidates is None:
        bin_candidates = list(range(30, 300, 5))

    mse_surface = np.zeros((len(bin_candidates), len(nsteps_candidates)))

    for tow in tow_range:
        print(f"Processing Tow {tow}...")
        df = get_synced_data(tow=tow, spacesynced=True)
        cam = df["center_CAM"].dropna().values
        lt = df["error_LT"].dropna().values
        x_pos = df["x"].dropna().values
        min_len = min(len(cam), len(lt), len(x_pos))
        cam, lt, x_pos = cam[:min_len], lt[:min_len], x_pos[:min_len]

        offset_real = cam + lt
        valid_indices = x_pos <= x_pos[0] + 1000
        offset_real_mm = offset_real[valid_indices]
        x_pos_valid = x_pos[valid_indices]

        # Real FFT
        length_between_points = (x_pos_valid[-1] - x_pos_valid[0]) / len(x_pos_valid)
        sampling_rate_real = 1 / length_between_points
        freq_real, mag_real = compute_fft(offset_real_mm, sampling_rate_real)

        for b_idx, num_bins in enumerate(bin_candidates):
            # Fit once per bin count (per tow)
            bin_stats_cam, slope_cam, intercept_cam, _, _, _, x_sorted_cam, bin_edges_cam, devs_cam = sim_model.consecutive_error(
                "CAM", test_ratio=0.5, num_bins=num_bins, bins_show=False, plot_fit=False)
            bin_stats_lt, slope_lt, intercept_lt, _, _, _, x_sorted_lt, bin_edges_lt, devs_lt = sim_model.consecutive_error(
                "LT", test_ratio=0.5, num_bins=num_bins, bins_show=False, plot_fit=False)

            for s_idx, n_steps in enumerate(nsteps_candidates):
                total_mse = 0.0

                for _ in range(n_repeats):
                    start_cam = np.random.uniform(-0.4, 0.6)
                    start_lt = np.random.uniform(-1.0, -0.8)

                    cam_path = sim_model.generate_error_path(start_cam, n_steps, slope_cam, intercept_cam,
                                                             x_sorted_cam, bin_edges_cam, devs_cam)
                    lt_path = sim_model.generate_error_path(start_lt, n_steps, slope_lt, intercept_lt,
                                                            x_sorted_lt, bin_edges_lt, devs_lt)
                    simulated_centerline = cam_path + lt_path
                    sampling_rate_sim = n_steps / 1000.0
                    freq_sim, mag_sim = compute_fft(simulated_centerline, sampling_rate_sim)

                    min_len_fft = min(len(mag_real), len(mag_sim))
                    mse = mean_squared_error(mag_real[:min_len_fft], mag_sim[:min_len_fft])
                    total_mse += mse

                mse_surface[b_idx, s_idx] += total_mse  # sum across tows

    # Plotting
    X, Y = np.meshgrid(nsteps_candidates, bin_candidates)
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, mse_surface, cmap='viridis')
    ax.set_xlabel("number of steps")
    ax.set_ylabel("number of bins")
    ax.set_zlabel("Total MSE")
    plt.tight_layout()
    plt.show()

    min_mse_idx = np.unravel_index(np.argmin(mse_surface), mse_surface.shape)
    optimal_bins = bin_candidates[min_mse_idx[0]]
    optimal_steps = nsteps_candidates[min_mse_idx[1]]
    print(f"Optimal Configuration -> n_steps: {optimal_steps}, num_bins: {optimal_bins}, Total MSE: {mse_surface[min_mse_idx]:.4f}")

    return mse_surface, optimal_steps, optimal_bins
find_best_nsteps_and_bins()

#360 steps x 180 bins, total MSE: 0.0017