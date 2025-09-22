"""This file is used to generate a fft plots to validate the model"""

##############################################################################################################

# External imports
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

#Internal imports
from Model_ALL_Validation_Tow_Visualiser import plot_real_tow, plot_simulated_vs_real_tow
from constants import number_of_steps, font_large, Consecutive_Error_Bins

##############################################################################################################
"""Functions"""

def fft_comparison(tow: int, plot=True):
    """
    Compare FFT of real vs simulated centerlines for a given tow.

    Parameters
    ----------
    tow : int
        Tow index to analyze.
    tow_length_mm : int, optional
        Length of the tow in mm (default 1000).
    plot : bool, optional
        Whether to plot amplitude and phase comparison (default True).

    Returns
    -------
    mse_amplitude : float
        Mean squared error between amplitude spectra.
    mse_phase : float
        Mean squared error between phase spectra.
    """
    print(f"--- FFT Comparison for Tow {tow} ---")

    # --- Get real and simulated data ---
    print("Generating real and simulated tow data...")
    real_data, sim_data = plot_simulated_vs_real_tow(tow, tow_length_mm=1000, plot=False)
    print(f"Real centerline length: {len(real_data)}")
    print(f"Simulated centerline length: {len(sim_data)}")

    # --- Extract centerlines ---
    y_real = real_data["centerline"].to_numpy()
    y_sim = np.interp(real_data["x_mm"], sim_data["x_mm"], sim_data["centerline"])
    print("Centerlines extracted and aligned for FFT.")

    # --- FFT ---
    fft_real = np.fft.fft(y_real)
    fft_sim = np.fft.fft(y_sim)
    print("FFT computed for both real and simulated centerlines.")

    # Only use positive frequencies
    n = len(y_real)
    freqs = np.fft.fftfreq(n, d=(real_data["x_mm"][1] - real_data["x_mm"][0]))
    pos_mask = freqs >= 0

    fft_real_pos = fft_real[pos_mask]
    fft_sim_pos = fft_sim[pos_mask]
    freqs_pos = freqs[pos_mask]
    print(f"Number of positive frequencies considered: {len(freqs_pos)}")

    # --- Amplitude and phase ---
    amp_real = np.abs(fft_real_pos)
    amp_sim = np.abs(fft_sim_pos)
    phase_real = np.angle(fft_real_pos)
    phase_sim = np.angle(fft_sim_pos)
    print("Amplitude and phase spectra computed.")

    # --- MSE calculation ---
    mse_amplitude = mean_squared_error(amp_real, amp_sim)
    mse_phase = mean_squared_error(phase_real, phase_sim)
    print(f"MSE Amplitude: {mse_amplitude:.4f}")
    print(f"MSE Phase: {mse_phase:.4f}")

    # --- Plot ---
    if plot:
        print("Plotting amplitude and phase comparison...")
        fig, ax = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

        ax[0].plot(freqs_pos, amp_real, label="Real amplitude", color="blue")
        ax[0].plot(freqs_pos, amp_sim, label="Sim amplitude", color="red", linestyle="--")
        ax[0].set_ylabel("Amplitude")
        ax[0].legend()
        ax[0].grid(True)

        ax[1].plot(freqs_pos, phase_real, label="Real phase", color="blue")
        ax[1].plot(freqs_pos, phase_sim, label="Sim phase", color="red", linestyle="--")
        ax[1].set_xlabel("Frequency (1/mm)")
        ax[1].set_ylabel("Phase (radians)")
        ax[1].legend()
        ax[1].grid(True)

        plt.suptitle(f"FFT Comparison: Real vs Simulated Centerlines (Tow {tow})", fontsize=16)
        plt.tight_layout()
        plt.show()
        print("Plotting done.")

    print(f"--- End of FFT Comparison for Tow {tow} ---\n")
    return mse_amplitude, mse_phase

def fft_comparison_all_tows_vs_simulated(plot=False):
    """
    Compare FFT of all 31 real tows against a single simulated tow.

    Parameters
    ----------
    simulated_tow_length_mm : int, optional
        Length of the simulated tow (default 1000).
    plot : bool, optional
        Whether to plot each tow's comparison (default False).

    Returns
    -------
    results : dict
        Dictionary containing MSEs for each tow:
        {
            tow_index: {'mse_amplitude': value, 'mse_phase': value},
            ...
        }
    avg_mse_amplitude : float
        Average MSE of amplitude across all 31 tows.
    avg_mse_phase : float
        Average MSE of phase across all 31 tows.
    """
    results = {}
    mse_amp_list = []
    mse_phase_list = []

    print("=== Comparing all 31 tows against a single simulated tow ===")

    # Generate a single simulated tow once
    print("Generating one simulated tow for comparison...")
    _, sim_data = plot_simulated_vs_real_tow(tow=2, tow_length_mm=1000, plot=False)
    print("Simulated tow ready.\n")

    # Loop through all 31 tows
    for tow_index in range(2, 31):
        print(f"--- Tow {tow_index} ---")

        # Get real tow
        real_data, _ = plot_simulated_vs_real_tow(tow=tow_index, tow_length_mm=1000, plot=False)

        # Run FFT comparison using aligned centerlines
        y_real = real_data["centerline"].to_numpy()
        y_sim = np.interp(real_data["x_mm"], sim_data["x_mm"], sim_data["centerline"])

        fft_real = np.fft.fft(y_real)
        fft_sim = np.fft.fft(y_sim)

        n = len(y_real)
        freqs = np.fft.fftfreq(n, d=(real_data["x_mm"][1] - real_data["x_mm"][0]))
        pos_mask = freqs >= 0

        amp_real = np.abs(fft_real[pos_mask])
        amp_sim = np.abs(fft_sim[pos_mask])
        phase_real = np.angle(fft_real[pos_mask])
        phase_sim = np.angle(fft_sim[pos_mask])

        mse_amp = mean_squared_error(amp_real, amp_sim)
        mse_phase = mean_squared_error(phase_real, phase_sim)

        print(f"MSE Amplitude: {mse_amp:.4f}, MSE Phase: {mse_phase:.4f}\n")

        results[tow_index] = {'mse_amplitude': mse_amp, 'mse_phase': mse_phase}
        mse_amp_list.append(mse_amp)
        mse_phase_list.append(mse_phase)

        # Optional plotting
        if plot:
            fig, ax = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
            ax[0].plot(freqs[pos_mask], amp_real, label="Real amplitude", color="blue")
            ax[0].plot(freqs[pos_mask], amp_sim, label="Sim amplitude", color="red", linestyle="--")
            ax[0].set_ylabel("Amplitude")
            ax[0].legend()
            ax[0].grid(True)

            ax[1].plot(freqs[pos_mask], phase_real, label="Real phase", color="blue")
            ax[1].plot(freqs[pos_mask], phase_sim, label="Sim phase", color="red", linestyle="--")
            ax[1].set_xlabel("Frequency (1/mm)")
            ax[1].set_ylabel("Phase (radians)")
            ax[1].legend()
            ax[1].grid(True)
            plt.suptitle(f"Tow {tow_index} FFT Comparison", fontsize=16)
            plt.tight_layout()
            plt.show()

    # Compute average MSEs
    avg_mse_amplitude = np.mean(mse_amp_list)
    avg_mse_phase = np.mean(mse_phase_list)
    print(f"=== Average MSE across all 31 tows ===")
    print(f"Average MSE Amplitude: {avg_mse_amplitude:.4f}")
    print(f"Average MSE Phase: {avg_mse_phase:.4f}")

    print("=== All 31 tows compared ===")
    return results, avg_mse_amplitude, avg_mse_phase

##############################################################################################################
"""Run this file"""

def main():
    fft_comparison(8)
    # fft_comparison_all_tows_vs_simulated()

if __name__ == "__main__":
    main() # makes sure this only runs if you run *this* file, not if this file is imported somewhere else