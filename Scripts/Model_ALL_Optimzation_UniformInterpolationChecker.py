# compare_interpolation_traverse.py
# Standalone comparison of NON-INTERPOLATED vs INTERPOLATED (uniform) traverse edges for a given tow.
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

# ---- Your project import (must be available on PYTHONPATH) ----
from Data_ALL_traverse import traverse_tow_constructor

# -------- helper: pick target_steps to match native density --------
def choose_target_steps_from_raw(x_raw: np.ndarray, scale: float = 1.0) -> int:
    """
    Choose target_steps so Δx_uniform ≈ (median Δx_native)/scale.
    scale = 1.0 -> match native; >1 densify; <1 coarsen (not recommended unless anti-aliased).
    """
    x_raw = np.asarray(x_raw, dtype=float)
    if x_raw.size < 2:
        return max(int(x_raw.size), 2)
    dx_med = float(np.median(np.diff(x_raw)))
    L = float(x_raw[-1] - x_raw[0])
    n = int(round(L / max(dx_med / max(scale, 1e-6), 1e-12))) + 1
    return max(n, 2)

def load_traverse_edges(tow: int, per_edge_normalize: bool = True):
    """
    Returns native (non-uniform) right-x, left edge, right edge arrays:
        x_raw, left_raw, right_raw (all np.ndarray)
    """
    df = traverse_tow_constructor(tow)  # expects columns: x_right, y_right, x_left, y_left
    if df is None:
        raise ValueError(f"Tow {tow} not available from traverse.")

    x_r = df["x_right"].to_numpy(dtype=float)
    y_r = df["y_right"].to_numpy(dtype=float)
    x_l = df["x_left"].to_numpy(dtype=float)
    y_l = df["y_left"].to_numpy(dtype=float)

    # Use right-edge x as the common axis
    x = x_r.copy()
    left = y_l.copy()
    right = y_r.copy()

    # Clean & sort
    m = np.isfinite(x) & np.isfinite(left) & np.isfinite(right)
    x, left, right = x[m], left[m], right[m]
    order = np.argsort(x, kind="mergesort")
    x, left, right = x[order], left[order], right[order]
    # Make x strictly increasing (drop exact duplicates)
    if x.size > 1:
        uniq = np.ones_like(x, dtype=bool)
        uniq[1:] = x[1:] > x[:-1]
        x, left, right = x[uniq], left[uniq], right[uniq]

    if per_edge_normalize and len(left) > 0 and len(right) > 0:
        left = left - left[0]
        right = right - right[0]

    return x, left, right

def uniform_interpolate(
    x: np.ndarray, left: np.ndarray, right: np.ndarray, target_steps: int | None = None, scale: float = 1.0
):
    """
    Uniform resampling on [x[0], x[-1]].
    - If target_steps is None, it auto-matches native density using median Δx (optionally densify with scale>1).
    Returns x_uni, left_uni, right_uni.
    """
    if target_steps is None:
        target_steps = choose_target_steps_from_raw(x, scale=scale)
    x_uni = np.linspace(x[0], x[-1], int(target_steps))
    left_uni = np.interp(x_uni, x, left)
    right_uni = np.interp(x_uni, x, right)
    return x_uni, left_uni, right_uni

def compare_interpolation_for_tow(
    tow: int,
    target_steps: int | None = None,     # None => auto-match native density
    per_edge_normalize: bool = True,
    scale: float = 1.0,                  # >1 = denser; keep 1.0 to match native
    figsize=(14, 10)
):
    # 1) Load native (non-interpolated)
    x_raw, left_raw, right_raw = load_traverse_edges(tow, per_edge_normalize=per_edge_normalize)
    if len(x_raw) < 3:
        raise ValueError("Not enough native samples to compare.")

    # 2) Interpolate to uniform grid (auto-matched density if target_steps=None)
    x_uni, left_uni, right_uni = uniform_interpolate(
        x_raw, left_raw, right_raw, target_steps=target_steps, scale=scale
    )

    # 3) Compare on RAW x-grid (avoids bias). Interpolate uniform back to raw-x.
    left_uni_on_raw  = np.interp(x_raw, x_uni, left_uni)
    right_uni_on_raw = np.interp(x_raw, x_uni, right_uni)

    mse_left  = mean_squared_error(left_raw,  left_uni_on_raw)
    mse_right = mean_squared_error(right_raw, right_uni_on_raw)

    # Residuals (interp_on_raw - raw)
    res_left  = left_uni_on_raw  - left_raw
    res_right = right_uni_on_raw - right_raw

    # Δx stats
    dx = np.diff(x_raw)
    dx_mean, dx_std = float(np.mean(dx)), float(np.std(dx))
    dx_min, dx_max  = float(np.min(dx)), float(np.max(dx))
    dx_uni = (x_uni[-1] - x_uni[0]) / (len(x_uni) - 1) if len(x_uni) > 1 else np.nan

    # 4) Plots (explicit color-differentiated overlays)
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(3, 2, height_ratios=[1.2, 0.8, 1.0], hspace=0.35, wspace=0.25)

    # Left edge overlay
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(x_raw, left_raw,  label="RAW (native)", linewidth=1.7)
    ax1.plot(x_uni, left_uni,  label=f"Interpolated (uniform, N={len(x_uni)})", linewidth=1.7, alpha=0.9)
    ax1.set_title(f"Left Edge — Overlay (MSE={mse_left:.6g})")
    ax1.set_xlabel("x [mm]"); ax1.set_ylabel("offset [mm]"); ax1.legend()

    # Right edge overlay
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(x_raw, right_raw, label="RAW (native)", linewidth=1.7)
    ax2.plot(x_uni, right_uni, label=f"Interpolated (uniform, N={len(x_uni)})", linewidth=1.7, alpha=0.9)
    ax2.set_title(f"Right Edge — Overlay (MSE={mse_right:.6g})")
    ax2.set_xlabel("x [mm]"); ax2.set_ylabel("offset [mm]"); ax2.legend()

    # Residuals left
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.axhline(0, linewidth=1, linestyle="--")
    ax3.plot(x_raw, res_left, linewidth=1.2)
    ax3.set_title("Residuals (interp_on_raw − RAW) — Left")
    ax3.set_xlabel("x [mm]"); ax3.set_ylabel("Δ [mm]")

    # Residuals right
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axhline(0, linewidth=1, linestyle="--")
    ax4.plot(x_raw, res_right, linewidth=1.2)
    ax4.set_title("Residuals (interp_on_raw − RAW) — Right")
    ax4.set_xlabel("x [mm]"); ax4.set_ylabel("Δ [mm]")

    # Δx histogram
    ax5 = fig.add_subplot(gs[2, :])
    ax5.hist(dx, bins=40, alpha=0.85, label=f"RAW Δx (mean={dx_mean:.3g}, std={dx_std:.3g})")
    if np.isfinite(dx_uni):
        ax5.axvline(dx_uni, linewidth=2, linestyle="--", label=f"Uniform Δx ≈ {dx_uni:.3g}")
    ax5.set_title(f"Native spacing Δx — min={dx_min:.3g}, max={dx_max:.3g}")
    ax5.set_xlabel("Δx [mm]"); ax5.set_ylabel("count"); ax5.legend()

    fig.suptitle(
        f"Tow {tow} — Interpolated vs Non-Interpolated (normalize={per_edge_normalize}, "
        f"Δx_native≈{np.median(dx):.4g} mm, Δx_uniform≈{dx_uni:.4g} mm, N={len(x_uni)})",
        y=0.995
    )
    plt.show()

    # 5) Print metrics
    print(f"[Tow {tow}] MSE Left : {mse_left:.6g}")
    print(f"[Tow {tow}] MSE Right: {mse_right:.6g}")
    print(f"[Tow {tow}] RAW Δx stats — mean={dx_mean:.6g}, std={dx_std:.6g}, min={dx_min:.6g}, max={dx_max:.6g}")
    if np.isfinite(dx_uni):
        print(f"[Tow {tow}] Uniform Δx ≈ {dx_uni:.6g}  |  Target steps: {len(x_uni)}")

if __name__ == "__main__":
    # Example usage: auto-match native density (no manual N)
    compare_interpolation_for_tow(
        tow=3,
        target_steps=None,   # <-- auto
        per_edge_normalize=True,
        scale=1.0            # 1.0=match; >1 densify slightly
    )
