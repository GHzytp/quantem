"""
QuantEM EELS Thickness Module
=============================
Tools for calculating specimen thickness from Low-Loss EELS using the Log-Ratio method.
t/λ = ln(I_total / I_ZLP)
"""

import matplotlib.pyplot as plt
import numpy as np


def visualize_thickness_windows(dataset, zlp_window=(-3.0, 3.0), total_window=(-3.0, 75.0)):
    """
    Visualizes integration windows for I0 (ZLP) and It (Total).
    Returns a configuration dictionary for the calculation step.
    """
    # 1. Extract Energy and Mean Spectrum
    data = dataset.array
    mean_spec = np.mean(data, axis=(1, 2))

    # Use built-in energy axis if available, else generate from metadata
    if hasattr(dataset, "energy_axis"):
        energy = dataset.energy_axis
    else:
        energy = dataset.origin[0] + np.arange(dataset.shape[0]) * dataset.sampling[0]

    # 2. Find indices for the windows
    zlp_idx = (
        np.argmin(np.abs(energy - zlp_window[0])),
        np.argmin(np.abs(energy - zlp_window[1])),
    )
    tot_idx = (
        np.argmin(np.abs(energy - total_window[0])),
        np.argmin(np.abs(energy - total_window[1])),
    )

    # 3. Create the Visualization
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(energy, mean_spec, "k-", lw=1.5, label="Mean Spectrum", zorder=5)

    # Highlight Windows
    z_mask = (energy >= zlp_window[0]) & (energy <= zlp_window[1])
    t_mask = (energy >= total_window[0]) & (energy <= total_window[1])

    ax.fill_between(
        energy[z_mask], 0, mean_spec[z_mask], color="red", alpha=0.3, label="$I_0$ (ZLP)"
    )
    ax.fill_between(
        energy[t_mask], 0, mean_spec[t_mask], color="blue", alpha=0.1, label="$I_t$ (Total)"
    )

    ax.axvline(0, color="green", lw=1.5, ls=":", label="0 eV")
    ax.set_title(f"QuantEM: Integration Windows ({dataset.name})", fontweight="bold")
    ax.set_xlabel("Energy Loss (eV)")
    ax.set_ylabel("Intensity (counts)")
    ax.set_xlim(energy[0], total_window[1] + 20)
    ax.legend()

    plt.tight_layout()
    plt.show()

    return {
        "zlp_idx": zlp_idx,
        "total_idx": tot_idx,
        "zlp_val": zlp_window,
        "total_val": total_window,
    }


def calculate_thickness_log_ratio(dataset, window_params, plot=True):
    """
    Calculates the relative thickness map (t/lambda) using the Log-Ratio method.
    """
    data = dataset.array
    z_start, z_end = window_params["zlp_idx"]
    t_start, t_end = window_params["total_idx"]

    print(f"QuantEM: Calculating thickness for {dataset.name}...")

    # 1. Vectorized Integration
    I_zlp = np.sum(data[z_start : z_end + 1, :, :], axis=0)
    I_total = np.sum(data[t_start : t_end + 1, :, :], axis=0)

    # 2. Log-Ratio Calculation (with epsilon to avoid log(0))
    epsilon = 1e-10
    t_over_lambda = np.log((I_total + epsilon) / (I_zlp + epsilon))

    # 3. Data Cleaning
    t_over_lambda = np.nan_to_num(t_over_lambda, nan=0.0, posinf=0.0, neginf=0.0)
    t_over_lambda = np.clip(t_over_lambda, 0, 4.0)

    if plot:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle(f"Thickness Analysis: {dataset.name}", fontsize=14)

        im = ax1.imshow(t_over_lambda, cmap="viridis", origin="lower")
        ax1.set_title(r"Relative Thickness Map ($t/\lambda$)")
        plt.colorbar(im, ax=ax1, label=r"$t/\lambda$")

        ax2.hist(t_over_lambda.flatten(), bins=50, color="steelblue", alpha=0.7, ec="k")
        ax2.axvline(
            np.mean(t_over_lambda),
            color="red",
            ls="--",
            label=f"Mean: {np.mean(t_over_lambda):.2f}",
        )
        ax2.set_title("Thickness Distribution")
        ax2.set_xlabel(r"$t/\lambda$")
        ax2.legend()

        plt.tight_layout()
        plt.show()

    return t_over_lambda


def interpret_thickness_quality(t_over_lambda, dataset=None):
    """
    Performs a scientific quality assessment on the calculated t/lambda map.
    """
    name = dataset.name if dataset else "Dataset"

    # Classification Masks
    vacuum = t_over_lambda < 0.3
    thin = (t_over_lambda >= 0.3) & (t_over_lambda < 1.0)
    medium = (t_over_lambda >= 1.0) & (t_over_lambda < 2.0)
    thick = t_over_lambda >= 2.0

    print(f"\n{'=' * 20} QUANTEM INTERPRETATION: {name} {'=' * 20}")
    for label, mask in [
        ("Vacuum (<0.3)", vacuum),
        ("Thin (0.3-1.0)", thin),
        ("Medium (1.0-2.0)", medium),
        ("Thick (>2.0)", thick),
    ]:
        pct = 100 * np.sum(mask) / t_over_lambda.size
        print(f"  {label:20}: {pct:5.1f}%")

    # Plotting Classification
    classified = np.zeros_like(t_over_lambda)
    classified[thin] = 1
    classified[medium] = 2
    classified[thick] = 3

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    im1 = ax1.imshow(classified, cmap="RdYlGn_r", origin="lower")
    ax1.set_title("Region Classification")
    cbar = plt.colorbar(im1, ax=ax1, ticks=[0, 1, 2, 3])
    cbar.ax.set_yticklabels(["Vacuum", "Thin", "Medium", "Thick"])

    t_masked = np.copy(t_over_lambda)
    t_masked[vacuum] = np.nan
    im2 = ax2.imshow(t_masked, cmap="viridis", origin="lower")
    ax2.set_title("Sample-Only Thickness")
    plt.colorbar(im2, ax=ax2, label=r"$t/\lambda$")

    plt.tight_layout()
    plt.show()


def plot_absolute_thickness(t_lambda_map, mfp_nm, dataset=None):
    """
    Converts relative thickness to nanometers and visualizes the absolute map.
    """
    thickness_nm = t_lambda_map * mfp_nm
    name = dataset.name if dataset else "Sample"

    # Mask vacuum for better visualization contrast
    display_map = np.copy(thickness_nm)
    display_map[t_lambda_map < 0.1] = np.nan

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"Physical Analysis: {name}", fontsize=14)

    im = ax1.imshow(display_map, cmap="magma", origin="lower")
    ax1.set_title("Absolute Thickness (nm)")
    plt.colorbar(im, ax=ax1, label="nm")

    valid_data = thickness_nm[t_lambda_map >= 0.1].flatten()
    ax2.hist(valid_data, bins=50, color="firebrick", alpha=0.7, ec="k")
    ax2.axvline(
        np.nanmean(display_map),
        color="blue",
        ls="--",
        label=f"Mean: {np.nanmean(display_map):.1f} nm",
    )
    ax2.set_title("Physical Distribution")
    ax2.set_xlabel("Thickness (nm)")
    ax2.legend()

    plt.tight_layout()
    plt.show()

    print(
        f"\nQuantEM Absolute Report:\n  Mean: {np.nanmean(display_map):.2f} nm\n  MFP:  {mfp_nm:.2f} nm"
    )
    return thickness_nm
