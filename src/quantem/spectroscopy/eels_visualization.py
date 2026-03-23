import matplotlib.pyplot as plt
import numpy as np
from ipywidgets import Checkbox, IntSlider, interact
from scipy.stats import norm


def plot_dual_eels_picker(ll, hl, coords=(9, 9), title="QuantEM: Dual-EELS Analysis"):
    """
    Interactive picker for side-by-side Low-Loss and High-Loss EELS analysis.
    """
    # 1. Pre-calculate sum images for spatial maps
    sum_ll = np.sum(ll.array, axis=0)
    sum_hl = np.sum(hl.array, axis=0)

    # 2. Generate energy axes using dataset metadata
    energy_ll = ll.origin[0] + np.arange(ll.shape[0]) * ll.sampling[0]
    energy_hl = hl.origin[0] + np.arange(hl.shape[0]) * hl.sampling[0]

    def _update_plot(x, y, log_scale):
        fig, axes = plt.subplots(2, 2, figsize=(14, 9))
        fig.suptitle(title, fontsize=16)

        # --- LOW LOSS ROW (Top) ---
        im_ll = axes[0, 0].imshow(sum_ll, cmap="viridis", origin="lower")
        axes[0, 0].plot(x, y, "r+", markersize=12, markeredgewidth=2)
        axes[0, 0].set_title("Low-Loss Map (Integrated)")
        fig.colorbar(im_ll, ax=axes[0, 0], label="Counts")

        axes[0, 1].plot(energy_ll, ll.array[:, y, x], color="tab:blue", lw=1.5)
        axes[0, 1].set_title(f"LL Spectrum at ({x}, {y})")
        axes[0, 1].set_ylabel("Intensity")
        if log_scale:
            axes[0, 1].set_yscale("log")

        # --- HIGH LOSS ROW (Bottom) ---
        im_hl = axes[1, 0].imshow(sum_hl, cmap="magma", origin="lower")
        axes[1, 0].plot(x, y, "r+", markersize=12, markeredgewidth=2)
        axes[1, 0].set_title("High-Loss Map (Integrated)")
        fig.colorbar(im_hl, ax=axes[1, 0], label="Counts")

        axes[1, 1].plot(energy_hl, hl.array[:, y, x], color="tab:red", lw=1.5)
        axes[1, 1].set_title(f"HL Spectrum at ({x}, {y})")
        axes[1, 1].set_xlabel("Energy Loss (eV)")
        axes[1, 1].set_ylabel("Intensity")
        if log_scale:
            axes[1, 1].set_yscale("log")

        plt.tight_layout()
        plt.show()

    # Standardized sliders with continuous_update=False for performance
    interact(
        _update_plot,
        x=IntSlider(min=0, max=ll.shape[2] - 1, step=1, value=coords[1], continuous_update=False),
        y=IntSlider(min=0, max=ll.shape[1] - 1, step=1, value=coords[0], continuous_update=False),
        log_scale=Checkbox(value=False, description="Log Y-Axis"),
    )


def plot_quantem_diagnostic(dataset, zlp_window=5.0, title_suffix=""):
    """
    QuantEM Diagnostic Dashboard: Visualizes mean spectra, spatial variation, and ZLP accuracy.
    """
    data = dataset.array
    energy = dataset.origin[0] + np.arange(data.shape[0]) * dataset.sampling[0]

    mean_spec = np.mean(data, axis=(1, 2))
    zlp_idx = np.argmax(mean_spec)
    zlp_pos = energy[zlp_idx]
    sum_img = np.sum(data, axis=0)

    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    fig.suptitle(f"QuantEM Diagnostic: {dataset.name} {title_suffix}", fontsize=16)

    # 1. Mean Spectrum with Alignment Targets
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(energy, mean_spec, color="black", label="Mean Spectrum")
    ax1.axvline(zlp_pos, color="red", ls="--", alpha=0.6, label=f"Peak: {zlp_pos:.2f} eV")
    ax1.axvline(0, color="green", ls=":", lw=2, label="Target (0 eV)")
    ax1.set_title("Global Average Spectrum")
    ax1.set_xlabel("Energy Loss (eV)")
    ax1.legend()

    # 2. Spatial Variability (Sampled 3x3 Grid)
    ax2 = fig.add_subplot(gs[0, 1])
    yy = np.linspace(0, data.shape[1] - 1, 3, dtype=int)
    xx = np.linspace(0, data.shape[2] - 1, 3, dtype=int)
    for y in yy:
        for x in xx:
            ax2.plot(energy, data[:, y, x], alpha=0.4, lw=1)
    ax2.set_title("Spatial Variation (Sampled Pixels)")
    ax2.set_xlabel("Energy Loss (eV)")

    # 3. Integrated Intensity Map
    ax3 = fig.add_subplot(gs[1, 0])
    im = ax3.imshow(sum_img, cmap="viridis", origin="lower")
    fig.colorbar(im, ax=ax3, label="Total Counts")
    ax3.set_title("Summed Intensity Map")
    ax3.set_xlabel("X (pixels)")
    ax3.set_ylabel("Y (pixels)")

    # 4. ZLP Zoom-in Detail
    ax4 = fig.add_subplot(gs[1, 1])
    mask = (energy > zlp_pos - zlp_window) & (energy < zlp_pos + zlp_window)
    ax4.plot(energy[mask], mean_spec[mask], color="blue", lw=2)
    ax4.axvline(0, color="green", ls=":", lw=2, label="0 eV Target")
    ax4.set_title(f"ZLP Alignment Detail (±{zlp_window} eV)")
    ax4.set_xlabel("Energy Loss (eV)")
    ax4.legend()

    plt.show()


def plot_zlp_drift_diagnostics(dataset, title="ZLP Drift Analysis"):
    """
    QuantEM Diagnostic: Maps the ZLP position and calculates the drift distribution.
    """
    data = dataset.array
    energy = dataset.origin[0] + np.arange(data.shape[0]) * dataset.sampling[0]

    # 1. Mask and find peak per pixel (Vectorized for speed)
    search_mask = (energy > -2.0) & (energy < 2.0)
    search_energies = energy[search_mask]
    peak_indices = np.argmax(data[search_mask, :, :], axis=0)
    zlp_map = search_energies[peak_indices]

    # 2. Setup Figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle(f"QuantEM: {dataset.name} - {title}", fontsize=16)

    # Plot A: Spatial Map of ZLP Shifts
    im = ax1.imshow(zlp_map, cmap="RdYlBu_r", origin="lower")
    ax1.set_title("Spatial Map of ZLP Positions")
    ax1.set_xlabel("X (pixels)")
    ax1.set_ylabel("Y (pixels)")
    cbar = fig.colorbar(im, ax=ax1)
    cbar.set_label("Energy Shift (eV)", rotation=270, labelpad=15)

    # Plot B: Histogram + Gaussian Fit
    flat_pos = zlp_map.flatten()
    mu, std = norm.fit(flat_pos)

    ax2.hist(flat_pos, bins=30, density=True, alpha=0.6, color="skyblue", ec="white")

    # Fit line display
    x_range = np.linspace(np.min(flat_pos), np.max(flat_pos), 100)
    ax2.plot(
        x_range,
        norm.pdf(x_range, mu, std),
        color="darkred",
        lw=2.5,
        label=f"Fit: μ={mu:.3f} eV\nσ={std:.3f} eV",
    )

    ax2.set_title("ZLP Drift Distribution")
    ax2.set_xlabel("Energy (eV)")
    ax2.set_ylabel("Density")
    ax2.legend()
    ax2.grid(True, alpha=0.15)

    plt.tight_layout()
    plt.show()
