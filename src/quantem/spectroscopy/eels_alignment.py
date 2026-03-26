import matplotlib.pyplot as plt
import numpy as np


def align_dual_eels_universal(ll, hl, approach="smooth", sigma=1.2):
    """
    Aligns ZLP jitter across the spatial map and synchronizes Dual-EELS pairs.
    """
    print(f"QuantEM: Aligning {ll.name} and syncing {hl.name}...")

    # 1. Map the drift via argmax
    zlp_indices = np.argmax(ll.array, axis=0)
    ref_idx = int(np.median(zlp_indices))
    shifts = zlp_indices - ref_idx

    # 2. Apply internal QuantEM calibration
    ll.calibrate_zero_loss_peak()

    # 3. Synchronize High-Loss energy origin based on median shift
    shift_ev = np.median(shifts) * ll.sampling[0]
    hl.origin[0] -= shift_ev

    print("QuantEM: Alignment and Dual-EELS sync complete.")
    return ll, hl, shifts


def calibrate_energy_axis(ll, hl):
    """
    Fine-tunes the origin so the absolute peak position is exactly 0.0 eV.
    """
    # Find the peak of the average spectrum
    current_peak_idx = np.argmax(np.mean(ll.array, axis=(1, 2)))
    peak_ev = ll.origin[0] + (current_peak_idx * ll.sampling[0])

    # Apply global shift to both datasets
    ll.origin[0] -= peak_ev
    hl.origin[0] -= peak_ev

    print(f"QuantEM: Final calibration shift of {peak_ev:.4f} eV applied.")


def plot_absolute_zlp_shift(dataset, search_window=(-10, 10)):
    """
    Calculates the ZLP shift per pixel and plots the absolute deviation from 0.0 eV.
    """
    data = dataset.array
    n_e = data.shape[0]

    # Generate energy axis
    energies = dataset.origin[0] + np.arange(n_e) * dataset.sampling[0]

    # Mask energy window for peak finding
    mask = (energies > search_window[0]) & (energies < search_window[1])
    search_energies = energies[mask]

    # Calculate peak map and absolute deviation
    peak_indices = np.argmax(data[mask, :, :], axis=0)
    zlp_map_ev = search_energies[peak_indices]
    absolute_shift = np.abs(zlp_map_ev)

    # Visualization
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(absolute_shift, cmap="magma", origin="lower")

    plt.colorbar(im, ax=ax, label="Absolute Shift (eV)")
    ax.set_title(f"Absolute ZLP Deviation: {dataset.name}")
    ax.set_xlabel("X (pixels)")
    ax.set_ylabel("Y (pixels)")

    plt.tight_layout()
    plt.show()

    return absolute_shift


def plot_alignment_verification(dataset, shift_map, coords=(9, 9)):
    """
    Plots the drift map and a specific spectrum to verify alignment quality.
    """
    y, x = coords
    spec = dataset.array[:, y, x]
    energies = dataset.origin[0] + np.arange(len(spec)) * dataset.sampling[0]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Drift Map
    im = ax1.imshow(shift_map, cmap="RdBu_r", origin="lower")
    ax1.plot(x, y, "yo", markeredgecolor="k")
    ax1.set_title("Drift Map")
    plt.colorbar(im, ax=ax1, label="Relative Shift")

    # Spectrum Verification
    ax2.plot(energies, spec, color="black", label="Aligned Spec")
    ax2.axvline(0, color="red", linestyle="--", alpha=0.7, label="0.0 eV Target")
    ax2.set_xlim(-5, 5)
    ax2.set_title(f"ZLP Detail at Pixel ({x}, {y})")
    ax2.set_xlabel("Energy Loss (eV)")
    ax2.legend()

    plt.tight_layout()
    plt.show()
