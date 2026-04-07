from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import interp1d
from scipy.ndimage import median_filter
from scipy.optimize import curve_fit

from quantem.spectroscopy.dataset3dspectroscopy import Dataset3dspectroscopy


class Dataset3deels(Dataset3dspectroscopy):
    """An EELS dataset class that inherits from Dataset3dspectroscopy.

    This class represents a scanning transmission electron microscopy (STEM) dataset,
    where the data consists of a 3D array with dimensions (energy, scan_y, scan_x).
    The first dimension represents the energy, while the latter
    two dimensions represent real space sampling.

    """

    element_info = None
    element_info_path = "eels_binding_energies.json"
    dataset_type = "EELS"

    def __init__(
        self,
        array: NDArray | Any,
        name: str,
        origin: NDArray | tuple | list | float | int,
        sampling: NDArray | tuple | list | float | int,
        units: list[str] | tuple | list,
        signal_units: str = "arb. units",
        _token: object | None = None,
    ):
        """Initialize a 3D EELS dataset.

        Parameters
        ----------
        array : NDArray | Any
            The underlying 3D array data
        name : str
            A descriptive name for the dataset
        origin : NDArray | tuple | list | float | int
            The origin coordinates for each dimension
        sampling : NDArray | tuple | list | float | int
            The sampling rate/spacing for each dimension
        units : list[str] | tuple | list
            Units for each dimension
        signal_units : str, optional
            Units for the array values, by default "arb. units"
        _token : object | None, optional
            Token to prevent direct instantiation, by default None
        """
        super().__init__(
            array=array,
            name=name,
            origin=origin,
            sampling=sampling,
            units=units,
            signal_units=signal_units,
            _token=_token,
        )
        self._virtual_images = {}
        self.dataset_type = "eels"

    def calculate_background_iterative(self, spectrum):
        """
        Subtract background typical for EELS using iterative Gaussian fitting.
        This method isolates the continuum background from the low-loss region.

        WARNING: Only use with EELS data! Will remove peaks if used with EDS.

        Parameters
        ----------
        spectrum : ndarray
            1D EELS spectrum
        energy_axis : ndarray
            Energy axis corresponding to spectrum

        Returns
        -------
        ndarray
            Background-subtracted spectrum
        """

        from scipy.ndimage import gaussian_filter
        from scipy.stats import norm

        # Smooth for better fitting
        spec_smooth = gaussian_filter(spectrum, sigma=1.0)
        pixel_vals = spec_smooth.copy()

        # Iteratively fit Gaussian to low-intensity values (the continuum)
        # Remove outliers (edge peaks) iteratively
        num_iterations = 10
        cutoff = 3  # +/- 3 sigma

        for _ in range(num_iterations):
            mu, std = norm.fit(pixel_vals)
            if std == 0:
                break
            # Keep only values within +/- 3 sigma (removes edge contributions)
            lower = mu - cutoff * std
            upper = mu + cutoff * std
            pixel_vals = pixel_vals[(pixel_vals >= lower) & (pixel_vals <= upper)]

        # Subtract the estimated background level
        background_fit = mu

        return background_fit

    def powerlaw_backgroundfit_eels(self, spectrum, energy_range, target_edge, window_size):
        """
        Using a window of the energy axis preceding the target edge, fit a power law function to use for background subtraction.
        The input window size should be 10-30% of the target edge energy.
        """

        dE = float(self.sampling[0])
        E0 = float(self.origin[0]) if hasattr(self, "origin") else 0.0
        energy_axis = E0 + dE * np.arange(self.shape[0])

        if energy_range is not None:
            energy_range[0] = np.maximum(energy_range[0], energy_axis[0])
            energy_range[1] = np.minimum(energy_range[1], energy_axis[-1])

            indices = np.where(
                (energy_axis >= energy_range[0]) & (energy_axis <= energy_range[1])
            )[0]
            energy_axis = energy_axis[indices]
        else:
            indices = np.arange(self.shape[0])

        # Check that input window size is between 10% and 30%

        if window_size < 10 or window_size > 30:
            raise ValueError("Invalid window size. Please input a value of between 10 and 30.")

        # Check that the target edge is within the energy range of the spectrum
        # and that a pre-edge region of size at least 10% of the target edge, ending 5 eV before the target edge
        # exists for pre-edge fitting.

        if target_edge < energy_axis[0] or target_edge > energy_axis[-1]:
            raise ValueError("Target edge is outside of energy range.")
        elif ((target_edge - 5) - target_edge * (window_size / 100)) < energy_axis[0]:
            raise ValueError(
                "Insufficient pre-edge background fitting region for this target edge and window size within given energy range."
            )

        # Fit power law function to spectrum within window region of the energy exis

        window_minE = (target_edge - 5) - target_edge * (window_size / 100)
        window_maxE = target_edge - 5

        window_indices = np.where((energy_axis >= window_minE) & (energy_axis <= window_maxE))[0]

        window_E = energy_axis[window_indices]
        window_I = spectrum[window_indices]

        def powerlaw_function(E, A, r):
            return A * (E ** (-r))

        popt, _ = curve_fit(powerlaw_function, window_E, window_I, maxfev=2000)
        background_fit = powerlaw_function(energy_axis, popt[0], popt[1])

        # Plot the region of the spectrum between user-specified energy range, overlaid with the background fit curve, with background estimation
        # window boundaries indicated

        fig, ax = plt.subplots()
        ax.plot(energy_axis, spectrum, label="spectrum", color="b")
        ax.plot(energy_axis, background_fit, label="background", color="r")
        ax.vlines(
            x=[window_minE, window_maxE],
            ymin=0,
            ymax=np.max(spectrum),
            label="window limits",
            color="k",
            linestyle="dashed",
        )
        ax.legend()

        return background_fit

    def smooth_eels_rollingaverage(
        self, roi=None, energy_range=None, ignore_range=None, mask=None, kernel_size=10
    ):
        dE = float(self.sampling[0])
        E0 = float(self.origin[0]) if hasattr(self, "origin") else 0.0
        energy_axis = E0 + dE * np.arange(self.shape[0])

        if energy_range is not None:
            energy_range[0] = np.maximum(energy_range[0], energy_axis[0])
            energy_range[1] = np.minimum(energy_range[1], energy_axis[-1])

            indices = np.where(
                (energy_axis >= energy_range[0]) & (energy_axis <= energy_range[1])
            )[0]
            energy_axis = energy_axis[indices]
        else:
            indices = np.arange(self.shape[0])

        array3d_subrange = self.array[indices, :, :]

        kernel = np.ones(kernel_size) / kernel_size

        # For each probe position, convolve spectral data with smoothing kernel

        array3d_smoothed = np.zeros(array3d_subrange.shape)

        for kk in range(array3d_subrange.shape[1]):
            for ll in range(array3d_subrange.shape[2]):
                probe_spectrum = self.array[:, kk, ll]
                spectrum_smoothed = np.convolve(probe_spectrum, kernel, mode="same")
                array3d_smoothed[:, kk, ll] = spectrum_smoothed

        smoothed_data3d = Dataset3deels.from_array(
            array=array3d_smoothed,
            sampling=self.sampling,
            origin=energy_axis[0],
            units=self.units,
        )

        # Plot raw and smoothed mean spectra on the same set of axes

        mean_spectrum_raw = self.calculate_mean_spectrum(roi, energy_range, ignore_range, mask)
        mean_spectrum_smoothed = smoothed_data3d.calculate_mean_spectrum(
            roi, energy_range, ignore_range, mask
        )

        fig, ax = plt.subplots()
        ax.plot(energy_axis, mean_spectrum_raw, label="raw spectrum", color="b")
        ax.plot(energy_axis, mean_spectrum_smoothed, label="kernel-smoothed spectrum", color="r")
        ax.legend()

        return smoothed_data3d

    def calibrate_zero_loss_peak(self, center_guess=None, search_window=10):
        """
        Calibrate the energy axis by centering the zero loss peak at 0 eV.
        Finds the ZLP at every pixel, fits a 2D plane to the ZLP positions,
        and shifts each spectrum individually so the ZLP sits at 0, while aligning
        all ZLPs to the same channel index, allowing a single origin to correctly
        calibrate the entire dataset.

        Parameters
        ----------
        center_guess : float or None
            Expected energy position of the ZLP in eV. If None, uses the
            tallest peak in each spectrum as the ZLP. If provided, searches
            for the tallest peak within the search window around that energy.
        search_window : int
            Number of channels to search on either side of center_guess.
            Only used when center_guess is not None. Default is 10.

        Returns
        -------
        Dataset3deels
            New dataset with corrected energy calibration.
        """

        n_energy, n_y, n_x = self.array.shape

        dE = float(self.sampling[0])
        E0 = float(self.origin[0])
        energy_axis = E0 + np.arange(n_energy) * dE

        # --- Build ZLP position map ---
        # For every pixel, find the energy where the ZLP sits.
        # A median filter is applied to each spectrum first to remove
        # hot pixels (cosmic rays, detector glitches) that could be
        # brighter than the ZLP and fool the peak finder.
        # If center_guess is provided, only look within a window
        # of search_window channels around that energy.
        # If center_guess is None, just find the tallest peak.

        zlp_map = np.zeros((n_y, n_x))

        if center_guess is not None:
            guess_index = int(round((center_guess - E0) / dE))
            lo = max(guess_index - search_window, 0)
            hi = min(guess_index + search_window + 1, n_energy)

        for iy in range(n_y):
            for ix in range(n_x):
                spectrum = median_filter(self.array[:, iy, ix], size=3)

                if center_guess is None:
                    peak_index = np.argmax(spectrum)
                else:
                    peak_index = lo + np.argmax(spectrum[lo:hi])

                zlp_map[iy, ix] = E0 + peak_index * dE

        # --- Fit a 2D plane to the ZLP map ---
        # The plane equation is: zlp_energy(y, x) = a*y + b*x + c
        # This smooths out noisy per-pixel ZLP measurements by assuming
        # the drift varies linearly across the scan area.

        y_coords, x_coords = np.meshgrid(np.arange(n_y), np.arange(n_x), indexing="ij")
        y_flat = y_coords.ravel()
        x_flat = x_coords.ravel()
        z_flat = zlp_map.ravel()

        A = np.column_stack([y_flat, x_flat, np.ones(len(y_flat))])
        coeffs, _, _, _ = np.linalg.lstsq(A, z_flat, rcond=None)
        a, b, c = coeffs

        zlp_plane = a * y_coords + b * x_coords + c

        # --- Shift each spectrum so the ZLP lands at 0 eV ---
        # For each pixel, subtract its plane-predicted ZLP position from
        # the energy axis, then interpolate the spectrum back onto the
        # original energy grid. This physically moves the data so all
        # ZLPs align at the same channel index.

        corrected_array = np.zeros_like(self.array)

        for iy in range(n_y):
            for ix in range(n_x):
                shift = zlp_plane[iy, ix]
                shifted_energy = energy_axis - shift
                interpolator = interp1d(
                    shifted_energy,
                    self.array[:, iy, ix],
                    kind="linear",
                    bounds_error=False,
                    fill_value=0.0,
                )
                corrected_array[:, iy, ix] = interpolator(energy_axis)

        return Dataset3deels.from_array(
            array=corrected_array,
            name=self.name,
            sampling=self.sampling,
            origin=self.origin,
            units=self.units,
        )

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
