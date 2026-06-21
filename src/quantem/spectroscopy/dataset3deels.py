from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import interp1d
from scipy.ndimage import median_filter
from scipy.optimize import curve_fit
from scipy.stats import norm

from quantem.core.visualization import show_2d
from quantem.spectroscopy.dataset3dspectroscopy import Dataset3dspectroscopy


class Dataset3deels(Dataset3dspectroscopy):
    """An EELS dataset class that inherits from Dataset3dspectroscopy.

    This class represents a scanning transmission electron microscopy (STEM) dataset,
    where the data consists of a 3D array with dimensions (scan_row, scan_col, energy).
    The first two dimensions represent real space sampling, while the last dimension
    represents the energy axis.

    """

    element_info = None
    element_info_path = "eels_edges.csv"
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

    # ========== NEW METHOD: Background subtraction for limited pre-edge data ==========

    def subtract_background_limited_preedge(
        self,
        target_edge,
        pre_edge_range=None,
        method="polynomial",
        polynomial_degree=2,
        show=True,
        return_dataset=True,
    ):
        """
        Background subtraction optimized for limited pre-edge data.

        This method bypasses the 10-30% window_size constraint in the standard
        subtract_background() method, allowing background fitting when only a
        small pre-edge region is available (common in high-loss only acquisitions).

        Parameters
        ----------
        target_edge : float
            Energy of the edge onset (eV)
            Examples: 285 for C K-edge, 532 for O K-edge, 284 for C K-edge
        pre_edge_range : tuple of float, optional
            Explicit (start, end) energies in eV for pre-edge fitting window.
            If None, automatically uses all available data before edge.
            Example: (519, 527) for O K-edge when data starts at 518 eV
        method : str, optional
            Background fitting method:
            - 'polynomial': Polynomial fit (default, most stable for short ranges)
            - 'linear': Linear fit (equivalent to polynomial degree=1)
            - 'powerlaw': Power-law A*E^(-r) (needs longer pre-edge, may fail)
        polynomial_degree : int, optional
            Degree of polynomial (1=linear, 2=quadratic, 3=cubic). Default is 2.
            Only used when method='polynomial'.
        show : bool, optional
            Display before/after visualization. Default True.
        return_dataset : bool, optional
            If True, return Dataset3deels. If False, return numpy array. Default True.

        Returns
        -------
        Dataset3deels or ndarray
            Background-subtracted data

        Raises
        ------
        ValueError
            If pre-edge region is insufficient or target_edge is out of range
        RuntimeError
            If fitting fails (typically with powerlaw on limited data)

        Notes
        -----
        **When to use this method:**
        - Data starts close to the edge (limited pre-edge region)
        - Standard subtract_background() fails with window_size error
        - High-loss only acquisitions (no low-loss data)
        - Cropped energy ranges

        **Recommended methods by pre-edge size:**
        - < 10 eV: method='linear' (most stable)
        - 10-20 eV: method='polynomial', degree=2
        - > 20 eV: method='polynomial', degree=2-3, or 'powerlaw'

        **Comparison to GMS background subtraction:**
        This mimics the GMS "Fit Background" function but without the
        window percentage constraint, using direct energy range specification.

        Examples
        --------
        >>> # O K-edge at 532 eV, data starts at 518 eV (only 14 eV pre-edge)
        >>> eels_sub = eels_hl.subtract_background_limited_preedge(
        ...     target_edge=532,
        ...     method='polynomial',
        ...     polynomial_degree=2
        ... )

        >>> # Specify exact pre-edge window
        >>> eels_sub = eels_hl.subtract_background_limited_preedge(
        ...     target_edge=532,
        ...     pre_edge_range=(519, 527),  # 8 eV window
        ...     method='linear',
        ...     show=True
        ... )

        >>> # C K-edge with enough pre-edge for power-law
        >>> eels_sub = eels_hl.subtract_background_limited_preedge(
        ...     target_edge=285,
        ...     pre_edge_range=(200, 280),  # 80 eV window
        ...     method='powerlaw'
        ... )

        See Also
        --------
        subtract_background : Standard method with window_size percentage
        powerlaw_backgroundfit_eels : Direct power-law fitting function
        """

        import warnings

        from scipy.optimize import curve_fit

        energy = self.energy_axis
        mean_spec = self.calculate_mean_spectrum()

        # ===== 1. Determine pre-edge fitting window =====
        if pre_edge_range is None:
            pre_edge_start = float(energy[0])
            pre_edge_end = float(target_edge - 5)
            print(f"Auto-detected pre-edge: {pre_edge_start:.1f} - {pre_edge_end:.1f} eV")
        else:
            pre_edge_start, pre_edge_end = float(pre_edge_range[0]), float(pre_edge_range[1])
            print(f"Using specified pre-edge: {pre_edge_start:.1f} - {pre_edge_end:.1f} eV")

        # ===== 2. Validate inputs =====
        if target_edge < energy[0] or target_edge > energy[-1]:
            raise ValueError(
                f"Target edge {target_edge} eV is outside data range "
                f"[{energy[0]:.1f}, {energy[-1]:.1f}] eV"
            )

        if pre_edge_start < energy[0]:
            raise ValueError(
                f"Pre-edge start {pre_edge_start:.1f} eV is before data start {energy[0]:.1f} eV"
            )

        if pre_edge_end >= target_edge:
            raise ValueError(
                f"Pre-edge end {pre_edge_end:.1f} eV must be before target edge {target_edge:.1f} eV"
            )

        available_preedge = pre_edge_end - pre_edge_start

        if available_preedge < 1:
            raise ValueError(
                f"Insufficient pre-edge region: only {available_preedge:.1f} eV available. "
                f"Need at least 1 eV for fitting."
            )

        # Warn if pre-edge is very limited
        if available_preedge < 10:
            warnings.warn(
                f"Limited pre-edge region ({available_preedge:.1f} eV). "
                f"Background fit may be unreliable. Consider method='linear' for stability.",
                UserWarning,
            )

        # ===== 3. Extract pre-edge data =====
        mask = (energy >= pre_edge_start) & (energy <= pre_edge_end)
        E_window = energy[mask]
        I_window = mean_spec[mask]

        n_points = len(E_window)
        print(f"Pre-edge region: {available_preedge:.1f} eV ({n_points} data points)")

        if n_points < 3:
            raise ValueError(
                f"Insufficient data points in pre-edge window: only {n_points} points. "
                f"Need at least 3 for fitting."
            )

        # ===== 4. Fit background using selected method =====
        if method == "linear" or (method == "polynomial" and polynomial_degree == 1):
            # Linear fit: y = m*x + b
            coeffs = np.polyfit(E_window, I_window, deg=1)
            background = np.polyval(coeffs, energy)
            fit_info = f"Linear: y = {coeffs[0]:.2e}*E + {coeffs[1]:.2e}"

        elif method == "polynomial":
            # Polynomial fit
            if polynomial_degree > n_points - 1:
                warnings.warn(
                    f"Polynomial degree {polynomial_degree} too high for {n_points} points. "
                    f"Using degree {n_points - 1} instead.",
                    UserWarning,
                )
                polynomial_degree = n_points - 1

            coeffs = np.polyfit(E_window, I_window, deg=polynomial_degree)
            background = np.polyval(coeffs, energy)
            fit_info = f"Polynomial (degree {polynomial_degree})"

        elif method == "powerlaw":
            # Power-law fit: A * E^(-r)
            def powerlaw(E, A, r):
                return A * (E ** (-r))

            # Initial guess
            A0 = I_window[0] * (E_window[0] ** 3)
            r0 = 3.0

            try:
                popt, _ = curve_fit(
                    powerlaw,
                    E_window,
                    I_window,
                    p0=[A0, r0],
                    bounds=([0, 0], [np.inf, 10]),
                    maxfev=5000,
                )
                background = powerlaw(energy, popt[0], popt[1])
                fit_info = f"Power-law: A={popt[0]:.2e}, r={popt[1]:.2f}"
            except RuntimeError as e:
                raise RuntimeError(
                    f"Power-law fit failed to converge with {available_preedge:.1f} eV pre-edge. "
                    f"Try method='polynomial' or 'linear' instead. Error: {e}"
                )
        else:
            raise ValueError(
                f"Unknown method '{method}'. Choose 'linear', 'polynomial', or 'powerlaw'."
            )

        print(f"✓ Fit method: {fit_info}")

        # ===== 5. Subtract background from 3D data =====
        data_sub = np.maximum(self.array - background[None, None, :], 0)

        # ===== 6. Visualize if requested =====
        if show:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
            fig.suptitle(
                f"Background Subtraction: {self.name}\nEdge at {target_edge} eV",
                fontsize=14,
                fontweight="bold",
            )

            # Before subtraction
            ax1.plot(energy, mean_spec, "k-", lw=1.5, label="Raw spectrum")
            ax1.plot(energy, background, "r--", lw=2, label=f"Background ({fit_info})")
            ax1.axvspan(
                pre_edge_start,
                pre_edge_end,
                alpha=0.2,
                color="green",
                label=f"Fit region ({available_preedge:.1f} eV)",
            )
            ax1.axvline(target_edge, color="orange", ls=":", lw=2, label="Edge onset")
            ax1.set_xlabel("Energy (eV)", fontsize=12)
            ax1.set_ylabel("Intensity", fontsize=12)
            ax1.set_title("Before Background Subtraction")
            ax1.legend(fontsize=10)
            ax1.grid(True, alpha=0.3)

            # After subtraction
            subtracted_spec = mean_spec - background
            ax2.plot(energy, subtracted_spec, "b-", lw=1.5, label="Background-subtracted")
            ax2.axvline(target_edge, color="orange", ls=":", lw=2, label="Edge onset")
            ax2.axhline(0, color="gray", ls="--", alpha=0.5)
            ax2.set_xlabel("Energy (eV)", fontsize=12)
            ax2.set_ylabel("Intensity", fontsize=12)
            ax2.set_title("After Background Subtraction")
            ax2.legend(fontsize=10)
            ax2.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.show()

        # ===== 7. Return result =====
        if return_dataset:
            result = Dataset3deels.from_array(
                data_sub,
                sampling=self.sampling,
                origin=self.origin,
                units=self.units,
                name=f"{self.name} (background subtracted)",
            )
            print(f"✓ Created background-subtracted dataset: {result.shape}")
            return result
        else:
            return data_sub

    def powerlaw_backgroundfit_eels(self, spectrum, energy_range, target_edge, window_size):
        """
        Using a window of the energy axis preceding the target edge, fit a power law function to use for background subtraction.
        The input window size should be 10-30% of the target edge energy.
        """

        energy_axis = self.energy_axis

        if energy_range is not None:
            energy_range[0] = np.maximum(energy_range[0], energy_axis[0])
            energy_range[1] = np.minimum(energy_range[1], energy_axis[-1])

            indices = np.where(
                (energy_axis >= energy_range[0]) & (energy_axis <= energy_range[1])
            )[0]
            energy_axis = energy_axis[indices]
        else:
            indices = np.arange(self.shape[2])

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

    def smooth_eels_rollingaverage(self, roi=None, energy_range=None, mask=None, kernel_size=10):
        energy_axis = self.energy_axis

        if energy_range is not None:
            energy_range[0] = np.maximum(energy_range[0], energy_axis[0])
            energy_range[1] = np.minimum(energy_range[1], energy_axis[-1])

            indices = np.where(
                (energy_axis >= energy_range[0]) & (energy_axis <= energy_range[1])
            )[0]
            energy_axis = energy_axis[indices]
        else:
            indices = np.arange(self.shape[2])

        array3d_subrange = self.array[:, :, indices]

        kernel = np.ones(kernel_size) / kernel_size

        # For each probe position, convolve spectral data with smoothing kernel

        array3d_smoothed = np.zeros(array3d_subrange.shape)

        scan_row, scan_col, _n_energy = array3d_subrange.shape
        for i_row in range(scan_row):
            for i_col in range(scan_col):
                probe_spectrum = array3d_subrange[i_row, i_col, :]
                spectrum_smoothed = np.convolve(probe_spectrum, kernel, mode="same")
                array3d_smoothed[i_row, i_col, :] = spectrum_smoothed

        output_origin = np.array(self.origin, dtype=float, copy=True)
        output_origin[2] = energy_axis[0]
        smoothed_data3d = Dataset3deels.from_array(
            array=array3d_smoothed,
            sampling=self.sampling,
            origin=output_origin,
            units=self.units,
        )

        # Plot raw and smoothed mean spectra on the same set of axes

        mean_spectrum_raw = self.calculate_mean_spectrum(
            roi=roi,
            energy_range=energy_range,
            mask=mask,
        )
        mean_spectrum_smoothed = smoothed_data3d.calculate_mean_spectrum(
            roi=roi,
            energy_range=energy_range,
            mask=mask,
        )

        fig, ax = plt.subplots()
        ax.plot(energy_axis, mean_spectrum_raw, label="raw spectrum", color="b")
        ax.plot(energy_axis, mean_spectrum_smoothed, label="kernel-smoothed spectrum", color="r")
        ax.legend()

        return smoothed_data3d

    def measure_zlp_offset(
        self,
        zlp_guess_x=None,
        fit_window=0.8,
        fit_to_plane=False,
        median_filter_pixels=3,
        fit_zlp=True,
    ):
        """
        Measure ZLP offset at each pixel position by using a guess of ZLP posfitting each spectrum to a Gaussian
        """

        # Define Gaussian constraint to fit ZLP to
        def _gaussian_fit(x, A, mu, sigma):
            return A * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

        def _plane_fit_2d(M, a, b, c):
            row, col = M
            return (a * row) + (b * col) + c

        scan_row, scan_col, _n_energy = self.array.shape
        energy_axis = self.energy_axis

        # For each pixel, measure the zlp position by fitting a Gaussian to the measured zero-loss signal and taking its center as the zlp position.

        zlp_measured = np.zeros((scan_row, scan_col))

        for i_row in range(scan_row):
            for i_col in range(scan_col):
                # Apply median filter to discount hot pixels that might spuriously produce the maximum intensity of the spectrum
                if median_filter_pixels > 0:
                    spec_filt = median_filter(self.array[i_row, i_col, :], median_filter_pixels)
                else:
                    spec_filt = self.array[i_row, i_col, :]

                if fit_zlp:
                    # Use initial guess for ZLP to define window for Gaussian fitting. If zlp_guess_x=None (default) use the maximum value of the spectrum
                    if zlp_guess_x is not None:
                        zlp_crude_idx = int(np.argmin(np.abs(energy_axis - zlp_guess_x)))
                    else:
                        zlp_crude_idx = int(np.argmax(spec_filt))

                    mu0 = float(energy_axis[zlp_crude_idx])

                    lo = mu0 - fit_window
                    hi = mu0 + fit_window

                    x_mask = (energy_axis >= lo) & (energy_axis <= hi)

                    xw = energy_axis[x_mask]
                    yw = spec_filt[x_mask]

                    A0 = float(spec_filt[zlp_crude_idx])
                    sigma0 = fit_window / 2

                    p0 = (A0, mu0, sigma0)

                    bounds = (
                        (
                            0.0,
                            lo,
                            1e-12,
                        ),
                        (
                            np.inf,
                            hi,
                            np.inf,
                        ),
                    )

                    popt, _ = curve_fit(_gaussian_fit, xw, yw, p0=p0, bounds=bounds)

                    zlp_measured[i_row, i_col] = float(popt[1])
                else:
                    zlp_crude_idx = int(np.argmax(spec_filt))
                    zlp_measured[i_row, i_col] = float(energy_axis[zlp_crude_idx])

        if fit_to_plane:
            # Fit a 2D plane to the array of measured ZLPs
            row_data, col_data = np.meshgrid(
                np.arange(scan_row), np.arange(scan_col), indexing="ij"
            )

            coord_data_unpacked = np.vstack((row_data.ravel(), col_data.ravel()))
            ydata_unpacked = zlp_measured.ravel()

            popt, _ = curve_fit(_plane_fit_2d, coord_data_unpacked, ydata_unpacked)

            zlp_plane_1d = _plane_fit_2d(coord_data_unpacked, popt[0], popt[1], popt[2])
            zlp_plane_2d = zlp_plane_1d.reshape(scan_row, scan_col)

            show_2d(
                [zlp_measured, zlp_plane_2d],
                cmap="magma",
                title=["Measured ZLP (mean of Gaussian fit)", "ZLP plane fit"],
            )
            return zlp_plane_2d
        else:
            show_2d(
                [zlp_measured],
                cmap="magma",
                title=["Measured ZLP (mean of Gaussian fit)"],
            )
            return zlp_measured

    def apply_zlp_correction(
        self,
        zlp_guess_x=None,
        zlp_shifts_array=None,
        fit_window=0.8,
        measure_offset=True,
        fit_to_plane=True,
        fit_zlp=True,
        return_3d_dataset=True,
        return_shifts=False,
    ):
        # Default behavior is to automatically call measure_zlp_offset to generate an array of ZLP shifts for each scan position.
        # Alternatively, a 2D array matching the scan_row and scan_col dimensions of the 3D dataset can be supplied as the value of zlp_shifts_array to skip this step.
        # If measure_offset is False and no 2D ZLP shifts array is provided, a scalar input for zlp_guess_x can be used to shift the energy axis at every scan position by that amount.
        if measure_offset:
            zlp_array = self.measure_zlp_offset(
                zlp_guess_x=zlp_guess_x,
                fit_window=fit_window,
                fit_to_plane=fit_to_plane,
                fit_zlp=fit_zlp,
            )
        elif zlp_shifts_array is not None:
            zlp_array = np.asarray(zlp_shifts_array, dtype=float)
            if zlp_array.shape != self.array.shape[0:2]:
                raise ValueError(
                    "Dimensions of input array for ZLP shifts do not match scan_row and scan_col dimensions of 3D spectroscopy dataset."
                )
        elif zlp_guess_x is not None:
            zlp_array = np.ones(self.array.shape[0:2], dtype=float) * zlp_guess_x
        else:
            raise ValueError(
                "measure_offset was set to False and no input argument for ZLP shifts was provided."
            )

        zlp_array = np.asarray(zlp_array, dtype=float)
        if not np.all(np.isfinite(zlp_array)):
            raise ValueError("ZLP shifts must contain only finite values.")

        # Initialize 3D array to populate with spectra aligned along the energy axis
        corrected_array = np.empty(self.array.shape, dtype=np.result_type(self.array.dtype, float))

        scan_row, scan_col, n_energy = self.array.shape

        energy_axis = self.energy_axis
        if np.all((zlp_array >= 0) & (zlp_array <= n_energy - 1)) and (
            np.min(zlp_array) < energy_axis[0] or np.max(zlp_array) > energy_axis[-1]
        ):
            zlp_array = np.interp(zlp_array, np.arange(n_energy), energy_axis)

        # Apply sub-channel ZLP shifts using 1D linear interpolation along the energy axis.
        for i_row in range(scan_row):
            for i_col in range(scan_col):
                spec = self.array[i_row, i_col, :]
                corrected_array[i_row, i_col, :] = np.interp(
                    energy_axis + zlp_array[i_row, i_col],
                    energy_axis,
                    spec,
                    left=np.nan,
                    right=np.nan,
                )

        # Remove all planes along energy axis containing NaN, to equalize spectra lengths across all scan positions
        mask = np.isnan(corrected_array).any(axis=(0, 1))
        aligned_data_3d = corrected_array[:, :, ~mask]
        new_Eaxis = energy_axis[~mask]

        if aligned_data_3d.shape[2] == 0:
            raise ValueError(
                "ZLP shifts leave no shared energy range after alignment. "
                "Check that zlp_shifts_array is in energy units, not channel indices."
            )

        new_origin = float(new_Eaxis[0])

        # Calculate mean spectra before and after correction for plotting
        mean_spectrum_raw = self.array.mean(axis=(0, 1))
        mean_spectrum_corrected = aligned_data_3d.mean(axis=(0, 1))

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        ax1.plot(energy_axis, mean_spectrum_raw, label="Raw mean spectrum", color="r")
        ax2.plot(new_Eaxis, mean_spectrum_corrected, label="ZLP-corrected spectrum", color="b")
        ax1.set_xlabel("Energy (eV)")
        ax1.set_ylabel("Intensity")
        ax1.grid(True, alpha=0.1)
        ax1.legend()
        ax2.set_xlabel("Energy (eV)")
        ax2.set_ylabel("Intensity")
        ax2.grid(True, alpha=0.1)
        ax2.legend()

        fig.tight_layout()

        # <<<--- CHANGED: Modified return logic to optionally include shifts
        if return_3d_dataset:
            corrected_dataset = Dataset3deels.from_array(
                array=aligned_data_3d,
                name=self.name,
                sampling=self.sampling,
                origin=new_origin,
                units=self.units,
            )
            if return_shifts:
                return corrected_dataset, zlp_array  # Return both dataset and shifts
            else:
                return corrected_dataset  # Original behavior
        else:
            if return_shifts:
                return aligned_data_3d, zlp_array  # Return both array and shifts
            else:
                return aligned_data_3d  # Original behavior

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

        scan_row, scan_col, n_energy = self.array.shape

        energy_axis = np.asarray(self.energy_axis, dtype=float)

        # --- Build ZLP position map ---
        # For every pixel, find the energy where the ZLP sits.
        # A median filter is applied to each spectrum first to remove
        # hot pixels (cosmic rays, detector glitches) that could be
        # brighter than the ZLP and fool the peak finder.
        # If center_guess is provided, only look within a window
        # of search_window channels around that energy.
        # If center_guess is None, just find the tallest peak.

        zlp_map = np.zeros((scan_row, scan_col))

        if center_guess is not None:
            guess_index = int(np.argmin(np.abs(energy_axis - center_guess)))
            lo = max(guess_index - search_window, 0)
            hi = min(guess_index + search_window + 1, n_energy)

        for i_row in range(scan_row):
            for i_col in range(scan_col):
                spectrum = median_filter(self.array[i_row, i_col, :], size=3)

                if center_guess is None:
                    peak_index = np.argmax(spectrum)
                else:
                    peak_index = lo + np.argmax(spectrum[lo:hi])

                zlp_map[i_row, i_col] = energy_axis[peak_index]

        # --- Fit a 2D plane to the ZLP map ---
        # The plane equation is: zlp_energy(scan_row, scan_col) = a*row + b*col + c
        # This smooths out noisy per-pixel ZLP measurements by assuming
        # the drift varies linearly across the scan area.

        row_coords, col_coords = np.meshgrid(
            np.arange(scan_row), np.arange(scan_col), indexing="ij"
        )
        row_flat = row_coords.ravel()
        col_flat = col_coords.ravel()
        z_flat = zlp_map.ravel()

        A = np.column_stack([row_flat, col_flat, np.ones(len(row_flat))])
        coeffs, _, _, _ = np.linalg.lstsq(A, z_flat, rcond=None)
        a, b, c = coeffs

        zlp_plane = a * row_coords + b * col_coords + c

        # --- Shift each spectrum so the ZLP lands at 0 eV ---
        # For each pixel, subtract its plane-predicted ZLP position from
        # the energy axis, then interpolate the spectrum back onto the
        # original energy grid. This physically moves the data so all
        # ZLPs align at the same channel index.

        corrected_array = np.zeros_like(self.array)

        for i_row in range(scan_row):
            for i_col in range(scan_col):
                shift = zlp_plane[i_row, i_col]
                shifted_energy = energy_axis - shift
                interpolator = interp1d(
                    shifted_energy,
                    self.array[i_row, i_col, :],
                    kind="linear",
                    bounds_error=False,
                    fill_value=0.0,
                )
                corrected_array[i_row, i_col, :] = interpolator(energy_axis)

        return Dataset3deels.from_array(
            array=corrected_array,
            name=self.name,
            sampling=self.sampling,
            origin=self.origin,
            units=self.units,
        )

    def correct_zlp_shift(ll, hl):
        """
        Aligns ZLP jitter across the spatial map and synchronizes Dual-EELS pairs.
        """
        print(f"QuantEM: Aligning {ll.name} and syncing {hl.name}...")

        # 1. Map the drift via argmax
        zlp_indices = np.argmax(ll.array, axis=2)
        ref_idx = int(np.median(zlp_indices))
        shifts = zlp_indices - ref_idx

        # 2. Apply internal QuantEM calibration
        ll.calibrate_zero_loss_peak()

        # 3. Synchronize High-Loss energy origin based on median shift
        shift_ev = np.median(shifts) * ll.sampling[2]
        hl.origin[2] -= shift_ev

        print("QuantEM: Alignment and Dual-EELS sync complete.")
        return ll, hl, shifts

    def plot_absolute_zlp_shift(dataset, search_window=(-10, 10)):
        """
        Calculates the ZLP shift per pixel and plots the absolute deviation from 0.0 eV.
        """
        data = dataset.array

        # Generate energy axis
        energies = np.asarray(dataset.energy_axis, dtype=float)

        # Mask energy window for peak finding
        mask = (energies > search_window[0]) & (energies < search_window[1])
        search_energies = energies[mask]

        # Calculate peak map and absolute deviation
        peak_indices = np.argmax(data[:, :, mask], axis=2)
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

    def visualize_thickness_windows(dataset, zlp_window=(-3.0, 3.0), total_window=(-3.0, 75.0)):
        """
        Visualizes integration windows for I0 (ZLP) and It (Total).
        Returns a configuration dictionary for the calculation step.
        """
        # 1. Extract Energy and Mean Spectrum
        data = dataset.array
        mean_spec = np.mean(data, axis=(0, 1))

        # Use built-in energy axis if available, else generate from metadata
        if hasattr(dataset, "energy_axis"):
            energy = np.asarray(dataset.energy_axis, dtype=float)
        else:
            energy = dataset.origin[2] + np.arange(dataset.shape[2]) * dataset.sampling[2]

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
        I_zlp = np.sum(data[:, :, z_start : z_end + 1], axis=2)
        I_total = np.sum(data[:, :, t_start : t_end + 1], axis=2)

        # 2. Log-Ratio Calculation (with epsilon to avoid log(0))
        epsilon = 1e-10
        t_over_lambda = np.log((I_total + epsilon) / (I_zlp + epsilon))

        # 3. Data Cleaning
        t_over_lambda = np.nan_to_num(t_over_lambda, nan=0.0, posinf=0.0, neginf=0.0)
        t_over_lambda = np.clip(t_over_lambda, 0, 4.0)

        if plot:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
            fig.suptitle(f"Thickness Analysis: {dataset.name}", fontsize=14)

            im = ax1.imshow(t_over_lambda, cmap="viridis", origin="upper")
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

    def interpret_thickness_quality(t_over_lambda, a=0.3, b=1, c=2, dataset=None):
        """
        Performs a scientific quality assessment on the calculated t/lambda map.

        The Physical Meaning of the ThresholdsThe t/lambda value represents the average number of inelastic scattering events
        an electron undergoes.
        Vacuum (< a):
            (default a = 0.3)
            In pure vacuum, t/lambda should be 0. In practice, values up to ~0.3 often indicate the presence of thin carbon support films,
            surface contamination, or detector noise. Measurements in this regime are highly sensitive to ZLP (Zero Loss Peak) estimation errors.

        Thin (a <t/lambda < b):
            (default b = 1)
            The "Sweet Spot" for EELS. At t/lambda ~1, the probability of a single inelastic scattering event is maximized.
            In this regime, core-loss edges are sharp and clearly visible without the immediate need for complex mathematical
            deconvolution (e.g., Fourier-Log) to remove multiple scattering effects.

        Medium (b < t/lambda < c):
            (default c = 2)
            Multiple scattering begins to dominate the spectrum. The plural scattering of plasmons creates "ghost" peaks
            that overlap with higher-energy chemical edges. While data is still usable, quantitative analysis typically
            requires plural scattering correction for high accuracy.

        Thick (t/lambda > c):
            The "Multiple Scattering Regime.
            " Most electrons have undergone three or more scattering events, resulting in a "spectral soup"
            where fine-structure details and high-resolution chemical information are significantly broadened or lost.
        """

        name = dataset.name if dataset else "Dataset"

        # Classification Masks
        vacuum = t_over_lambda < a
        thin = (t_over_lambda >= a) & (t_over_lambda < b)
        medium = (t_over_lambda >= b) & (t_over_lambda < c)
        thick = t_over_lambda >= c

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

    def plot_dual_eels_picker(ll, hl, coords=None, title="QuantEM: Dual-EELS Analysis"):
        """
        Dual-EELS Picker with starting coordinates.

        coords, when provided, is interpreted as (scan_row, scan_col).
        """
        # 1. Setup Data
        sum_ll = np.sum(ll.array, axis=2)
        sum_hl = np.sum(hl.array, axis=2)
        energy_ll = np.asarray(ll.energy_axis, dtype=float)
        energy_hl = np.asarray(hl.energy_axis, dtype=float)

        # 2. Handle Initial Coordinates
        if coords is not None:
            i_row, i_col = coords
        else:
            i_row, i_col = ll.shape[0] // 2, ll.shape[1] // 2

        # 3. Create Figure
        fig, axes = plt.subplots(2, 2, figsize=(14, 9))
        fig.suptitle(f"{title}\n(Click on maps to update spectra)", fontsize=16)
        ax_map_ll, ax_spec_ll = axes[0, 0], axes[0, 1]
        ax_map_hl, ax_spec_hl = axes[1, 0], axes[1, 1]

        # Plot Maps & Markers
        ax_map_ll.imshow(sum_ll, cmap="viridis", origin="lower")
        (marker_ll,) = ax_map_ll.plot(i_col, i_row, "r+", ms=15, mew=2)

        ax_map_hl.imshow(sum_hl, cmap="magma", origin="lower")
        (marker_hl,) = ax_map_hl.plot(i_col, i_row, "r+", ms=15, mew=2)

        # Plot Initial Spectra
        (line_ll,) = ax_spec_ll.plot(energy_ll, ll.array[i_row, i_col, :], color="tab:blue")
        (line_hl,) = ax_spec_hl.plot(energy_hl, hl.array[i_row, i_col, :], color="tab:red")

        def update_plots(i_row, i_col):
            marker_ll.set_data([i_col], [i_row])
            marker_hl.set_data([i_col], [i_row])

            new_ll = ll.array[i_row, i_col, :]
            new_hl = hl.array[i_row, i_col, :]
            line_ll.set_ydata(new_ll)
            line_hl.set_ydata(new_hl)

            # Rescale
            ax_spec_ll.set_ylim(0, np.max(new_ll) * 1.1)
            ax_spec_hl.set_ylim(0, np.max(new_hl) * 1.1)

            ax_spec_ll.set_title(f"LL Spectrum at ({i_row}, {i_col})")
            ax_spec_hl.set_title(f"HL Spectrum at ({i_row}, {i_col})")
            fig.canvas.draw_idle()

        def on_click(event):
            if event.inaxes in [ax_map_ll, ax_map_hl]:
                i_col, i_row = int(round(event.xdata)), int(round(event.ydata))
                if 0 <= i_row < ll.shape[0] and 0 <= i_col < ll.shape[1]:
                    update_plots(i_row, i_col)

        fig.canvas.mpl_connect("button_press_event", on_click)

        ax_spec_ll.set_title(f"LL Spectrum at ({i_row}, {i_col})")
        ax_spec_hl.set_title(f"HL Spectrum at ({i_row}, {i_col})")

        plt.tight_layout()
        plt.close(fig)  # Prevents double-plotting in VS Code
        return fig

    def plot_quantem_diagnostic(dataset, zlp_window=5.0, title_suffix=""):
        """
        QuantEM Diagnostic Dashboard: Visualizes mean spectra, spatial variation,
        and Zero Loss Peak (ZLP) centering accuracy.

        1. Global Average Spectrum (Top Left): Shows the mean intensity across the entire scan.
        It is used to check the signal-to-noise ratio and see if the Zero Loss Peak (ZLP) is roughly centered at 0 eV.
        2. Spatial Variation (Top Right): Plots spectra from a 5x5 grid of pixels across your sample.
        This helps you see if the energy shift or intensity changes drastically from one side of the scan to the other
        (e.g., due to sample thickness changes or beam drift).
        3. Integrated Intensity Map (Bottom Left): A spatial image of the total counts.
        This is your "search image" to help you correlate the spectral data with the physical structure of your sample.
        4. ZLP Alignment Detail (Bottom Right): A high-zoom view of the energy region around 0 eV of the Mean Spectrum.
        It includes a dashed green line at the "Target 0" to show exactly how much residual calibration error remains
        after your alignment.

        Parameters:
        -----------
        dataset : QuantEM Object
            The EELS dataset containing .array, .origin, and .sampling attributes.
        zlp_window : float, optional
            The energy range (± eV) to display in the ZLP zoom plot. Default is 5.0.
        title_suffix : str, optional
            Additional text to append to the figure title (e.g., "(RAW)" or "(Aligned)").

        Returns:
        --------
        fig : matplotlib.figure.Figure
            The figure object for further manipulation or saving.
        """
        data = dataset.array
        energy = np.asarray(dataset.energy_axis, dtype=float)

        mean_spec = np.mean(data, axis=(0, 1))
        zlp_pos = energy[np.argmax(mean_spec)]
        sum_img = np.sum(data, axis=2)

        fig = plt.figure(figsize=(14, 9))
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.2)
        fig.suptitle(f"QuantEM Diagnostic: {dataset.name} {title_suffix}", fontsize=16)

        # 1. Mean Spectrum
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(energy, mean_spec, color="black", label="Mean")
        ax1.axvline(0, color="green", ls=":", label="Target")
        ax1.set_title("Global Average Spectrum")
        ax1.legend()

        # 2. Spatial Variability
        ax2 = fig.add_subplot(gs[0, 1])
        # Take a 5x5 grid for better representation than 3x3
        yy, xx = np.meshgrid(
            np.linspace(0, data.shape[0] - 1, 5, dtype=int),
            np.linspace(0, data.shape[1] - 1, 5, dtype=int),
        )
        for y, x in zip(yy.flatten(), xx.flatten()):
            ax2.plot(energy, data[y, x, :], alpha=0.3, lw=0.5)
        ax2.set_title("Spatial Variation (Grid Samples)")

        # 3. Map
        ax3 = fig.add_subplot(gs[1, 0])
        im = ax3.imshow(sum_img, cmap="viridis", origin="lower")
        plt.colorbar(im, ax=ax3)
        ax3.set_title("Integrated Intensity")

        # 4. ZLP Zoom
        ax4 = fig.add_subplot(gs[1, 1])
        mask = (energy > zlp_pos - zlp_window) & (energy < zlp_pos + zlp_window)
        ax4.plot(energy[mask], mean_spec[mask], lw=2)
        ax4.axvline(0, color="green", ls=":")
        ax4.set_title("ZLP Alignment Detail")
        plt.close(fig)

        return fig

    def plot_zlp_drift_diagnostics(dataset, title="ZLP Drift Analysis"):
        """
        QuantEM Diagnostic: Maps the ZLP position and calculates the drift distribution.
        Uses scipy.stats for Gaussian fitting.
        """
        data = dataset.array
        energy = np.asarray(dataset.energy_axis, dtype=float)

        # 1. Mask and find peak per pixel
        search_mask = (energy > -2.0) & (energy < 2.0)
        search_energies = energy[search_mask]
        peak_indices = np.argmax(data[:, :, search_mask], axis=2)
        zlp_map = search_energies[peak_indices]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle(f"QuantEM: {dataset.name} - {title}", fontsize=16)

        # Plot A: Map
        im = ax1.imshow(zlp_map, cmap="RdYlBu_r", origin="lower")
        plt.colorbar(im, ax=ax1, label="Energy Shift (eV)")

        # Plot B: Histogram + Scipy Fit
        flat_pos = zlp_map.flatten()
        mu, std = norm.fit(flat_pos)  # Professional scipy fitting

        ax2.hist(flat_pos, bins=30, density=True, alpha=0.6, color="skyblue")
        x_range = np.linspace(np.min(flat_pos), np.max(flat_pos), 100)
        ax2.plot(
            x_range,
            norm.pdf(x_range, mu, std),
            color="darkred",
            lw=2,
            label=f"Fit: μ={mu:.3f} eV, σ={std:.3f} eV",
        )
        ax2.legend()

        plt.tight_layout()

        plt.close(fig)

        return fig
