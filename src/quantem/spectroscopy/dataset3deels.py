from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import interp1d
from scipy.ndimage import median_filter
from scipy.optimize import curve_fit
from sklearn.decomposition import PCA

from quantem.spectroscopy import Dataset3dspectroscopy


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
        self.dataset_type = "EELS"

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

    def smooth_eels_pca(self, roi=None, energy_range=None, ignore_range=None, mask=None):
        pca = PCA(n_components=2)
        # kpca = KernelPCA(n_components=10, kernel='rbf', gamma=50, fit_inverse_transform=True)

        # #test on mean spectrum
        # spec = self.calculate_mean_spectrum(roi, energy_range, ignore_range, mask)

        dE = float(self.sampling[0])
        E0 = float(self.origin[0]) if hasattr(self, "origin") else 0.0
        energy_axis = E0 + dE * np.arange(self.shape[0])

        # if energy_range is not None:
        #     energy_range[0] = np.maximum(energy_range[0], energy_axis[0])
        #     energy_range[1] = np.minimum(energy_range[1], energy_axis[-1])

        #     indices = np.where(
        #         (energy_axis >= energy_range[0]) & (energy_axis <= energy_range[1])
        #     )[0]
        #     energy_axis = energy_axis[indices]
        # else:
        #     indices = np.arange(self.shape[0])

        # Try denoising on 2D SI images

        # transformed_array3d = np.empty([self.array.shape[0], self.array.shape[1], self.array.shape[2]], dtype=float)

        # for kk in range(self.array.shape[1]):
        #     spec_image = self.array[:,kk,:]
        #     # array2d_transformed = pca.fit(spec_image)
        #     # array2d_smoothed = pca.inverse_transform(array2d_transformed)
        #     array2d_transformed = kpca.fit_transform(spec_image)
        #     array2d_smoothed = kpca.inverse_transform(array2d_transformed)
        #     transformed_array3d[:,kk,:] = array2d_smoothed

        # Reduce 3D dataset to two dimensions

        # array2d = self.array.reshape(self.array.shape[0],self.array.shape[1]*self.array.shape[2])
        # array2d_transformed = kpca.fit_transform(array2d)
        # array2d_smoothed = kpca.inverse_transform(array2d_transformed)

        array2d = self.array.reshape(
            self.array.shape[0], self.array.shape[1] * self.array.shape[2]
        )
        pca.fit(array2d)
        variance = pca.explained_variance_

        array2d_smoothed = pca.inverse_transform(pca.transform(array2d))
        array3d_smoothed = array2d_smoothed.reshape(self.array.shape)

        fig, scree = plt.subplots()
        values = np.arange(len(variance)) + 1
        scree.plot(values, variance, label="Scree plot", marker="o")
        scree.legend()

        smoothed_data3d = Dataset3deels.from_array(
            array=array3d_smoothed,
            sampling=self.sampling,
            origin=self.origin,
            units=self.units,
        )

        mean_spectrum_raw = self.calculate_mean_spectrum(roi, energy_range, ignore_range, mask)
        mean_spectrum_smoothed = smoothed_data3d.calculate_mean_spectrum(
            roi, energy_range, ignore_range, mask
        )

        fig, ax = plt.subplots()
        ax.plot(energy_axis, mean_spectrum_raw, label="raw spectrum", color="b")
        ax.plot(energy_axis, mean_spectrum_smoothed, label="kpca-fit spectrum", color="r")
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
