import csv
import json
import os
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from numpy.typing import NDArray
from sklearn.decomposition import PCA

from quantem.core.datastructures.dataset3d import Dataset3d
from quantem.spectroscopy.utils import load_xray_lines_database


class Dataset3dspectroscopy(Dataset3d):
    # stores the element line info so you don't need to reload each time
    element_info = None
    element_info_path = "x_ray_lines.csv"
    atomic_weights = None
    atomic_weights_path = "atomic_weights.csv"
    dataset_type = "EDS"

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
        super().__init__(
            array=array,
            name=name,
            origin=origin,
            sampling=sampling,
            units=units,
            signal_units=signal_units,
            _token=type(self)._token if _token is None else _token,
        )

        # Initialize model elements storage
        self.model_elements = None
        # Initialize spectra storage
        self.attached_spectra = None

    # loads elemental information
    @classmethod
    def load_element_info(cls):
        """Load element database for EDS (X-ray lines) or EELS (binding energies)."""
        if cls.element_info is not None:
            return

        class_type = str(getattr(cls, "dataset_type", "")).strip().lower()
        path = (
            "eels_binding_energies.json"
            if class_type == "eels"
            else getattr(cls, "element_info_path", "x_ray_lines.csv")
        )
        full_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), path)

        if path.lower().endswith(".csv"):
            cls.element_info = load_xray_lines_database(full_path)
        else:
            with open(full_path, "r", encoding="utf-8") as f:
                cls.element_info = json.load(f)["elements"]

    @classmethod
    def load_atomic_weights(cls):
        """Load atomic weights table from CSV once per class."""
        if cls.atomic_weights is not None:
            return

        full_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), cls.atomic_weights_path
        )
        data = {}
        with open(full_path, "r", newline="") as f:
            reader = csv.reader(f)
            for row_index, row in enumerate(reader, start=1):
                if not row:
                    continue
                if len(row) < 2:
                    raise ValueError(
                        f"{cls.atomic_weights_path} row {row_index} must contain element symbol and weight"
                    )
                symbol = str(row[0]).strip()
                weight_raw = str(row[1]).strip()
                if not symbol:
                    continue
                try:
                    weight = float(weight_raw)
                except ValueError as exc:
                    raise ValueError(
                        f"{cls.atomic_weights_path} row {row_index} has invalid weight: {weight_raw!r}"
                    ) from exc
                data[symbol] = weight

        if not data:
            raise ValueError(f"{cls.atomic_weights_path} did not contain any atomic weights")

        cls.atomic_weights = data

    @staticmethod
    def _normalize_element_specs(specs):
        if isinstance(specs, str):
            return [s.strip() for s in specs.split(",") if s.strip()]
        if isinstance(specs, (list, tuple, set)):
            out = []
            for spec in specs:
                out.extend([s.strip() for s in str(spec).split(",") if s.strip()])
            return out
        raise TypeError("elements must be a string or a sequence of strings")

    @staticmethod
    def _resolve_element_key(all_info, token):
        token_norm = str(token).strip().lower()
        return next((key for key in all_info if str(key).lower() == token_norm), None)

    @staticmethod
    def _line_matches_selectors(line_name, selectors):
        if not selectors:
            return True
        line_norm = str(line_name).strip().lower()
        return any(line_norm == sel or line_norm.startswith(sel) for sel in selectors)

    @classmethod
    def _select_lines(cls, line_dict, selectors):
        if not isinstance(line_dict, dict):
            return {}
        if not selectors:
            return dict(line_dict)

        selector_norm = [str(sel).strip().lower() for sel in selectors if str(sel).strip()]
        return {
            line_name: line_info
            for line_name, line_info in line_dict.items()
            if cls._line_matches_selectors(line_name, selector_norm)
        }

    def add_elements_to_model(self, elements):
        """
        Add elements to the model for persistent use in show_mean_spectrum.

        Parameters
        ----------
        elements : list or str
            Element/line spec(s) to add. Examples:
            - 'Al' (all lines for Al)
            - 'Te La' (only Te La line)
            - ['Au Ma', 'Te La', 'Si']
        """
        # Load element info if not already loaded
        if type(self).element_info is None:
            type(self).load_element_info()

        all_info = type(self).element_info
        if all_info is None:
            return

        specs = type(self)._normalize_element_specs(elements)
        if self.model_elements is None:
            self.model_elements = {}

        for spec in specs:
            tokens = str(spec).split()
            if not tokens:
                continue

            element_key = type(self)._resolve_element_key(all_info, tokens[0])
            if element_key is None:
                continue

            selectors = tokens[1:]
            selected_lines = type(self)._select_lines(all_info[element_key], selectors)
            if not selected_lines:
                continue

            if not selectors:
                self.model_elements[element_key] = selected_lines
            else:
                existing = self.model_elements.get(element_key)
                if not isinstance(existing, dict):
                    existing = {}
                existing.update(selected_lines)
                self.model_elements[element_key] = existing

        if not self.model_elements:
            self.model_elements = None

    def remove_elements_from_model(self, elements):
        """
        Remove element(s) from the persistent model used in show_mean_spectrum.

        Parameters
        ----------
        elements : list or str
            Element/line spec(s) to remove. Examples:
            - 'Al' (remove all Al lines)
            - 'Te La' (remove only Te La line)
            - ['Au Ma', 'Te La']
        """
        if self.model_elements is None:
            return

        specs = type(self)._normalize_element_specs(elements)
        for spec in specs:
            tokens = str(spec).split()
            if not tokens:
                continue

            element_key = type(self)._resolve_element_key(self.model_elements, tokens[0])
            if element_key is None:
                continue

            selectors = [str(token).strip().lower() for token in tokens[1:] if str(token).strip()]
            if not selectors:
                self.model_elements.pop(element_key, None)
                continue

            lines_info = self.model_elements.get(element_key)
            if not isinstance(lines_info, dict):
                self.model_elements.pop(element_key, None)
                continue

            self.model_elements[element_key] = {
                line_name: line_info
                for line_name, line_info in lines_info.items()
                if not type(self)._line_matches_selectors(line_name, selectors)
            }
            if not self.model_elements[element_key]:
                self.model_elements.pop(element_key, None)

        if not self.model_elements:
            self.model_elements = None

    def clear_model_elements(self):
        """Clear all elements from the model."""
        self.model_elements = None

    # Storage of spectra alongside dataset

    def add_spectrum_to_data(self, spectrum, energy_axis):
        """
        Store processed spectra in the 3D spectroscopy dataset structure, in a 1D array of 2D arrays. By default, calculate_mean_spectrum will
        """
        from quantem.core.datastructures.dataset1d import Dataset1d

        two_d_spectrum = Dataset1d.from_array(
            array=spectrum, origin=energy_axis[0], sampling=self.sampling[0], units=self.units[0]
        )

        if self.attached_spectra is not None:
            self.attached_spectra.append(two_d_spectrum)
        else:
            self.attached_spectra = []
            self.attached_spectra.append(two_d_spectrum)

    def clear_attached_spectra(self):
        self.attached_spectra = None

    def plot_attached_spectrum(self, data_type="eds", spectrum_index=0):
        fig, (ax_spec) = plt.subplots(1, 1, figsize=(12, 4))

        ax_spec.plot(
            self.attached_spectra[spectrum_index][1],
            self.attached_spectra[spectrum_index][0],
            linewidth=1.5,
        )
        if data_type == "eds":
            ax_spec.set_xlabel("Energy (keV)")
        elif data_type == "eels":
            ax_spec.set_xlabel("Energy (eV)")
        ax_spec.set_ylabel("Intensity")
        ax_spec.set_title(f"Spectrum in index {spectrum_index}")
        ax_spec.grid(True, alpha=0.1)

        fig.tight_layout()
        plt.show()

    ## PCA ANALYSIS METHODS

    def perform_pca(
        self,
        n_components: int = 10,
        standardize: bool = True,
        mask: Optional[NDArray] = None,
        plot_results: bool = True,
        random_state: Optional[int] = 42,
    ) -> dict:
        """
        Perform Principal Component Analysis (PCA) on the spectroscopy dataset.

        Parameters
        ----------
        n_components : int
            Number of principal components to compute
        standardize : bool
            If True, standardize the data before PCA (zero mean, unit variance)
        mask : Optional[NDArray]
            Optional spatial mask to select pixels for analysis
        plot_results : bool
            If True, plot the explained variance and first few components
        random_state : Optional[int]
            Random state for reproducibility

        Returns
        -------
        dict
            Dictionary containing:
            - 'pca': fitted PCA object
            - 'components': principal component spectra (n_components x n_energy)
            - 'loadings': spatial loadings (n_components x n_pixels)
            - 'explained_variance_ratio': explained variance for each component
            - 'reconstructed': reconstructed dataset using n_components
        """
        data = np.asarray(self.array, dtype=float)
        n_energy, ny, nx = data.shape

        # Reshape data to (n_pixels, n_energy) for PCA
        data_reshaped = data.reshape(n_energy, -1).T  # (n_pixels, n_energy)

        if mask is not None:
            mask_flat = mask.flatten()
            data_masked = data_reshaped[mask_flat]
        else:
            data_masked = data_reshaped

        if standardize:
            mean = np.mean(data_masked, axis=0)
            std = np.std(data_masked, axis=0)
            std[std == 0] = 1  # Avoid division by zero
            data_processed = (data_masked - mean) / std
        else:
            data_processed = data_masked

        # Perform PCA
        pca = PCA(n_components=n_components, random_state=random_state)
        loadings = pca.fit_transform(data_processed)  # (n_pixels, n_components)
        components = pca.components_  # (n_components, n_energy)

        # Reconstruct data
        if standardize:
            reconstructed = pca.inverse_transform(loadings) * std + mean
        else:
            reconstructed = pca.inverse_transform(loadings)

        if mask is None:
            loadings_spatial = loadings.T.reshape(n_components, ny, nx)
        else:
            loadings_spatial = np.zeros((n_components, ny * nx))
            loadings_spatial[:, mask_flat] = loadings.T
            loadings_spatial = loadings_spatial.reshape(n_components, ny, nx)

        if plot_results:
            self._plot_pca_results(
                components,
                loadings_spatial,
                pca.explained_variance_ratio_,
                n_show=min(4, n_components),
            )

        return {
            "pca": pca,
            "components": components,
            "loadings": loadings_spatial,
            "explained_variance_ratio": pca.explained_variance_ratio_,
            "explained_variance": pca.explained_variance_,
            "reconstructed": reconstructed.T.reshape(n_energy, ny, nx)
            if mask is None
            else reconstructed,
        }

    def _plot_pca_results(
        self,
        components: NDArray,
        loadings: NDArray,
        explained_variance_ratio: NDArray,
        n_show: int = 4,
    ):
        """
        Plot PCA results including scree plot, components, and loadings.

        Parameters
        ----------
        components : NDArray
            Principal component spectra
        loadings : NDArray
            Spatial loadings for each component
        explained_variance_ratio : NDArray
            Explained variance ratios
        n_show : int
            Number of components to show
        """
        fig = plt.figure(figsize=(15, 10))
        gs = fig.add_gridspec(3, n_show + 1, width_ratios=[1.5] + [1] * n_show)

        # Plot 1: Scree plot (explained variance)
        ax_scree = fig.add_subplot(gs[0, 0])
        cumsum_var = np.cumsum(explained_variance_ratio)

        ax_scree.bar(
            range(1, len(explained_variance_ratio) + 1),
            explained_variance_ratio * 100,
            alpha=0.6,
            label="Individual",
        )
        ax_scree.plot(
            range(1, len(explained_variance_ratio) + 1),
            cumsum_var * 100,
            "ro-",
            label="Cumulative",
        )
        ax_scree.set_xlabel("Component Number")
        ax_scree.set_ylabel("Explained Variance (%)")
        ax_scree.set_title("Scree Plot")
        ax_scree.legend()
        ax_scree.grid(True, alpha=0.3)

        # Get energy axis
        energy_sampling = float(self.sampling[0])
        energy_origin = float(self.origin[0])
        energy_axis = energy_origin + energy_sampling * np.arange(components.shape[1])

        # Plot components and loadings
        for i in range(n_show):
            ax_comp = fig.add_subplot(gs[1, i + 1])
            ax_comp.plot(energy_axis, components[i])
            ax_comp.set_title(f"PC{i + 1} ({explained_variance_ratio[i] * 100:.1f}%)")
            ax_comp.set_xlabel("Energy")
            if i == 0:
                ax_comp.set_ylabel("Component")
            ax_comp.grid(True, alpha=0.3)

            ax_load = fig.add_subplot(gs[2, i + 1])
            im = ax_load.imshow(loadings[i], cmap="RdBu_r", origin="lower")
            ax_load.set_title(f"Loading {i + 1}")
            ax_load.axis("off")
            plt.colorbar(im, ax=ax_load, fraction=0.046, pad=0.04)

        ax_stats = fig.add_subplot(gs[1:, 0])
        ax_stats.axis("off")

        stats_text = "PCA Summary\n" + "=" * 20 + "\n\n"
        stats_text += f"Total components: {len(explained_variance_ratio)}\n"
        stats_text += f"Components for 95% var: {np.argmax(cumsum_var >= 0.95) + 1}\n"
        stats_text += f"Components for 99% var: {np.argmax(cumsum_var >= 0.99) + 1}\n\n"

        for i in range(min(5, len(explained_variance_ratio))):
            stats_text += f"PC{i + 1}: {explained_variance_ratio[i] * 100:.2f}%\n"

        ax_stats.text(
            0.1,
            0.9,
            stats_text,
            transform=ax_stats.transAxes,
            fontsize=10,
            verticalalignment="top",
            fontfamily="monospace",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

        plt.suptitle("PCA Analysis Results", fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.show()

    def calculate_mean_spectrum(
        self, roi=None, energy_range=None, ignore_range=None, mask=None, attach_mean_spectrum=True
    ):
        # ADJUST ROI BASED ON GIVEN FLAGS -----------------------------------------------
        # Parse ROI parameter
        if roi is None:
            # Full image
            y, x, dy, dx = 0, 0, int(self.shape[1]), int(self.shape[2])
        elif len(roi) == 2:
            # Single pixel [y, x]
            y, x, dy, dx = int(roi[0]), int(roi[1]), 1, 1
        elif len(roi) == 4:
            # Full ROI [y, x, dy, dx] with None support for defaults
            y_val, x_val, dy_val, dx_val = roi

            # Handle None values with defaults
            y = 0 if y_val is None else int(y_val)
            x = 0 if x_val is None else int(x_val)
            dy = int(self.shape[1]) - y if dy_val is None else int(dy_val)
            dx = int(self.shape[2]) - x if dx_val is None else int(dx_val)
        else:
            raise ValueError(
                "roi must be None, [y, x], or [y, x, dy, dx] (with None for defaults)"
            )

        # VALIDATE ROI BOUNDS ---------------------------------------------------------------------------
        errs = []
        Ymax = int(self.shape[1])
        Xmax = int(self.shape[2])

        # type/NaN checks (optional if you already cast to int above)
        for name, val in (("y", y), ("x", x), ("dy", dy), ("dx", dx)):
            if val is None:
                errs.append(f"{name} is None (missing after normalization).")

        # if any None, bail early to avoid arithmetic errors
        if errs:
            raise ValueError("Invalid ROI:\n - " + "\n - ".join(errs))

        # basic constraints
        if y < 0:
            errs.append(f"y={y} < 0")
        if x < 0:
            errs.append(f"x={x} < 0")
        if dy < 1:
            errs.append(f"dy={dy} < 1")
        if dx < 1:
            errs.append(f"dx={dx} < 1")

        # starts within image
        if y >= Ymax:
            errs.append(f"y start {y} out of bounds [0, {Ymax - 1}]")
        if x >= Xmax:
            errs.append(f"x start {x} out of bounds [0, {Xmax - 1}]")

        # ends within image
        end_y = y + dy
        end_x = x + dx
        if end_y > Ymax:
            errs.append(f"y+dy = {end_y} exceeds height {Ymax}")
        if end_x > Xmax:
            errs.append(f"x+dx = {end_x} exceeds width {Xmax}")

        if errs:
            raise ValueError("Invalid ROI:\n - " + "\n - ".join(errs))

        # SPECTRUM CALCULATION --------------------------------------------------------------

        dE = float(self.sampling[0])
        E0 = float(self.origin[0]) if hasattr(self, "origin") else 0.0
        E = E0 + dE * np.arange(self.shape[0])

        # MASK HANDLING ---------------------------------------------------------------------
        if mask is not None:
            # Convert to ndarray and validate
            mask = np.asarray(mask)

            # Check that it's a proper ndarray
            if not isinstance(mask, np.ndarray):
                raise TypeError(f"Mask must be a numpy ndarray, got {type(mask)}")

            # Check dimensions - must be 1D
            if mask.ndim != 1:
                raise ValueError(
                    f"Mask must be 1-dimensional, got {mask.ndim}D array with shape {mask.shape}"
                )

            # Convert to bool dtype and validate
            if mask.dtype != bool:
                try:
                    mask = mask.astype(bool)
                except (ValueError, TypeError):
                    raise TypeError(f"Mask cannot be converted to boolean dtype from {mask.dtype}")

            # Check shape matches energy axis
            arr = np.asarray(self.array, dtype=float)
            if mask.shape != (arr.shape[0],):
                raise ValueError(
                    f"Mask shape {mask.shape} does not match energy axis shape ({arr.shape[0]},)"
                )

            arr = arr[mask, :, :]  # select only masked energy channels
            spec = arr.sum(axis=(1, 2)) if arr.shape[0] > 0 else np.zeros(0)
            E = E[mask]  # Mask the energy axis as well
        else:
            spec = np.empty(self.shape[0], dtype=float)
            for k in range(self.shape[0]):
                img = np.asarray(self.array[k], dtype=float)
                roi = img[y : y + dy, x : x + dx]
                if roi.size == 0:
                    raise ValueError("ROI is empty; check y/x/dy/dx.")
                spec[k] = roi.mean()

        # APPLY ENERGY RANGE ---------------------------------------------------------------

        if energy_range is not None:
            # Check for errors in energy_range input
            if energy_range[0] >= energy_range[1]:
                raise ValueError("Invalid energy range parameter.")

            # If the entire energy range specified is outside the original energy range of the data, raise an error.
            if energy_range[1] < E[0] or energy_range[0] > E[-1]:
                raise ValueError("Energy range parameter is outside of data bounds.")

            # If either side of input energy_range is beyond the original energy range of the data, default to the limit of the data instead.
            energy_range[0] = np.maximum(energy_range[0], E[0])
            energy_range[1] = np.minimum(energy_range[1], E[-1])

            indices = np.where((E >= energy_range[0]) & (E <= energy_range[1]))[0]
            spec = spec[indices]
            E = E[indices]

        if attach_mean_spectrum:
            self.add_spectrum_to_data(spec, E)

        return spec

    def show_mean_spectrum(
        self,
        roi=None,
        energy_range=None,
        elements=None,
        ignore_range=None,
        threshold=5.0,
        tolerance=0.15,
        mask=None,
        show_lines=True,
        show_text=True,
        snr_min=None,
        snr_threshold=None,
        distance_threshold_for_sample=0.05,
        grid_peaks=None,
        data_type="eds",
        peaks=15,
        show=True,
    ):
        """
        Plot the mean spectrum from a spatial ROI in a 3D spectroscopy cube (E, Y, X).

        Parameters
        ----------
        roi : list or tuple, optional
            Region of interest as [y, x, dy, dx] where:
            - y, x: top-left pixel coordinates
            - dy, dx: height and width of ROI
            Use None for default values:
            - [y, None, dy, None] = row y with height dy, full width
            - [None, x, None, dx] = column x with width dx, full height
            - [y, x, None, None] = from (y,x) to bottom-right corner
            If roi=None, uses full image. Can also be [y, x] for single pixel.
        energy_range : list or tuple, optional
            Energy range to display as [min_energy, max_energy] in keV.
        ignore_range : list or tuple, optional
            Ignored in this plotting-only method. Kept for backward compatibility.
        mask : array, optional
            Boolean mask for pixel selection.
        show : bool, optional
            If True, display the plot with ``plt.show()``. Set False to add overlays before showing.
        data_type : str, optional
            Type of spectroscopy data. Options: 'eds' (default) or 'eels'.

        Returns
        -------
        (fig, ax) : tuple
            The Matplotlib Figure and Axes of the spectrum plot.
        """

        # ADJUST ROI BASED ON GIVEN FLAGS -----------------------------------------------
        # Parse ROI parameter
        if roi is None:
            # Full image
            y, x, dy, dx = 0, 0, int(self.shape[1]), int(self.shape[2])
        elif len(roi) == 2:
            # Single pixel [y, x]
            y, x, dy, dx = int(roi[0]), int(roi[1]), 1, 1
        elif len(roi) == 4:
            # Full ROI [y, x, dy, dx] with None support for defaults
            y_val, x_val, dy_val, dx_val = roi

            # Handle None values with defaults
            y = 0 if y_val is None else int(y_val)
            x = 0 if x_val is None else int(x_val)
            dy = int(self.shape[1]) - y if dy_val is None else int(dy_val)
            dx = int(self.shape[2]) - x if dx_val is None else int(dx_val)
        else:
            raise ValueError(
                "roi must be None, [y, x], or [y, x, dy, dx] (with None for defaults)"
            )

        # CALCULATE MEAN SPECTRUM FOR GIVEN ROI AND ENERGY RANGE --------------------------

        spec = self.calculate_mean_spectrum(roi, energy_range, ignore_range, mask)

        dE = float(self.sampling[0])
        E0 = float(self.origin[0]) if hasattr(self, "origin") else 0.0
        E = E0 + dE * np.arange(self.shape[0])

        if energy_range is not None:
            indices = np.where((E >= energy_range[0]) & (E <= energy_range[1]))[0]
            E = E[indices]

        # Store ignore_range for later use in element line filtering
        if ignore_range is None:
            ignore_range = [0, 0.25]  # Default: ignore 0-0.25 keV for element lines only

        # PLOTTING ---------------------------------------------------------------------------

        # Create subplot layout: image on left, spectrum on right
        fig, (ax_img, ax_spec) = plt.subplots(1, 2, figsize=(12, 4))

        # LEFT PLOT: Show sum image with ROI highlighted
        # Create sum image across all energy channels (or masked channels)
        if mask is not None:
            sum_img = np.asarray(self.array, dtype=float)[mask, :, :].sum(axis=0)
            title_suffix = " (masked energies)"
        else:
            sum_img = np.asarray(self.array, dtype=float).sum(axis=0)
            title_suffix = ""

        im = ax_img.imshow(sum_img, cmap="viridis", origin="lower")
        if data_type == "eds":
            ax_img.set_title(f"EDS Sum Image{title_suffix}")
        else:
            ax_img.set_title(f"EELS Sum Image{title_suffix}")
        ax_img.set_xlabel("X (pixels)")
        ax_img.set_ylabel("Y (pixels)")

        # Highlight the ROI with a rectangle
        rect = Rectangle(
            (x - 0.5, y - 0.5), dx, dy, linewidth=2, edgecolor="red", facecolor="none", alpha=0.8
        )
        ax_img.add_patch(rect)

        # Add colorbar for the image
        plt.colorbar(im, ax=ax_img)

        # RIGHT PLOT: Show spectrum
        (spectrum_line,) = ax_spec.plot(E, spec, linewidth=1.5)
        spectrum_color = spectrum_line.get_color()
        if data_type == "eds":
            ax_spec.set_xlabel("Energy (keV)")
        else:
            ax_spec.set_xlabel("Energy (eV)")
        ax_spec.set_ylabel("Intensity")
        ax_spec.set_title(f"Spectrum from ROI [{y}:{y + dy}, {x}:{x + dx}]")
        ax_spec.grid(True, alpha=0.1)

        if show_lines and isinstance(self.model_elements, dict) and len(self.model_elements) > 0:
            x_min = float(np.nanmin(E)) if E.size > 0 else None
            x_max = float(np.nanmax(E)) if E.size > 0 else None
            model_marker_energies = []

            energy_keys = (
                "energy (keV)",
                "energy_keV",
                "energy (eV)",
                "onset (eV)",
                "edge (eV)",
                "energy",
            )

            for _, lines_info in self.model_elements.items():
                if not isinstance(lines_info, dict):
                    continue

                for _, line_info in lines_info.items():
                    if not isinstance(line_info, dict):
                        continue

                    line_energy = None
                    for key in energy_keys:
                        if key in line_info:
                            try:
                                line_energy = float(line_info[key])
                                break
                            except (TypeError, ValueError):
                                continue

                    if line_energy is None:
                        continue
                    if x_min is not None and (line_energy < x_min or line_energy > x_max):
                        continue

                    model_marker_energies.append(line_energy)

            if len(model_marker_energies) > 0:
                marker_x = np.unique(np.asarray(model_marker_energies, dtype=float))
                y_min = float(np.nanmin(spec)) if spec.size > 0 else 0.0
                y_max = float(np.nanmax(spec)) if spec.size > 0 else 1.0
                y_scale = max(y_max - y_min, 1e-12)
                y_dot = y_min - 0.04 * y_scale

                ax_spec.plot(
                    marker_x,
                    np.full(marker_x.shape, y_dot, dtype=float),
                    marker="o",
                    markersize=2.5,
                    color=spectrum_color,
                    alpha=0.5,
                    linestyle="None",
                    zorder=5,
                )

                current_bottom, current_top = ax_spec.get_ylim()
                dot_padding = 0.02 * y_scale
                ax_spec.set_ylim(bottom=min(current_bottom, y_dot - dot_padding), top=current_top)

        fig.tight_layout()
        if show:
            plt.show()
        return fig, (ax_img, ax_spec)

    def show_energy_window_map(
        self,
        energy_window,
        roi=None,
        mask=None,
        data_type="eds",
        cmap="viridis",
        show=True,
    ):
        """Show a spatial map integrated over a selected energy window.

        This is a complementary view to ``show_mean_spectrum``:
        - ``show_mean_spectrum`` answers *what energies are present*.
        - ``show_energy_window_map`` answers *where a chosen energy range is present*.

        Parameters
        ----------
        energy_window : list[float] | tuple[float, float]
            Energy interval [emin, emax] to integrate.
        roi : list | tuple | None, optional
            ROI as ``[y, x]`` or ``[y, x, dy, dx]`` (with ``None`` defaults),
            used only for overlay rectangle.
        mask : array-like | None, optional
            Optional boolean mask over energy channels. If provided, it is
            combined with ``energy_window``.
        data_type : str, optional
            "eds" (keV) or "eels" (eV), used for title/unit text.
        cmap : str, optional
            Matplotlib colormap for the map.
        show : bool, optional
            If True, call ``plt.show()``.

        Returns
        -------
        tuple
            ``(fig, ax, energy_map)`` where ``energy_map`` is the integrated 2D array.
        """
        if energy_window is None or len(energy_window) != 2:
            raise ValueError("energy_window must be [min_energy, max_energy]")

        emin = float(energy_window[0])
        emax = float(energy_window[1])
        if not np.isfinite(emin) or not np.isfinite(emax) or emin >= emax:
            raise ValueError(
                "Invalid energy_window. Expected [min_energy, max_energy] with min < max"
            )

        # Parse ROI (for optional overlay only)
        if roi is None:
            y, x, dy, dx = 0, 0, int(self.shape[1]), int(self.shape[2])
            has_roi_overlay = False
        elif len(roi) == 2:
            y, x, dy, dx = int(roi[0]), int(roi[1]), 1, 1
            has_roi_overlay = True
        elif len(roi) == 4:
            y_val, x_val, dy_val, dx_val = roi
            y = 0 if y_val is None else int(y_val)
            x = 0 if x_val is None else int(x_val)
            dy = int(self.shape[1]) - y if dy_val is None else int(dy_val)
            dx = int(self.shape[2]) - x if dx_val is None else int(dx_val)
            has_roi_overlay = True
        else:
            raise ValueError("roi must be None, [y, x], or [y, x, dy, dx] (with None defaults)")

        dE = float(self.sampling[0])
        E0 = float(self.origin[0]) if hasattr(self, "origin") else 0.0
        E = E0 + dE * np.arange(self.shape[0])

        window_mask = (E >= emin) & (E <= emax)
        if mask is not None:
            mask = np.asarray(mask, dtype=bool)
            if mask.shape != (self.shape[0],):
                raise ValueError(
                    f"Mask shape {mask.shape} does not match energy axis shape ({self.shape[0]},)"
                )
            window_mask = window_mask & mask

        if not np.any(window_mask):
            raise ValueError("No energy channels selected. Adjust energy_window or mask")

        arr = np.asarray(self.array, dtype=float)
        energy_map = arr[window_mask, :, :].sum(axis=0)

        fig, ax = plt.subplots(1, 1, figsize=(6, 5))
        im = ax.imshow(energy_map, cmap=cmap, origin="lower")

        unit_label = "keV" if str(data_type).lower() == "eds" else "eV"
        ax.set_title(f"Energy-Window Map [{emin:.3f}, {emax:.3f}] {unit_label}")
        ax.set_xlabel("X (pixels)")
        ax.set_ylabel("Y (pixels)")

        if has_roi_overlay:
            rect = Rectangle(
                (x - 0.5, y - 0.5),
                dx,
                dy,
                linewidth=2,
                edgecolor="red",
                facecolor="none",
                alpha=0.8,
            )
            ax.add_patch(rect)

        plt.colorbar(im, ax=ax, label="Integrated Intensity")
        fig.tight_layout()

        if show:
            plt.show()

        return fig, ax, energy_map

    # BACKGROND SUBTRACTION

    def subtract_background(
        self,
        roi=None,
        energy_range=None,
        ignore_range=None,
        mask=None,
        data_type="eds",
        return_dataset=True,
        attach_spectrum=True,
    ):
        """
        Perform appropriate background subtraction routine on mean spectrum from a 3D spectroscopy dataset.


        returns:
        A dataset3dspectroscopy object with background subtraction performed at all probe positions

        """

        from quantem.spectroscopy import (
            Dataset3deds as Dataset3deds,
        )
        from quantem.spectroscopy import (
            Dataset3deels as Dataset3deels,
        )

        spec = self.calculate_mean_spectrum(roi, energy_range, ignore_range, mask)

        if data_type == "eds":
            background = self.calculate_background_powerlaw(spec)
        elif data_type == "eels":
            background = self.calculate_background_iterative(spec)

        subtracted_mean_spectrum = np.maximum(spec - background, 0)

        # PLOT MEAN BACKGROUND-SUBTRACTED SPECTRUM ---------------------------------------------------------------------------

        # TODO: store energy axis variable so it doesn't have to be reinitialized repeatedly.
        # for now, this chunk is borrowed from calculate_mean_spectrum
        ###
        dE = float(self.sampling[0])
        E0 = float(self.origin[0]) if hasattr(self, "origin") else 0.0
        E = E0 + dE * np.arange(self.shape[0])

        if energy_range is not None:
            energy_range[0] = np.maximum(energy_range[0], E[0])
            energy_range[1] = np.minimum(energy_range[1], E[-1])

            indices = np.where((E >= energy_range[0]) & (E <= energy_range[1]))[0]
            E = E[indices]
        else:
            indices = np.arange(self.shape[0])
        ###

        fig, (ax_specbacksub) = plt.subplots(1, 1, figsize=(12, 4))

        ax_specbacksub.plot(E, subtracted_mean_spectrum, linewidth=1.5)
        if data_type == "eds":
            ax_specbacksub.set_xlabel("Energy (keV)")
        else:
            ax_specbacksub.set_xlabel("Energy (eV)")
        ax_specbacksub.set_ylabel("Intensity")
        ax_specbacksub.set_title("Background-subtracted spectrum from ROI")
        ax_specbacksub.grid(True, alpha=0.1)

        fig.tight_layout()
        plt.show()

        # NOTE: currently, if an energy_range parameter is set, subtract_background considers ONLY
        # the spectrum data within that energy range, and the output dataset3dspectroscopy object
        # only includes data from that energy range embedded. Not sure that's the best way to implement this.

        spec3D_subtracted = np.empty([spec.shape[0], self.shape[1], self.shape[2]], dtype=float)

        for p in range(self.shape[1]):
            for q in range(self.shape[2]):
                spec3D_subtracted[:, p, q] = np.maximum(self.array[indices, p, q] - background, 0)

        if return_dataset:
            if data_type == "eds":
                return Dataset3deds.from_array(
                    array=spec3D_subtracted,
                    sampling=self.sampling,
                    origin=self.origin,
                    units=self.units,
                )

            elif data_type == "eels":
                return Dataset3deels.from_array(
                    array=spec3D_subtracted,
                    sampling=self.sampling,
                    origin=self.origin,
                    units=self.units,
                )
        else:
            print("Notice: no 3D dataset was returned")

        if attach_spectrum:
            self.add_spectrum_to_data(subtracted_mean_spectrum, E)
        else:
            print(f"Notice: no spectrum recorded to attached_spectra in {self}")

    @property
    def energy_axis(self):
        energy_axis = np.arange(self.shape[0]) * self.sampling[0] + self.origin[0]
        return energy_axis
