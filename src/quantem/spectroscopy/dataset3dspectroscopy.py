import csv
import json
import os
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Rectangle
from numpy.typing import NDArray

from quantem.core.datastructures.dataset3d import Dataset3d
from quantem.core.visualization import show_2d
from quantem.spectroscopy.utils import load_eels_edges_database, load_xray_lines_database


class _ModelElementsDict(dict):
    """dict subclass for model_elements with a readable repr."""

    def __repr__(self):
        if not self:
            return "Model Elements:\n  None"
        lines = ["Model Elements:"]
        for element, line_info in self.items():
            if isinstance(line_info, dict) and line_info:
                line_names = ", ".join(sorted(line_info.keys()))
                lines.append(f"  {element}: {line_names}")
            else:
                lines.append(f"  {element}")
        return "\n".join(lines)

    def _repr_html_(self):
        if not self:
            return "<b>Model Elements:</b><br>&nbsp;&nbsp;None"
        rows = "".join(
            f"<tr><td><b>{el}</b></td><td>{', '.join(sorted(info.keys())) if isinstance(info, dict) and info else ''}</td></tr>"
            for el, info in self.items()
        )
        return f"<b>Model Elements:</b><table>{rows}</table>"


class Dataset3dspectroscopy(Dataset3d):
    # stores the element line info so you don't need to reload each time
    element_info = None
    element_info_path = None
    atomic_weights = None

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

        self.model_elements = _ModelElementsDict()
        self.attached_spectra = None

    # loads elemental information
    @classmethod
    def load_element_info(cls):
        """Load element database for EDS X-ray lines or EELS edges."""
        if cls.element_info is not None:
            return cls.element_info

        path = getattr(cls, "element_info_path", None)
        if path is None:
            raise NotImplementedError(
                f"{cls.__name__} must define `element_info_path` to load element metadata."
            )
        full_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), path)

        dataset_type = str(getattr(cls, "dataset_type", "")).lower()
        if path.lower().endswith(".csv"):
            if dataset_type == "eels":
                cls.element_info = load_eels_edges_database(full_path)
            else:
                cls.element_info = load_xray_lines_database(full_path)
        else:
            with open(full_path, "r", encoding="utf-8") as f:
                cls.element_info = json.load(f)["elements"]

        if dataset_type == "eds":
            cls._normalize_element_info()

        return cls.element_info

    @classmethod
    def _ensure_element_info(cls):
        """Load and return the cached element metadata."""
        return cls.load_element_info() or {}

    @classmethod
    def load_atomic_weights(cls):
        """Load atomic weights table from CSV once per class."""
        if cls.atomic_weights is not None:
            return cls.atomic_weights

        atomic_weights_path = "atomic_weights.csv"
        full_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), atomic_weights_path)
        data = {}
        with open(full_path, "r", newline="") as f:
            reader = csv.reader(f)
            for row_index, row in enumerate(reader, start=1):
                if not row:
                    continue
                if len(row) < 2:
                    raise ValueError(
                        f"{atomic_weights_path} row {row_index} must contain element symbol and weight"
                    )
                symbol = str(row[0]).strip()
                weight_raw = str(row[1]).strip()
                if not symbol:
                    continue
                try:
                    weight = float(weight_raw)
                except ValueError as exc:
                    raise ValueError(
                        f"{atomic_weights_path} row {row_index} has invalid weight: {weight_raw!r}"
                    ) from exc
                data[symbol] = weight

        if not data:
            raise ValueError(f"{atomic_weights_path} did not contain any atomic weights")

        cls.atomic_weights = data
        return cls.atomic_weights

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
        all_info = type(self)._ensure_element_info()
        if all_info is None:
            return

        specs = type(self)._normalize_element_specs(elements)
        added_this_call = {}

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

            existing_before = self.model_elements.get(element_key)
            if not isinstance(existing_before, dict):
                existing_before = {}
            existing_keys_before = set(existing_before.keys())
            if not selectors:
                self.model_elements[element_key] = selected_lines
            else:
                existing = self.model_elements.get(element_key)
                if not isinstance(existing, dict):
                    existing = {}
                existing.update(selected_lines)
                self.model_elements[element_key] = existing

            added_keys = [
                line_name
                for line_name in selected_lines.keys()
                if line_name not in existing_keys_before
            ]
            if added_keys:
                if element_key not in added_this_call:
                    added_this_call[element_key] = []
                added_this_call[element_key].extend(added_keys)
        if not self.model_elements:
            self.model_elements = _ModelElementsDict()

        if added_this_call:
            print("Added to model:")
            for element_key in sorted(added_this_call.keys()):
                unique_lines = sorted(
                    set(str(line_name) for line_name in added_this_call[element_key])
                )
                print(f" - {element_key}: {', '.join(unique_lines)}")
        else:
            print("Added to model: nothing new")

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
        if not self.model_elements:
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
            self.model_elements = _ModelElementsDict()

    def clear_model_elements(self):
        """Clear all elements from the model."""
        self.model_elements = _ModelElementsDict()

    # Storage of spectra alongside dataset

    def add_spectrum_to_data(self, spectrum, energy_axis):
        """
        Store processed spectra in the 3D spectroscopy dataset structure, in a 1D array of 2D arrays. By default, calculate_mean_spectrum will store spectrum at first available index
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

    def plot_attached_spectrum(self, spectrum_index=0):
        fig, (ax_spec) = plt.subplots(1, 1, figsize=(12, 4))

        ax_spec.plot(
            self.attached_spectra[spectrum_index][1],
            self.attached_spectra[spectrum_index][0],
            linewidth=1.5,
        )
        if self.dataset_type == "eds":
            ax_spec.set_xlabel("Energy (keV)")
        elif self.dataset_type == "eels":
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
        return_results=False,
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
            Optional spatial mask to select pixels for analysis. Accepts shape
            (scan_y, scan_x) or a flattened spatial mask.
        plot_results : bool
            If True, plot the explained variance and first few components
        random_state : Optional[int]
            Accepted for API compatibility. PCA uses deterministic SVD.

        Returns
        -------
        dict
            Dictionary containing:
            - 'pca': PCA result attributes
            - 'components': principal component spectra (n_components x n_energy)
            - 'loadings': spatial loadings (n_components x scan_y x scan_x)
            - 'explained_variance_ratio': explained variance for each component
            - 'reconstructed': reconstructed dataset (dataset3dspectroscopy) using n_components
        """

        from quantem.spectroscopy import Dataset3deds, Dataset3deels

        data = np.asarray(self.array, dtype=float)
        n_energy, ny, nx = data.shape
        n_pixels = ny * nx

        spectra = np.moveaxis(data, 0, -1).reshape(n_pixels, n_energy)
        pixel_mask = np.ones(n_pixels, dtype=bool)

        if mask is not None:
            mask_array = np.asarray(mask, dtype=bool)
            if mask_array.shape == (ny, nx):
                pixel_mask = mask_array.reshape(-1)
            elif mask_array.shape == (n_pixels,):
                pixel_mask = mask_array
            else:
                raise ValueError(
                    f"mask shape {mask_array.shape} must match spatial shape {(ny, nx)} "
                    f"or flattened shape {(n_pixels,)}"
                )

        if not np.any(pixel_mask):
            raise ValueError("mask must select at least one spatial pixel")

        selected_spectra = spectra[pixel_mask]

        if standardize:
            mean = np.mean(selected_spectra, axis=0)
            std = np.std(selected_spectra, axis=0)
            std[std == 0] = 1  # Avoid division by zero
            pca_input = (selected_spectra - mean) / std
        else:
            mean = np.zeros(n_energy)
            std = np.ones(n_energy)
            pca_input = selected_spectra

        (
            components,
            loadings,
            explained_variance,
            explained_variance_ratio,
            reconstructed,
        ) = self._run_pca(pca_input, n_components)

        reconstructed = reconstructed * std + mean

        loadings_flat = np.zeros((n_components, n_pixels), dtype=loadings.dtype)
        loadings_flat[:, pixel_mask] = loadings.T
        loadings_spatial = loadings_flat.reshape(n_components, ny, nx)

        if plot_results:
            self._plot_pca_results(
                components,
                loadings_spatial,
                explained_variance_ratio,
                n_show=min(4, n_components),
            )

        reconstructed_spectra = spectra.copy()
        reconstructed_spectra[pixel_mask] = reconstructed
        reconstructed_array = reconstructed_spectra.reshape(ny, nx, n_energy).transpose(2, 0, 1)

        dataset_type = str(self.dataset_type).lower()
        if dataset_type == "eds":
            dataset_class = Dataset3deds
        elif dataset_type == "eels":
            dataset_class = Dataset3deels
        else:
            raise ValueError(f"Unsupported spectroscopy dataset_type {self.dataset_type!r}")

        reconstructed_data3d = dataset_class.from_array(
            array=reconstructed_array,
            sampling=self.sampling,
            origin=self.origin,
            units=self.units,
        )

        if return_results:
            return {
                "pca": {
                    "components_": components,
                    "explained_variance_": explained_variance,
                    "explained_variance_ratio_": explained_variance_ratio,
                },
                "components": components,
                "loadings": loadings_spatial,
                "explained_variance_ratio": explained_variance_ratio,
                "explained_variance": explained_variance,
                "reconstructed": reconstructed_data3d,
            }

    def _run_pca(self, data: NDArray | Any, n_components: int):
        array = np.asarray(data, dtype=float)
        n_samples, n_features = array.shape
        max_components = min(n_samples, n_features)
        if not 1 <= n_components <= max_components:
            raise ValueError(f"n_components={n_components} must be between 1 and {max_components}")

        mean = np.mean(array, axis=0)
        centered = torch.as_tensor(array - mean, dtype=torch.float64)
        _, s, vh = torch.linalg.svd(centered, full_matrices=False)

        components = vh[:n_components].cpu().numpy()
        loadings = (centered @ vh[:n_components].T).cpu().numpy()

        denom = max(n_samples - 1, 1)
        explained_variance = ((s[:n_components] ** 2) / denom).cpu().numpy()
        total_variance = torch.sum((s**2) / denom).item()
        explained_variance_ratio = (
            explained_variance / total_variance
            if total_variance > 0
            else np.zeros_like(explained_variance)
        )
        reconstructed = loadings @ components + mean

        return (
            components,
            loadings,
            explained_variance,
            explained_variance_ratio,
            reconstructed,
        )

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
        fig, (ax_scree, ax_components) = plt.subplots(1, 2, figsize=(12, 4))
        cumsum_var = np.cumsum(explained_variance_ratio)
        component_numbers = np.arange(1, len(explained_variance_ratio) + 1)

        ax_scree.bar(
            component_numbers,
            explained_variance_ratio * 100,
            alpha=0.6,
            label="Individual",
        )
        ax_scree.plot(component_numbers, cumsum_var * 100, "ro-", label="Cumulative")
        ax_scree.set_xlabel("Component Number")
        ax_scree.set_ylabel("Explained Variance (%)")
        ax_scree.set_title("Scree Plot")
        ax_scree.legend()
        ax_scree.grid(True, alpha=0.3)

        energy_sampling = float(self.sampling[0])
        energy_origin = float(self.origin[0])
        energy_axis = energy_origin + energy_sampling * np.arange(components.shape[1])

        for i in range(n_show):
            ax_components.plot(
                energy_axis,
                components[i],
                label=f"PC{i + 1} ({explained_variance_ratio[i] * 100:.1f}%)",
            )
        ax_components.set_xlabel("Energy")
        ax_components.set_ylabel("Component")
        ax_components.set_title("Principal Component Spectra")
        ax_components.legend()
        ax_components.grid(True, alpha=0.3)

        fig.suptitle("PCA Analysis")
        fig.tight_layout()
        plt.show()

        show_2d(
            [loadings[i] for i in range(n_show)],
            title=[
                f"Loading {i + 1} ({explained_variance_ratio[i] * 100:.1f}%)"
                for i in range(n_show)
            ],
            cmap="RdBu_r",
            cbar=True,
            scalebar={
                "sampling": float(self.sampling[1]),
                "units": str(self.units[1]),
            },
        )
        plt.show()

    def _calibrated_position_to_pixel(self, value, axis):
        if value is None:
            return None

        sampling = float(self.sampling[axis])
        if sampling == 0:
            raise ValueError(f"Cannot convert calibrated ROI on axis {axis}: sampling is zero")

        origin = float(self.origin[axis]) if hasattr(self, "origin") else 0.0
        return int(np.round((float(value) - origin) / sampling))

    def _calibrated_span_to_pixels(self, value, axis):
        if value is None:
            return None

        sampling = abs(float(self.sampling[axis]))
        if sampling == 0:
            raise ValueError(
                f"Cannot convert calibrated ROI span on axis {axis}: sampling is zero"
            )

        pixels = int(np.round(float(value) / sampling))
        if pixels < 1:
            raise ValueError(
                f"Calibrated ROI span on axis {axis} converts to {pixels} pixels; expected >= 1"
            )
        return pixels

    def _validate_roi_bounds(self, y, x, dy, dx):
        errs = []
        ymax = int(self.shape[1])
        xmax = int(self.shape[2])

        for name, val in (("y", y), ("x", x), ("dy", dy), ("dx", dx)):
            if val is None:
                errs.append(f"{name} is None (missing after normalization).")

        if errs:
            raise ValueError("Invalid ROI:\n - " + "\n - ".join(errs))

        if y < 0:
            errs.append(f"y={y} < 0")
        if x < 0:
            errs.append(f"x={x} < 0")
        if dy < 1:
            errs.append(f"dy={dy} < 1")
        if dx < 1:
            errs.append(f"dx={dx} < 1")

        if y >= ymax:
            errs.append(f"y start {y} out of bounds [0, {ymax - 1}]")
        if x >= xmax:
            errs.append(f"x start {x} out of bounds [0, {xmax - 1}]")

        end_y = y + dy
        end_x = x + dx
        if end_y > ymax:
            errs.append(f"y+dy = {end_y} exceeds height {ymax}")
        if end_x > xmax:
            errs.append(f"x+dx = {end_x} exceeds width {xmax}")

        if errs:
            raise ValueError("Invalid ROI:\n - " + "\n - ".join(errs))

    def _resolve_roi(self, roi=None, roi_cal=None):
        selector_count = int(roi is not None) + int(roi_cal is not None)
        if selector_count > 1:
            raise ValueError("Use only one ROI selector: roi or roi_cal")

        if roi is not None:
            roi_spec = roi
        elif roi_cal is not None:
            if len(roi_cal) == 2:
                y_cal, x_cal = roi_cal
                roi_spec = [
                    self._calibrated_position_to_pixel(y_cal, axis=1),
                    self._calibrated_position_to_pixel(x_cal, axis=2),
                ]
            elif len(roi_cal) == 4:
                y_cal, x_cal, dy_cal, dx_cal = roi_cal
                roi_spec = [
                    self._calibrated_position_to_pixel(y_cal, axis=1),
                    self._calibrated_position_to_pixel(x_cal, axis=2),
                    self._calibrated_span_to_pixels(dy_cal, axis=1),
                    self._calibrated_span_to_pixels(dx_cal, axis=2),
                ]
            else:
                raise ValueError("roi_cal must be [y, x] or [y, x, dy, dx]")
        else:
            roi_spec = None

        if roi_spec is None:
            y, x, dy, dx = 0, 0, int(self.shape[1]), int(self.shape[2])
        elif len(roi_spec) == 2:
            y, x = roi_spec
            y, x, dy, dx = int(y), int(x), 1, 1
        elif len(roi_spec) == 4:
            y_val, x_val, dy_val, dx_val = roi_spec
            y = 0 if y_val is None else int(y_val)
            x = 0 if x_val is None else int(x_val)
            dy = int(self.shape[1]) - y if dy_val is None else int(dy_val)
            dx = int(self.shape[2]) - x if dx_val is None else int(dx_val)
        else:
            raise ValueError(
                "ROI must be None, [y, x], or [y, x, dy, dx]. Use one selector: roi or roi_cal"
            )

        self._validate_roi_bounds(y, x, dy, dx)
        return y, x, dy, dx

    def calculate_mean_spectrum(
        self,
        roi=None,
        energy_range=None,
        ignore_range=None,
        mask=None,
        attach_mean_spectrum=True,
        roi_cal=None,
        normalize=True,
    ):
        """Calculate a spectrum from a spatial ROI.

        Parameters
        ----------
        normalize : bool, optional
            If ``True``, average over the ROI pixels.  If ``False``, sum counts
            over the ROI pixels.
        """
        y, x, dy, dx = self._resolve_roi(roi=roi, roi_cal=roi_cal)

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

            arr = arr[mask, y : y + dy, x : x + dx]  # select masked energies and ROI
            if arr.shape[0] > 0:
                spec = arr.mean(axis=(1, 2)) if normalize else arr.sum(axis=(1, 2))
            else:
                spec = np.zeros(0)
            E = E[mask]  # Mask the energy axis as well
        else:
            spec = np.empty(self.shape[0], dtype=float)
            for k in range(self.shape[0]):
                img = np.asarray(self.array[k], dtype=float)
                roi_data = img[y : y + dy, x : x + dx]
                if roi_data.size == 0:
                    raise ValueError("ROI is empty; check y/x/dy/dx.")
                spec[k] = roi_data.mean() if normalize else roi_data.sum()

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
        roi_cal=None,
        energy_range=None,
        mask=None,
        intensity_range=None,
        **kwargs,
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
        mask : array, optional
            Boolean mask for pixel selection.
        intensity_range : 2-tuple, None
            If not None, sets intensity range on spectrum plot
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

        y, x, dy, dx = self._resolve_roi(roi=roi, roi_cal=roi_cal)

        spec = self.calculate_mean_spectrum(
            roi=roi,
            roi_cal=roi_cal,
            energy_range=energy_range,
            mask=mask,
            normalize=False,
        )

        dE = float(self.sampling[0])
        E0 = float(self.origin[0]) if hasattr(self, "origin") else 0.0
        E = E0 + dE * np.arange(self.shape[0])

        if mask is not None:
            E = E[np.asarray(mask, dtype=bool)]

        if energy_range is not None:
            indices = np.where((E >= energy_range[0]) & (E <= energy_range[1]))[0]
            E = E[indices]

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

        map_title = f"Integrated Intensity Map{title_suffix}"
        show_2d(
            sum_img,
            figax=(fig, ax_img),
            title=map_title,
            cmap="viridis",
            cbar=True,
            show_ticks=True,
            scalebar={
                "sampling": float(self.sampling[1]),
                "units": str(self.units[1]),
            },
            **kwargs,
        )
        # Highlight the ROI with a rectangle
        rect = Rectangle(
            (x - 0.5, y - 0.5), dx, dy, linewidth=2, edgecolor="red", facecolor="none", alpha=0.8
        )
        ax_img.add_patch(rect)

        # RIGHT PLOT: Show spectrum
        ax_spec.plot(E, spec, linewidth=1.5, color="k")
        if self.dataset_type == "eds":
            ax_spec.set_xlabel("Energy (keV)")
        else:
            ax_spec.set_xlabel("Energy (eV)")
        ax_spec.set_ylabel("Intensity")
        ax_spec.set_title(f"Spectrum from ROI [{y}:{y + dy}, {x}:{x + dx}]")
        ax_spec.grid(True, alpha=0.1)
        if intensity_range is not None:
            ax_spec.set_ylim([intensity_range[0], intensity_range[1]])

        fig.tight_layout()
        return fig, (ax_img, ax_spec)

    def refline(
        self,
        elements,
        ax=None,
        energy=None,
        energy_range=None,
        linestyle=":",
        linewidth=1.2,
        alpha=0.35,
        show_text=True,
    ):
        """Overlay reference lines for selected element specifiers on a spectrum axis.

        This is plotting-only and does not modify auto-identification behavior.
        """
        if elements is None:
            raise ValueError("elements must be specified")
        if energy is not None and energy_range is not None:
            raise ValueError("Specify either energy or energy_range, not both")

        all_info = type(self)._ensure_element_info()

        specs = type(self)._normalize_element_specs(elements)
        if len(specs) == 0:
            raise ValueError("elements must contain at least one selector")

        if ax is None:
            ax = plt.gca()

        if energy_range is not None:
            if len(energy_range) != 2:
                raise ValueError("energy_range must be [min_energy, max_energy]")
            e_min = float(min(energy_range[0], energy_range[1]))
            e_max = float(max(energy_range[0], energy_range[1]))
        elif energy is not None:
            tol = max(2.0 * abs(float(self.sampling[0])), 1e-9)
            center = float(energy)
            e_min = center - tol
            e_max = center + tol
        else:
            xlim = ax.get_xlim()
            e_min = float(min(xlim))
            e_max = float(max(xlim))

        artists = []
        labels = []
        energies = []

        y_top = ax.get_ylim()[1]
        y_text = y_top - 0.08 * max(abs(y_top), 1e-12)

        for spec in specs:
            tokens = str(spec).split()
            if len(tokens) == 0:
                continue

            element_key = type(self)._resolve_element_key(all_info, tokens[0])
            if element_key is None:
                continue

            selectors = tokens[1:]
            selected_lines = type(self)._select_lines(all_info.get(element_key, {}), selectors)

            for line_name, line_info in selected_lines.items():
                if not isinstance(line_info, dict):
                    continue

                energy_value = line_info.get("energy (keV)")
                if energy_value is None:
                    energy_value = line_info.get("onset_energy (eV)")
                if energy_value is None:
                    energy_value = line_info.get("energy")
                if energy_value is None:
                    continue

                try:
                    line_energy = float(energy_value)
                except (TypeError, ValueError):
                    continue

                if not (e_min <= line_energy <= e_max):
                    continue

                label = f"{element_key} {line_name}"
                line_artist = ax.axvline(
                    line_energy,
                    linestyle=linestyle,
                    linewidth=linewidth,
                    alpha=alpha,
                    label=label,
                )
                artists.append(line_artist)
                labels.append(label)
                energies.append(line_energy)

                if show_text:
                    text_artist = ax.text(
                        line_energy,
                        y_text,
                        label,
                        rotation=90,
                        ha="center",
                        va="top",
                        alpha=min(1.0, alpha + 0.2),
                        clip_on=True,
                    )
                    artists.append(text_artist)

        return {"ax": ax, "artists": artists, "labels": labels, "energies": np.asarray(energies)}

    def show_energy_window_map(
        self,
        energy_window=None,
        roi=None,
        roi_cal=None,
        mask=None,
        cmap="viridis",
        show=True,
    ):
        """Show a spatial map integrated over a selected energy window.

        This is a complementary view to ``show_mean_spectrum``:
        - ``show_mean_spectrum`` answers *what energies are present*.
        - ``show_energy_window_map`` answers *where a chosen energy range is present*.

        Parameters
        ----------
        energy_window : list[float] | tuple[float, float] | None
            Energy interval [emin, emax] to integrate. If None, use the
            full calibrated energy range of the dataset.
        roi : list | tuple | None, optional
            ROI as ``[y, x]`` or ``[y, x, dy, dx]`` (with ``None`` defaults),
            used only for overlay rectangle.
        mask : array-like | None, optional
            Optional boolean mask over energy channels. If provided, it is
            combined with ``energy_window``.
        cmap : str, optional
            Matplotlib colormap for the map.
        show : bool, optional
            If True, call ``plt.show()``.

        Returns
        -------
        tuple
            ``(fig, (ax_map, ax_spec), energy_map)`` where ``energy_map`` is the integrated 2D array.
        """
        y, x, dy, dx = self._resolve_roi(roi=roi, roi_cal=roi_cal)
        has_roi_overlay = any(val is not None for val in (roi, roi_cal))

        dE = float(self.sampling[0])
        E0 = float(self.origin[0]) if hasattr(self, "origin") else 0.0
        E = E0 + dE * np.arange(self.shape[0])

        if energy_window is None:
            emin = float(np.min(E))
            emax = float(np.max(E))
        else:
            if len(energy_window) != 2:
                raise ValueError("energy_window must be [min_energy, max_energy]")

            emin = float(energy_window[0])
            emax = float(energy_window[1])
            if not np.isfinite(emin) or not np.isfinite(emax) or emin >= emax:
                raise ValueError(
                    "Invalid energy_window. Expected [min_energy, max_energy] with min < max"
                )

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

        spec = self.calculate_mean_spectrum(
            roi=roi,
            roi_cal=roi_cal,
            mask=mask,
            attach_mean_spectrum=False,
        )
        if mask is not None:
            E_spec = E[mask]
        else:
            E_spec = E

        unit_label = "keV" if str(self.dataset_type).lower() == "eds" else "eV"
        fig, (ax_map, ax_spec) = plt.subplots(1, 2, figsize=(12, 4))
        show_2d(
            energy_map,
            figax=(fig, ax_map),
            title=f"Energy-Window Map [{emin:.3f}, {emax:.3f}] {unit_label}",
            cmap=cmap,
            cbar=True,
            show_ticks=True,
            scalebar={
                "sampling": float(self.sampling[1]),
                "units": str(self.units[1]),
            },
        )

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
            ax_map.add_patch(rect)

        ax_spec.plot(E_spec, spec, linewidth=1.5, color="k")
        ax_spec.axvspan(emin, emax, color="orange", alpha=0.2, label="Selected window")
        ax_spec.set_xlabel(f"Energy ({unit_label})")
        ax_spec.set_ylabel("Intensity")
        ax_spec.set_title(f"Spectrum from ROI [{y}:{y + dy}, {x}:{x + dx}]")
        ax_spec.grid(True, alpha=0.1)
        ax_spec.legend(loc="best")

        fig.tight_layout()

        if show:
            plt.show()

        return fig, (ax_map, ax_spec), energy_map

    # BACKGROND SUBTRACTION

    def subtract_background(
        self,
        roi=None,
        energy_range=None,
        ignore_range=None,
        mask=None,
        target_edge=None,
        window_size=10,
        method="powerlaw",
        return_dataset=True,
        attach_spectrum=True,
        fit_mode="global",
        kernel_width=1,
        show=True,
        return_background=False,
    ):
        """
        Subtract fitted background from a 3D spectroscopy dataset.

        Parameters
        ----------
        fit_mode : {"global", "local"}, optional
            ``"global"`` fits one background to the ROI mean spectrum and subtracts
            it from every probe position. ``"local"`` fits a background at each
            probe position from the average spectrum of its nearest spatial
            neighbors.
        kernel_width : int, optional
            Number of nearest spatial neighbors to average for each local
            background fit. The current pixel is included. Used only when
            ``fit_mode="local"``.
        show : bool, optional
            If True, plot the mean raw spectrum, fitted background, and
            background-subtracted spectrum.
        return_background : bool, optional
            If True, return ``(dataset, background_cube)`` when ``return_dataset``
            is True, otherwise return the background cube.

        Returns
        -------
        Dataset3dspectroscopy or tuple or ndarray or None
            Background-subtracted dataset by default. If ``return_background`` is
            True, also returns the fitted background cube.
        """

        del ignore_range
        from quantem.spectroscopy import Dataset3deds, Dataset3deels

        fit_mode = str(fit_mode).lower()
        if fit_mode not in {"global", "local"}:
            raise ValueError("fit_mode must be 'global' or 'local'")

        E, indices = self._background_energy_axis_and_indices(energy_range, mask)
        array3d = np.asarray(self.array, dtype=float)[indices, :, :]
        y, x, dy, dx = self._resolve_roi(roi=roi)

        if fit_mode == "global":
            input_spectrum = array3d[:, y : y + dy, x : x + dx].mean(axis=(1, 2))
            background = self._fit_background_spectrum(
                input_spectrum,
                E,
                method=method,
                target_edge=target_edge,
                window_size=window_size,
            )
            background_cube = np.broadcast_to(background[:, None, None], array3d.shape)
        else:
            background_cube = self._fit_local_background_cube(
                array3d,
                E,
                method=method,
                target_edge=target_edge,
                window_size=window_size,
                kernel_width=kernel_width,
            )

        spec3D_subtracted = np.maximum(array3d - background_cube, 0)
        input_mean_spectrum = array3d[:, y : y + dy, x : x + dx].mean(axis=(1, 2))
        background_mean_spectrum = background_cube[:, y : y + dy, x : x + dx].mean(axis=(1, 2))
        subtracted_mean_spectrum = spec3D_subtracted[:, y : y + dy, x : x + dx].mean(axis=(1, 2))

        if attach_spectrum:
            self.add_spectrum_to_data(subtracted_mean_spectrum, E)

        if show:
            self._plot_background_subtraction(
                E,
                input_mean_spectrum,
                background_mean_spectrum,
                subtracted_mean_spectrum,
                fit_mode=fit_mode,
            )

        dataset_type = str(self.dataset_type).lower()
        if dataset_type == "eds":
            dataset_class = Dataset3deds
        elif dataset_type == "eels":
            dataset_class = Dataset3deels
        else:
            raise ValueError(f"Unsupported spectroscopy dataset_type {self.dataset_type!r}")

        output_origin = np.array(self.origin, dtype=float, copy=True)
        output_origin[0] = E[0]

        if return_dataset:
            subtracted_dataset = dataset_class.from_array(
                array=spec3D_subtracted,
                sampling=self.sampling,
                origin=output_origin,
                units=self.units,
            )
            if return_background:
                background_dataset = dataset_class.from_array(
                    array=np.array(background_cube, copy=True),
                    sampling=self.sampling,
                    origin=output_origin,
                    units=self.units,
                )
                return subtracted_dataset, background_dataset
            return subtracted_dataset

        if return_background:
            return background_cube

        print("Notice: no 3D dataset was returned")

    def _background_energy_axis_and_indices(self, energy_range, mask):
        E = np.asarray(self.energy_axis, dtype=float)
        selected = np.ones(E.shape, dtype=bool)

        if energy_range is not None:
            if len(energy_range) != 2:
                raise ValueError("energy_range must be [min_energy, max_energy]")
            e_min = float(energy_range[0])
            e_max = float(energy_range[1])
            if e_min >= e_max:
                raise ValueError("Invalid energy range parameter.")
            if e_max < E[0] or e_min > E[-1]:
                raise ValueError("Energy range parameter is outside of data bounds.")
            e_min = max(e_min, float(E[0]))
            e_max = min(e_max, float(E[-1]))
            selected &= (E >= e_min) & (E <= e_max)

        if mask is not None:
            mask = np.asarray(mask, dtype=bool)
            if mask.shape != E.shape:
                raise ValueError(
                    f"Mask shape {mask.shape} does not match energy axis shape {E.shape}"
                )
            selected &= mask

        if not np.any(selected):
            raise ValueError("No energy channels selected. Adjust energy_range or mask")

        indices = np.where(selected)[0]
        return E[indices], indices

    def _fit_background_spectrum(self, spectrum, energy_axis, method, target_edge, window_size):
        dataset_type = str(self.dataset_type).lower()
        spectrum = np.asarray(spectrum, dtype=float)

        if dataset_type == "eds":
            return self.calculate_background_powerlaw(spectrum)

        if dataset_type != "eels":
            raise ValueError(f"Unsupported spectroscopy dataset_type {self.dataset_type!r}")

        method = str(method).lower()
        if method == "iterative":
            return np.full_like(spectrum, float(self.calculate_background_iterative(spectrum)))
        if method != "powerlaw":
            raise ValueError("EELS background method must be 'powerlaw' or 'iterative'")
        if target_edge is None:
            raise ValueError("target_edge is required for EELS powerlaw background fitting")

        return self._fit_eels_powerlaw_background(
            spectrum,
            np.asarray(energy_axis, dtype=float),
            target_edge=target_edge,
            window_size=window_size,
        )

    def _fit_eels_powerlaw_background(self, spectrum, energy_axis, target_edge, window_size):
        from scipy.optimize import curve_fit

        if window_size < 10 or window_size > 30:
            raise ValueError("Invalid window size. Please input a value of between 10 and 30.")

        target_edge = float(target_edge)
        if target_edge < energy_axis[0] or target_edge > energy_axis[-1]:
            raise ValueError("Target edge is outside of energy range.")

        window_minE = (target_edge - 5) - target_edge * (float(window_size) / 100)
        window_maxE = target_edge - 5
        if window_minE < energy_axis[0]:
            raise ValueError(
                "Insufficient pre-edge background fitting region for this target edge "
                "and window size within given energy range."
            )

        window_indices = np.where((energy_axis >= window_minE) & (energy_axis <= window_maxE))[0]
        if len(window_indices) < 2:
            raise ValueError("Insufficient points in EELS pre-edge background fitting window.")

        window_E = energy_axis[window_indices]
        window_I = np.asarray(spectrum, dtype=float)[window_indices]

        def powerlaw_function(E, A, r):
            return A * (E ** (-r))

        popt, _ = curve_fit(powerlaw_function, window_E, window_I, maxfev=2000)
        return powerlaw_function(energy_axis, popt[0], popt[1])

    def _fit_local_background_cube(
        self,
        array3d,
        energy_axis,
        method,
        target_edge,
        window_size,
        kernel_width,
    ):
        from scipy.spatial import cKDTree

        n_energy, ny, nx = array3d.shape
        n_pixels = ny * nx
        try:
            n_neighbors = int(kernel_width)
        except (TypeError, ValueError) as exc:
            raise TypeError("kernel_width must be an integer") from exc
        if n_neighbors < 1:
            raise ValueError("kernel_width must be >= 1")
        n_neighbors = min(n_neighbors, n_pixels)

        yy, xx = np.indices((ny, nx))
        coords = np.column_stack((yy.reshape(-1), xx.reshape(-1)))
        _, neighbor_indices = cKDTree(coords).query(coords, k=n_neighbors)
        if n_neighbors == 1:
            neighbor_indices = neighbor_indices[:, None]

        spectra = np.moveaxis(array3d, 0, -1).reshape(n_pixels, n_energy)
        background = np.empty_like(spectra)

        for pixel_index, neighbors in enumerate(neighbor_indices):
            local_spectrum = spectra[neighbors].mean(axis=0)
            try:
                background[pixel_index] = self._fit_background_spectrum(
                    local_spectrum,
                    energy_axis,
                    method=method,
                    target_edge=target_edge,
                    window_size=window_size,
                )
            except Exception as exc:
                y, x = divmod(pixel_index, nx)
                raise RuntimeError(f"Background fit failed at pixel ({y}, {x})") from exc

        return background.reshape(ny, nx, n_energy).transpose(2, 0, 1)

    def _plot_background_subtraction(
        self,
        energy_axis,
        input_spectrum,
        background_spectrum,
        subtracted_spectrum,
        fit_mode,
    ):
        fig, (ax_specbacksub) = plt.subplots(1, 1, figsize=(12, 4))

        ax_specbacksub.plot(energy_axis, input_spectrum, linewidth=1.2, label="Input")
        ax_specbacksub.plot(energy_axis, background_spectrum, linewidth=1.2, label="Background")
        ax_specbacksub.plot(
            energy_axis,
            subtracted_spectrum,
            linewidth=1.5,
            label="Background-subtracted",
        )
        if self.dataset_type == "eds":
            ax_specbacksub.set_xlabel("Energy (keV)")
        else:
            ax_specbacksub.set_xlabel("Energy (eV)")
        ax_specbacksub.set_ylabel("Intensity")
        ax_specbacksub.set_title(f"Background-subtracted spectrum from ROI ({fit_mode})")
        ax_specbacksub.grid(True, alpha=0.1)
        ax_specbacksub.legend()

        fig.tight_layout()
        plt.show()

    @property
    def energy_axis(self):
        energy_axis = np.arange(self.shape[0]) * self.sampling[0] + self.origin[0]
        return energy_axis
