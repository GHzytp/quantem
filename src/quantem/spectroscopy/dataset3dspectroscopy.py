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
    def load_element_info(
        cls,
    ):
        """Load element database for EDS (X-ray lines) or EELS (binding energies)."""
        class_type = str(getattr(cls, "dataset_type", "")).strip().lower()
        if class_type == "eels":
            path = "eels_binding_energies.json"
        elif class_type == "eds":
            path = getattr(cls, "element_info_path", "x_ray_lines.csv")
        else:
            path = getattr(cls, "element_info_path", "x_ray_lines.csv")

        if cls.element_info is not None:
            # don't reload if already loaded
            return
        base_dir = os.path.dirname(os.path.abspath(__file__))
        full_path = os.path.join(base_dir, path)
        if str(path).lower().endswith(".csv"):
            cls.element_info = load_xray_lines_database(full_path)
        else:
            with open(full_path, "r") as f:
                cls.element_info = json.load(f)["elements"]

    @classmethod
    def load_atomic_weights(cls):
        """Load atomic weights table from CSV once per class."""
        if cls.atomic_weights is not None:
            return

        base_dir = os.path.dirname(os.path.abspath(__file__))
        full_path = os.path.join(base_dir, cls.atomic_weights_path)
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
    def _resolve_spectral_source(source: str, dataset_type: str) -> str:
        source_norm = source.strip().lower()
        if source_norm not in {"auto", "xray", "eels"}:
            raise ValueError("source must be one of: 'auto', 'xray', 'eels'")
        if source_norm == "auto":
            return "eels" if dataset_type.strip().lower() == "eels" else "xray"
        return source_norm

    @staticmethod
    def _build_spectral_table(
        element: str, source: str, rows: list[tuple[str, float, float]], precision: int
    ) -> str:
        unit = "eV" if source == "eels" else "keV"
        title = f"{element} {'EELS edges' if source == 'eels' else 'X-ray lines'}"

        display_rows = [
            (
                feature,
                f"{energy:.{precision}f}",
                "" if not np.isfinite(weight) else f"{weight:.{precision}f}",
            )
            for feature, energy, weight in rows
        ]
        show_weight = any(weight for _, _, weight in display_rows)

        feature_w = max(len("Feature"), *(len(feature) for feature, _, _ in display_rows))
        energy_header = f"Energy ({unit})"
        energy_w = max(len(energy_header), *(len(energy) for _, energy, _ in display_rows))

        columns = [("Feature", feature_w, "<"), (energy_header, energy_w, ">")]
        if show_weight:
            weight_w = max(len("Weight"), *(len(weight) for _, _, weight in display_rows))
            columns.append(("Weight", weight_w, ">"))

        header = "  ".join(f"{name:{align}{width}}" for name, width, align in columns)
        lines = [title, header, "-" * len(header)]
        for feature, energy, weight in display_rows:
            values = [feature, energy]
            if show_weight:
                values.append(weight)
            line = "  ".join(
                f"{value:{align}{width}}" for value, (_, width, align) in zip(values, columns)
            )
            lines.append(line)
        return "\n".join(lines)

    def _print_spectral_table(
        self,
        element: str,
        source: str,
        sort_by: str = "energy",
        ascending: bool = True,
        precision: int = 4,
    ) -> str:
        table = self.format_spectral_features_table(
            element=element,
            source=source,
            sort_by=sort_by,
            ascending=ascending,
            precision=precision,
        )
        print(table)
        return table

    def format_spectral_features_table(
        self,
        element: str,
        source: str = "auto",
        sort_by: str = "energy",
        ascending: bool = True,
        precision: int = 4,
    ) -> str:
        """Format X-ray lines or EELS edges for one element as a simple text table."""
        if type(self).element_info is None:
            type(self).load_element_info()

        all_info = type(self).element_info or {}
        if element not in all_info:
            available = sorted(all_info.keys())
            msg = f"Element '{element}' not found."
            if available:
                msg += f" Available examples: {', '.join(available[:10])}"
            raise ValueError(msg)

        source_norm = type(self)._resolve_spectral_source(
            source=source, dataset_type=str(getattr(type(self), "dataset_type", ""))
        )

        energy_keys = (
            ("energy (eV)", "onset (eV)", "edge (eV)", "energy")
            if source_norm == "eels"
            else ("energy (keV)", "energy_keV", "energy")
        )
        rows = []
        for feature_name, info in all_info[element].items():
            if isinstance(info, dict):
                energy_raw = next(
                    (info.get(k) for k in energy_keys if info.get(k) is not None), None
                )
                weight_raw = info.get("weight", info.get("strength"))
            else:
                energy_raw = info
                weight_raw = None

            try:
                energy = float(energy_raw)
            except (TypeError, ValueError):
                continue

            try:
                weight = float(weight_raw) if weight_raw is not None else np.nan
            except (TypeError, ValueError):
                weight = np.nan

            rows.append((str(feature_name), energy, weight))

        if not rows:
            return f"{element}: no spectral features found."

        sort_index = {
            "feature": 0,
            "line": 0,
            "edge": 0,
            "energy": 1,
            "weight": 2,
            "strength": 2,
        }.get(sort_by.strip().lower())
        if sort_index is None:
            raise ValueError("sort_by must be one of: feature/line/edge, energy, weight/strength")
        rows.sort(key=lambda r: r[sort_index], reverse=not ascending)
        return type(self)._build_spectral_table(
            element=element, source=source_norm, rows=rows, precision=precision
        )

    def format_xray_lines_table(
        self,
        element: str,
        sort_by: str = "energy",
        ascending: bool = True,
        precision: int = 4,
    ) -> str:
        """Backward-compatible wrapper for X-ray lines."""
        return self.format_spectral_features_table(
            element=element,
            source="xray",
            sort_by=sort_by,
            ascending=ascending,
            precision=precision,
        )

    def format_eels_edges_table(
        self,
        element: str,
        sort_by: str = "energy",
        ascending: bool = True,
        precision: int = 4,
    ) -> str:
        """Format EELS edge entries for one element."""
        return self.format_spectral_features_table(
            element=element,
            source="eels",
            sort_by=sort_by,
            ascending=ascending,
            precision=precision,
        )

    def print_xray_lines(
        self,
        element: str,
        sort_by: str = "energy",
        ascending: bool = True,
        precision: int = 4,
    ) -> str:
        """Print and return a formatted table of X-ray lines for one element."""
        return self._print_spectral_table(
            element=element,
            source="xray",
            sort_by=sort_by,
            ascending=ascending,
            precision=precision,
        )

    def print_eels_edges(
        self,
        element: str,
        sort_by: str = "energy",
        ascending: bool = True,
        precision: int = 4,
    ) -> str:
        """Print and return a formatted table of EELS edges for one element."""
        return self._print_spectral_table(
            element=element,
            source="eels",
            sort_by=sort_by,
            ascending=ascending,
            precision=precision,
        )

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

        def _normalize_specs(specs):
            if isinstance(specs, str):
                return [s.strip() for s in specs.split(",") if s.strip()]
            if isinstance(specs, (list, tuple, set)):
                out = []
                for spec in specs:
                    out.extend([s.strip() for s in str(spec).split(",") if s.strip()])
                return out
            raise TypeError("elements must be a string or a sequence of strings")

        def _resolve_element_key(all_info, token):
            token_norm = str(token).strip().lower()
            for key in all_info.keys():
                if str(key).lower() == token_norm:
                    return key
            return None

        def _select_lines(line_dict, selectors):
            if not isinstance(line_dict, dict):
                return {}
            if selectors is None or len(selectors) == 0:
                return dict(line_dict)

            selector_norm = [str(sel).strip().lower() for sel in selectors if str(sel).strip()]
            selected = {}
            for line_name, line_info in line_dict.items():
                line_norm = str(line_name).strip().lower()
                if any(line_norm == sel or line_norm.startswith(sel) for sel in selector_norm):
                    selected[line_name] = line_info
            return selected

        all_info = type(self).element_info
        if all_info is None:
            return

        specs = _normalize_specs(elements)
        if self.model_elements is None:
            self.model_elements = {}

        for spec in specs:
            tokens = str(spec).split()
            if len(tokens) == 0:
                continue

            element_key = _resolve_element_key(all_info, tokens[0])
            if element_key is None:
                continue

            selectors = tokens[1:]
            selected_lines = _select_lines(all_info[element_key], selectors)
            if len(selected_lines) == 0:
                continue

            if len(selectors) == 0:
                self.model_elements[element_key] = selected_lines
            else:
                existing = self.model_elements.get(element_key)
                if not isinstance(existing, dict):
                    existing = {}
                existing.update(selected_lines)
                self.model_elements[element_key] = existing

        if len(self.model_elements) == 0:
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

        def _normalize_specs(specs):
            if isinstance(specs, str):
                return [s.strip() for s in specs.split(",") if s.strip()]
            if isinstance(specs, (list, tuple, set)):
                out = []
                for spec in specs:
                    out.extend([s.strip() for s in str(spec).split(",") if s.strip()])
                return out
            raise TypeError("elements must be a string or a sequence of strings")

        def _resolve_element_key(model_elements, token):
            token_norm = str(token).strip().lower()
            for key in model_elements.keys():
                if str(key).lower() == token_norm:
                    return key
            return None

        specs = _normalize_specs(elements)
        for spec in specs:
            tokens = str(spec).split()
            if len(tokens) == 0:
                continue

            element_key = _resolve_element_key(self.model_elements, tokens[0])
            if element_key is None:
                continue

            selectors = [str(token).strip().lower() for token in tokens[1:] if str(token).strip()]
            if len(selectors) == 0:
                self.model_elements.pop(element_key, None)
                continue

            lines_info = self.model_elements.get(element_key)
            if not isinstance(lines_info, dict):
                self.model_elements.pop(element_key, None)
                continue

            for line_name in list(lines_info.keys()):
                line_norm = str(line_name).strip().lower()
                if any(line_norm == sel or line_norm.startswith(sel) for sel in selectors):
                    lines_info.pop(line_name, None)

            if len(lines_info) == 0:
                self.model_elements.pop(element_key, None)

        if len(self.model_elements) == 0:
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

    # QUANTIFICATION -----------------------------------------------

    def quantify_composition(
        self, roi=None, elements=None, k_factors=None, method="cliff_lorimer", mask=None
    ):
        """
        Quantify elemental composition from EDS spectrum using Cliff-Lorimer approach.

        The Cliff-Lorimer equation relates atomic fractions to X-ray intensities:
        CA/CB = kAB * (IA/IB)

        Parameters
        ----------
        roi : list or tuple, optional
            Region of interest as [y, x, dy, dx]. If None, uses full image.
        elements : list, required
            List of element symbols to quantify (e.g., ['Pt', 'Co']).
        k_factors : dict or array-like, required
            K-factors for the quantified elements.
            - dict format: {'Pt': 1.0, 'Co': 1.23}
            - array/list format: [1.0, 1.23] mapped in the same order as ``elements``
                        - per-shell dict format:
                            {'Pt': {'K': 0, 'L': 1.12, 'M': 0}, 'Co': {'K': 1.23, 'L': 0, 'M': 0}}
                            where 0 means shell unavailable.
        method : str, optional
            Quantification method. Currently supports 'cliff_lorimer'.
        mask : array, optional
            Boolean mask for energy channel selection.

        Returns
        -------
        dict : Composition results containing:
            - 'atomic_percent': dict of element -> atomic %
            - 'weight_percent': dict of element -> weight %
            - 'intensities': dict of element -> integrated intensity
            - 'k_factors': dict of k-factors used

        Examples
        --------
        # With dictionary k-factors
        k_factors = {'Pt': 1.0, 'Co': 1.23}
        comp = dataset.quantify_composition(elements=['Pt', 'Co'], k_factors=k_factors)

        # With array-like k-factors (same order as elements)
        comp = dataset.quantify_composition(elements=['Pt', 'Co'], k_factors=[1.0, 1.23])

        # With per-shell k-factors (0 means unavailable shell)
        shell_kf = {
            'Pt': {'K': 0, 'L': 1.12, 'M': 0},
            'Co': {'K': 1.23, 'L': 0, 'M': 0},
        }
        comp = dataset.quantify_composition(elements=['Pt', 'Co'], k_factors=shell_kf)

        # Access results
        print(f"Pt: {comp['atomic_percent']['Pt']:.1f} at%")
        print(f"Co: {comp['atomic_percent']['Co']:.1f} at%")
        """

        # Input validation
        if elements is None or len(elements) < 2:
            raise ValueError("At least 2 elements required for quantification")

        # Load element info if not available
        if type(self).element_info is None:
            type(self).load_element_info()

        # Extract spectrum from ROI
        spectrum_data = self._extract_spectrum_for_quantification(roi, mask)
        spec = spectrum_data["spectrum"]
        E = spectrum_data["energy"]

        # Determine max usable energy from the actual dataset
        max_energy = float(E.max()) if len(E) > 0 else 20.0

        # Determine shell for each element and validate/normalize k-factors
        if k_factors is None:
            raise ValueError("Must provide k_factors as a dict or array-like")

        element_shells = self._determine_element_shells(elements, max_energy)
        k_factors = self._normalize_k_factors(elements, k_factors, element_shells)

        # Get X-ray line intensities for each element using the correct shell
        intensities = {}
        for element in elements:
            shell = element_shells.get(element, "K")  # Default to K if not determined
            intensity = self._integrate_element_intensity(element, spec, E, shell)
            intensities[element] = intensity

        # Apply Cliff-Lorimer quantification
        if method == "cliff_lorimer":
            results = self._cliff_lorimer_quantification(
                elements, intensities, k_factors, method, roi
            )
        else:
            raise ValueError(f"Unknown quantification method: {method}")

        return results

    def _extract_spectrum_for_quantification(self, roi, mask):
        """Extract spectrum data for quantification (similar to show_mean_spectrum)."""
        # Parse ROI (reuse logic from show_mean_spectrum)
        if roi is None:
            y, x, dy, dx = 0, 0, int(self.shape[1]), int(self.shape[2])
        elif len(roi) == 2:
            y, x, dy, dx = int(roi[0]), int(roi[1]), 1, 1
        elif len(roi) == 4:
            y_val, x_val, dy_val, dx_val = roi
            y = 0 if y_val is None else int(y_val)
            x = 0 if x_val is None else int(x_val)
            dy = int(self.shape[1]) - y if dy_val is None else int(dy_val)
            dx = int(self.shape[2]) - x if dx_val is None else int(dx_val)
        else:
            raise ValueError("roi must be None, [y, x], or [y, x, dy, dx]")

        # Energy axis
        dE = float(self.sampling[0])
        E0 = float(self.origin[0]) if hasattr(self, "origin") else 0.0
        E = E0 + dE * np.arange(self.shape[0])

        # Extract spectrum with mask handling
        if mask is not None:
            mask = np.asarray(mask, dtype=bool)
            if mask.shape != (self.shape[0],):
                raise ValueError(
                    f"Mask shape {mask.shape} doesn't match energy axis ({self.shape[0]},)"
                )
            arr = np.asarray(self.array, dtype=float)[mask, :, :]
            spec = arr.sum(axis=(1, 2)) if arr.shape[0] > 0 else np.zeros(0)
            E = E[mask]
        else:
            spec = np.empty(self.shape[0], dtype=float)
            for k in range(self.shape[0]):
                img = np.asarray(self.array[k], dtype=float)
                roi_data = img[y : y + dy, x : x + dx]
                if roi_data.size == 0:
                    raise ValueError("ROI is empty")
                spec[k] = roi_data.mean()

        return {"spectrum": spec, "energy": E}

    def _integrate_element_intensity(self, element, spectrum, energy, shell="K"):
        """Integrate X-ray intensity for a specific element using characteristic lines from the specified shell.

        Parameters
        ----------
        element : str
            Element symbol
        spectrum : array
            Spectrum intensities
        energy : array
            Energy axis in keV
        shell : str
            X-ray shell to use: 'K', 'L', or 'M'
        """
        all_info = type(self).element_info
        if element not in all_info:
            raise ValueError(f"Element {element} not found in database")

        total_intensity = 0.0
        element_lines = all_info[element]

        # Filter lines by the specified shell (K, L, or M)
        # For K-shell: Ka, Kb lines
        # For L-shell: La, Lb, Lg lines
        # For M-shell: Ma, Mb lines
        shell_lines = []
        for line_name, info in element_lines.items():
            line_energy = info["energy (keV)"]
            line_weight = info["weight"]

            # Check if line belongs to the specified shell
            if shell == "K" and ("Ka" in line_name or "Kb" in line_name):
                shell_lines.append((line_weight, line_energy, line_name))
            elif shell == "L" and ("La" in line_name or "Lb" in line_name or "Lg" in line_name):
                shell_lines.append((line_weight, line_energy, line_name))
            elif shell == "M" and ("Ma" in line_name or "Mb" in line_name):
                shell_lines.append((line_weight, line_energy, line_name))

        # Sort by weight (highest first) and ignore lines beyond detector range
        shell_lines = [(w, e, n) for w, e, n in shell_lines if e <= 12.0]
        shell_lines.sort(reverse=True)

        # Use top 3 most intense lines from the specified shell for integration
        for weight, line_energy, line_name in shell_lines[:3]:
            if weight > 0.1:  # Only significant lines
                # Find integration window around the line
                # Use +/- 0.1 keV window or adaptive based on energy resolution
                window_width = max(0.1, line_energy * 0.01)  # 1% of energy or 0.1 keV minimum

                # Find energy indices for integration
                energy_mask = (energy >= line_energy - window_width) & (
                    energy <= line_energy + window_width
                )

                if np.any(energy_mask):
                    # Simple background subtraction: use linear interpolation at edges
                    line_spectrum = spectrum[energy_mask]
                    if len(line_spectrum) > 2:
                        # Background level from edges of integration window
                        bg_level = (line_spectrum[0] + line_spectrum[-1]) / 2
                        # Integrate above background, weighted by line intensity
                        net_intensity = np.sum(line_spectrum - bg_level) * weight
                        total_intensity += max(0, net_intensity)  # No negative intensities

        return total_intensity

    def _determine_element_shells(self, elements, max_energy):
        """Determine the appropriate X-ray shell (K, L, or M) for each element based on available lines.

        Parameters
        ----------
        elements : list
            List of element symbols
        max_energy : float
            Maximum energy in keV from the dataset
        """
        all_info = type(self).element_info
        element_shells = {}

        for element in elements:
            if element not in all_info:
                element_shells[element] = "K"  # Default
                continue

            element_lines = all_info[element]

            # Check which X-ray series is present AND within usable energy range
            has_usable_k_lines = any(
                ("Ka" in line or "Kb" in line) and info["energy (keV)"] <= max_energy
                for line, info in element_lines.items()
            )
            has_usable_l_lines = any(
                ("La" in line or "Lb" in line or "Lg" in line)
                and info["energy (keV)"] <= max_energy
                for line, info in element_lines.items()
            )
            has_usable_m_lines = any(
                ("Ma" in line or "Mb" in line) and info["energy (keV)"] <= max_energy
                for line, info in element_lines.items()
            )

            # Prioritize K-lines, then L-lines, then M-lines (only if within usable range)
            if has_usable_k_lines:
                element_shells[element] = "K"
            elif has_usable_l_lines:
                element_shells[element] = "L"
            elif has_usable_m_lines:
                element_shells[element] = "M"
            else:
                element_shells[element] = "K"  # Default fallback

        return element_shells

    def _normalize_k_factors(self, elements, k_factors, element_shells=None):
        """Normalize k-factors input to a dict keyed by element symbol.

        Supports:
        - scalar dict per element, e.g. {'Pt': 1.0, 'Co': 1.23}
        - array-like values aligned with ``elements`` order
        - per-shell dict per element, e.g. {'Pt': {'K': 0, 'L': 1.1, 'M': 0}}
          where non-positive values are treated as unavailable shell entries.
        """
        shell_order = ("K", "L", "M")
        if element_shells is None:
            element_shells = {}

        def _to_positive_float_or_none(value):
            try:
                parsed = float(value)
            except (TypeError, ValueError):
                return None
            if not np.isfinite(parsed) or parsed <= 0:
                return None
            return parsed

        def _extract_shell_value(elem, shell_values):
            preferred_shell = str(element_shells.get(elem, "K")).upper()
            candidate_order = [preferred_shell] + [s for s in shell_order if s != preferred_shell]

            normalized_shell_values = {}
            for shell in shell_order:
                raw_value = shell_values.get(shell)
                if raw_value is None:
                    raw_value = shell_values.get(shell.lower())
                normalized_shell_values[shell] = _to_positive_float_or_none(raw_value)

            for shell in candidate_order:
                value = normalized_shell_values.get(shell)
                if value is not None:
                    return value

            raise ValueError(f"k_factors['{elem}'] has no usable positive shell value in K/L/M")

        if isinstance(k_factors, dict):
            missing = [elem for elem in elements if elem not in k_factors]
            if missing:
                raise ValueError(f"k_factors is missing elements: {missing}")

            normalized = {}
            for elem in elements:
                raw_entry = k_factors[elem]

                if isinstance(raw_entry, dict):
                    value = _extract_shell_value(elem, raw_entry)
                else:
                    try:
                        value = float(raw_entry)
                    except (TypeError, ValueError):
                        raise TypeError(
                            f"k_factors['{elem}'] must be numeric or a dict with K/L/M entries"
                        )
                    if not np.isfinite(value) or value <= 0:
                        raise ValueError(f"k_factors['{elem}'] must be a positive finite number")

                normalized[elem] = value
            return normalized

        if isinstance(k_factors, (str, bytes)):
            raise TypeError("k_factors must be a dict or array-like of numeric values")

        try:
            values = list(k_factors)
        except TypeError:
            raise TypeError("k_factors must be a dict or array-like of numeric values")

        if len(values) != len(elements):
            raise ValueError(
                "Array-like k_factors length must match elements length "
                f"({len(values)} != {len(elements)})"
            )

        normalized = {}
        for elem, raw_value in zip(elements, values):
            try:
                value = float(raw_value)
            except (TypeError, ValueError):
                raise TypeError(f"k_factors value for '{elem}' must be numeric")
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"k_factors value for '{elem}' must be a positive finite number")
            normalized[elem] = value

        return normalized

    def _cliff_lorimer_quantification(self, elements, intensities, k_factors, method, roi):
        """Apply Cliff-Lorimer quantification method."""
        # Cliff-Lorimer equation: CA/CB = kAB * (IA/IB)
        # For multiple elements: CA = kA * IA / SUM(ki * Ii)

        # Calculate weighted intensities
        weighted_sum = 0.0
        weighted_intensities = {}

        for element in elements:
            weighted_intensity = k_factors[element] * intensities[element]
            weighted_intensities[element] = weighted_intensity
            weighted_sum += weighted_intensity

        # Calculate atomic percentages
        atomic_percent = {}
        for element in elements:
            if weighted_sum > 0:
                atomic_percent[element] = (weighted_intensities[element] / weighted_sum) * 100.0
            else:
                atomic_percent[element] = 0.0

        # Calculate weight percentages (requires atomic weights)
        if type(self).atomic_weights is None:
            type(self).load_atomic_weights()
        atomic_weights = type(self).atomic_weights or {}

        missing_weights = [element for element in elements if element not in atomic_weights]
        if missing_weights:
            raise ValueError(
                f"Atomic weights not found for elements: {missing_weights}. "
                "Use valid element symbols (e.g., 'Fe', 'Au', 'Te')."
            )

        # Convert atomic % to weight %
        weight_sum = 0.0
        for element in elements:
            atomic_wt = atomic_weights[element]
            weight_sum += (atomic_percent[element] / 100.0) * atomic_wt

        weight_percent = {}
        for element in elements:
            if weight_sum > 0:
                atomic_wt = atomic_weights[element]
                weight_percent[element] = (
                    (atomic_percent[element] / 100.0) * atomic_wt / weight_sum
                ) * 100.0
            else:
                weight_percent[element] = 0.0

        # Print summary in Cliff-Lorimer format
        print("\n=== Quantification (Cliff-Lorimer) ===")
        print(f"ROI: {'Full image' if roi is None else roi}")
        print(f"Elements: {', '.join(elements)}")

        print("\nRaw Intensities:")
        for elem in elements:
            print(f"  {elem}: {intensities[elem]:.2f}")

        print("\nk-factors:")
        for elem in elements:
            print(f"  {elem}: {k_factors[elem]:.2f}")

        print("\nAtomic %:")
        for elem in elements:
            print(f"  {elem}: {atomic_percent[elem]:.1f} at%")

        print("\nWeight %:")
        for elem in elements:
            print(f"  {elem}: {weight_percent[elem]:.1f} wt%")

        return {
            "atomic_percent": atomic_percent,
            "weight_percent": weight_percent,
            "intensities": intensities,
            "k_factors": k_factors,
            "method": "cliff_lorimer",
        }

    def _find_best_element_combinations(self, peak_energies, peak_intensities, tolerance=0.15):
        """
        Find the best combination of elements that explains the detected peaks using a cost function.

        Parameters:
        peak_energies : array-like
            Detected peak positions in keV
        peak_intensities : array-like
            Detected peak intensities
        tolerance : float, default 0.15
            Energy tolerance for peak matching in keV

        Returns:
        set : Set of element symbols that best explain the detected peaks
        """
        from itertools import combinations

        # Get element database
        all_info = type(self).element_info
        if all_info is None:
            return set()

        # Consider combinations of 1-4 elements (reasonable for most samples)
        best_elements = set()
        best_score = float("inf")

        # Get commonly analyzed elements (general EDS candidates)
        general_elements = [
            "Fe",
            "Pt",
            "Cu",
            "C",
            "O",
            "Ni",
            "Co",
            "Al",
            "Si",
            "Ti",
            "Cr",
            "Mn",
            "Au",
            "Ag",
            "Zn",
            "Ca",
            "K",
            "Na",
            "Mg",
        ]
        available_elements = [el for el in general_elements if el in all_info]

        # Test combinations of different sizes
        top_combinations = []  # Store combinations for analysis
        for num_elements in range(1, min(5, len(available_elements) + 1)):
            for element_combo in combinations(available_elements, num_elements):
                score = self._calculate_element_combo_score(
                    element_combo, peak_energies, peak_intensities, all_info, tolerance
                )

                top_combinations.append((score, element_combo))

                if score < best_score:
                    best_score = score
                    best_elements = set(element_combo)
        return best_elements

    def _calculate_element_combo_score(
        self, element_combo, peak_energies, peak_intensities, all_info, tolerance
    ):
        """
        Calculate a cost function score for a given combination of elements.
        Lower scores are better.

        Strategy: Prioritize explaining ALL major peaks with the FEWEST elements.
        Only accept combinations that explain most peaks with high-weight lines.
        """
        score = 0.0
        explained_peaks = {}  # peak_idx -> (matched_distance, line_weight, element)

        # For each detected peak, find the BEST match in the element combination
        for i, (peak_energy, peak_intensity) in enumerate(zip(peak_energies, peak_intensities)):
            best_match_distance = float("inf")
            best_line_weight = 0.0
            best_element = None
            found_match = False

            # Check all elements in the combination
            for element in element_combo:
                if element in all_info:
                    for line_name, line_info in all_info[element].items():
                        line_energy = line_info["energy (keV)"]
                        line_weight = line_info.get("weight", 0.5)
                        distance = abs(peak_energy - line_energy)

                        # Only consider lines with significant weight (major lines only)
                        if line_weight > 0.2 and distance <= tolerance:
                            # Update best match if this line is better
                            if distance < best_match_distance or (
                                distance == best_match_distance and line_weight > best_line_weight
                            ):
                                best_match_distance = distance
                                best_line_weight = line_weight
                                best_element = element
                                found_match = True

            if found_match:
                explained_peaks[i] = (best_match_distance, best_line_weight, best_element)
                # Penalty for distance (prefer closer matches)
                score += best_match_distance * 10.0
                # Bonus for high-weight lines (major lines score much better)
                score -= best_line_weight * 3.0
            else:
                # HEAVY penalty for unexplained peaks - this is the key constraint
                score += 50.0

        # Primary objective: explain ALL detected peaks
        unexplained_peaks = len(peak_energies) - len(explained_peaks)
        if unexplained_peaks > 0:
            score += unexplained_peaks * 100.0  # Very high penalty for unexplained peaks

        # Secondary objective: prefer simpler explanations (fewer elements)
        score += len(element_combo) * 5.0

        # Tertiary objective: prefer explanations with multiple peaks per element
        # This avoids one-off false matches and encourages coherent solutions
        peaks_per_element = {}
        for peak_idx, (dist, weight, elem) in explained_peaks.items():
            if elem not in peaks_per_element:
                peaks_per_element[elem] = []
            peaks_per_element[elem].append((dist, weight))

        # Bonus if each element explains multiple peaks (coherence - more likely correct)
        for elem, matches in peaks_per_element.items():
            if len(matches) > 1:
                # Elements with 2+ peak matches are much more likely correct
                score -= len(matches) * 2.0

        return score

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


Dataset3dspectroscopy.load_element_info()
