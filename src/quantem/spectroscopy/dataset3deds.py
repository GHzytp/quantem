import re
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from matplotlib.lines import Line2D
from numpy.typing import NDArray
from scipy.optimize import curve_fit
from scipy.signal import find_peaks, peak_prominences, peak_widths

from quantem.core.visualization import show_2d
from quantem.spectroscopy import Dataset3dspectroscopy
from quantem.spectroscopy.spectroscopy_models import (
    EDSModel,
    GaussianPeaks,
    PolynomialBackground,
    abundance_smoothness_l2,
    build_element_basis,
    eds_data_loss,
    inverse_softplus,
    polynomial_energy_basis,
)


class Dataset3deds(Dataset3dspectroscopy):
    """An EDS dataset class that inherits from Dataset3dspectroscopy.

    This class represents a scanning transmission electron microscopy (STEM) dataset,
    where the data consists of a 3D array with dimensions (scan_row, scan_col, energy).
    The first two dimensions represent real space sampling, while the last dimension
    represents the energy axis.

    """

    element_info = None
    element_info_path = "x_ray_lines.csv"

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
        """Initialize a 3D EDS dataset."""
        super().__init__(
            array=array,
            name=name,
            origin=origin,
            sampling=sampling,
            units=units,
            signal_units=signal_units,
            _token=_token,
        )
        self.dataset_type = "eds"

    @staticmethod
    def _normalize_specs(specs, param_name="spec", allow_none=False):
        """Parse specs into a flat list of stripped strings."""
        if specs is None:
            if allow_none:
                return None
            raise TypeError(f"{param_name} must be a string or sequence of strings")
        if isinstance(specs, str):
            return [s.strip() for s in specs.split(",") if s.strip()]
        if isinstance(specs, (list, tuple, set)):
            return [s.strip() for item in specs for s in str(item).split(",") if s.strip()]
        raise TypeError(f"{param_name} must be a string or sequence of strings")

    @staticmethod
    def _normalize_token(text):
        """Return a lowercase alphanumeric-only token for fuzzy matching."""
        return re.sub(r"[^a-z0-9]", "", str(text).lower())

    @staticmethod
    def _ordered_element_keys(all_info):
        """Return element keys sorted longest-first for greedy prefix matching."""
        return sorted(map(str, all_info), key=lambda k: (-len(k), k))

    @classmethod
    def _resolve_element_from_label(cls, label, ordered_elements):
        """Extract the element name from a line label like 'FeKa1'."""
        label = str(label)
        for element in ordered_elements:
            if label.startswith(element):
                return element
        m = re.match(r"^[A-Z][a-z]?", label)
        return m.group(0) if m else None

    @classmethod
    def _ensure_element_info(cls):
        """Load element X-ray line data if not already cached."""
        if cls.element_info is None:
            cls.load_element_info()
        return cls.element_info or {}

    @classmethod
    def _normalize_element_info(cls, combine_close_peaks=True, energy_threshold_ev=15):
        """Normalize EDS X-ray lines and optionally merge unresolved line families."""
        if not isinstance(cls.element_info, dict):
            return cls.element_info

        threshold_kev = float(energy_threshold_ev) / 1000.0

        def line_family(line_name):
            canonical = cls._canonical_line_name(line_name).strip()
            match = re.match(r"^([A-Za-z]+)", canonical)
            return match.group(1) if match else canonical

        def normalized_line_name(line_name):
            canonical = cls._canonical_line_name(line_name).strip()
            match = re.match(r"^([A-Za-z]+)\d+(?:,\d+)+$", canonical)
            return match.group(1) if match else canonical

        def unique_name(lines, name):
            if name not in lines:
                return name
            idx = 2
            while f"{name}__{idx}" in lines:
                idx += 1
            return f"{name}__{idx}"

        def merged_info(entries):
            weights = np.asarray([entry["weight"] for entry in entries], dtype=float)
            energies = np.asarray([entry["energy"] for entry in entries], dtype=float)
            weight_sum = float(np.sum(weights))
            if weight_sum > 0.0:
                energy = float(np.sum(energies * weights) / weight_sum)
            else:
                energy = float(np.mean(energies))
            return {"energy (keV)": energy, "weight": weight_sum}

        normalized_info = {}
        for element, lines in cls.element_info.items():
            if not isinstance(lines, dict):
                normalized_info[element] = lines
                continue

            entries_by_family = {}
            normalized_lines = {}
            for line_name, line_info in lines.items():
                if not isinstance(line_info, dict):
                    continue
                try:
                    energy = float(line_info.get("energy (keV)", line_info.get("energy")))
                except (TypeError, ValueError):
                    continue
                try:
                    weight = float(line_info.get("weight", 0.0))
                except (TypeError, ValueError):
                    weight = 0.0

                entry = {
                    "line": normalized_line_name(line_name),
                    "family": line_family(line_name),
                    "energy": energy,
                    "weight": weight,
                }
                entries_by_family.setdefault(entry["family"], []).append(entry)

            for family, entries in entries_by_family.items():
                entries = sorted(entries, key=lambda entry: entry["energy"])
                if not combine_close_peaks:
                    for entry in entries:
                        name = unique_name(normalized_lines, entry["line"])
                        normalized_lines[name] = {
                            "energy (keV)": entry["energy"],
                            "weight": entry["weight"],
                        }
                    continue

                clusters = []
                current = []
                for entry in entries:
                    if not current or entry["energy"] - current[0]["energy"] <= threshold_kev:
                        current.append(entry)
                    else:
                        clusters.append(current)
                        current = [entry]
                if current:
                    clusters.append(current)

                for cluster in clusters:
                    name = family if len(cluster) > 1 else cluster[0]["line"]
                    name = unique_name(normalized_lines, name)
                    normalized_lines[name] = merged_info(cluster)

            normalized_info[element] = dict(
                sorted(
                    normalized_lines.items(),
                    key=lambda item: (item[1]["energy (keV)"], item[0]),
                )
            )

        cls.element_info = normalized_info
        return cls.element_info

    @classmethod
    def _parse_element_selectors(cls, specs, *, allow_none=False, param_name="spec"):
        """Parse element/line specifiers into a dict of {element: set_of_suffixes | None}."""
        tokens = cls._normalize_specs(specs, param_name=param_name, allow_none=allow_none)
        if tokens is None:
            return None

        ordered = cls._ordered_element_keys(cls._ensure_element_info())
        out: dict[str, set[str] | None] = {}
        for raw in tokens:
            compact = re.sub(r"[\s_-]+", "", str(raw).strip())
            if not compact:
                continue
            element = next((k for k in ordered if compact.lower().startswith(k.lower())), None)
            if element is None:
                raise ValueError(f"Could not resolve element from specifier '{raw}'")
            suffix = compact[len(element) :]
            out.setdefault(element, None if not suffix else set())
            if suffix and out[element] is not None:
                out[element].add(suffix)
        return out or None

    @staticmethod
    def _canonical_line_name(line_name: str) -> str:
        """Strip any suffix after '__' from a line name."""
        return str(line_name).split("__", 1)[0]

    @classmethod
    def _iter_selected_lines(cls, element: str, suffix: str, *, raw_spec: str):
        """Yield (line_name, line_info) pairs matching an element and optional suffix."""
        lines = cls._ensure_element_info().get(element) or {}
        if not lines:
            raise ValueError(f"No X-ray lines found for element '{element}'")
        if not suffix:
            yield from lines.items()
            return

        suffix = cls._normalize_token(suffix)
        exact, prefix = [], []
        for line_name, line_info in lines.items():
            token = cls._normalize_token(cls._canonical_line_name(line_name))
            if token == suffix:
                exact.append((line_name, line_info))
            if token.startswith(suffix):
                prefix.append((line_name, line_info))
        matches = exact or prefix
        if not matches:
            raise ValueError(
                f"No X-ray lines matched specifier '{raw_spec}' for element '{element}'"
            )
        yield from matches

    @classmethod
    def _group_labels_by_element(cls, labels: list[str]):
        """Group line labels by their parent element."""
        ordered = cls._ordered_element_keys(cls._ensure_element_info())
        grouped: dict[str, list[str]] = {}
        for lbl in sorted(map(str, labels)):
            element = cls._resolve_element_from_label(lbl, ordered)
            if element:
                grouped.setdefault(element, []).append(lbl)
        return grouped

    @classmethod
    def _select_labels(
        cls, selector: str, *, labels: list[str], labels_by_element: dict[str, list[str]]
    ):
        """Return labels matching a selector string (exact, element, or prefix)."""
        selector = str(selector).strip()
        if not selector:
            return []

        lower_map = {lbl.lower(): lbl for lbl in labels}
        if selector.lower() in lower_map:
            return [lower_map[selector.lower()]]

        elem_map = {elem.lower(): elem for elem in labels_by_element}
        if selector.lower() in elem_map:
            return list(labels_by_element[elem_map[selector.lower()]])

        token = cls._normalize_token(selector)
        return [lbl for lbl in labels if cls._normalize_token(lbl).startswith(token)]

    @staticmethod
    def _line_shell(line_name: str) -> str:
        """Return the shell letter ('K', 'L', 'M', or '?') for a line name."""
        line_name = str(line_name).upper()
        return (
            "K"
            if line_name.startswith("K")
            else "L"
            if line_name.startswith("L")
            else "M"
            if line_name.startswith("M")
            else "?"
        )

    @staticmethod
    def _peak_confidence(
        snr_value: float, line_weight: float, distance_value: float, tolerance: float
    ) -> float:
        """Compute a confidence score for a peak-to-line match."""
        sigma = max(float(tolerance) / 3.0, 1e-9)
        return (
            np.log1p(max(float(snr_value), 0.0))
            * max(float(line_weight), 0.0)
            * np.exp(-0.5 * (float(distance_value) / sigma) ** 2)
        )

    @staticmethod
    def _line_matches_selector(line_name: str, selector: str) -> bool:
        """Check whether a line name matches a shell or substring selector."""
        line = str(line_name).strip().lower()
        selector = str(selector).strip().lower()
        return line.startswith(selector) if selector in {"k", "l", "m"} else selector in line

    @classmethod
    def _line_allowed_for_element(
        cls, element_name: str, line_name: str, edge_filters=None
    ) -> bool:
        """Return True if the line passes the edge filter for its element."""
        selectors = None if edge_filters is None else edge_filters.get(str(element_name))
        return selectors is None or any(
            cls._line_matches_selector(line_name, token) for token in selectors
        )

    def _get_spectrum_images(self, method="integration"):
        """Retrieve cached spectrum images for the given method."""
        return {
            "integration": getattr(self, "_spectrum_images", None),
            "fit": getattr(self, "_spectrum_images_pytorch", None),
        }.get(method)

    @staticmethod
    def _shell_preference_factor(shell_name: str) -> float:
        """Return a down-weighting factor for M-shell lines."""
        return 0.72 if shell_name == "M" else 1.0

    @staticmethod
    def _merge_edge_filters(requested, saved):
        """Merge requested and saved edge filters, unioning selectors per element."""
        if requested and saved:
            merged = dict(saved)
            for element, selectors in requested.items():
                current = merged.get(element)
                merged[element] = (
                    None if current is None or selectors is None else set(current).union(selectors)
                )
            return merged
        return requested or saved

    @staticmethod
    def _estimate_snr_thresholds(snr_values, floor=None, snr_threshold=None):
        """Auto-estimate SNR floors/thresholds from the peak SNR distribution."""
        snr_values = np.asarray(snr_values, dtype=float)
        snr_values = snr_values[np.isfinite(snr_values)]

        if floor is None:
            if snr_values.size:
                # Robust quantile floor: center near the middle/high-middle SNR
                # band so the floor tracks "visible" peaks without being pulled
                # down by noise tails or up by a few extreme peaks.
                q30, q40, q50, q60 = np.percentile(snr_values, [30, 40, 50, 60])
                floor = 0.5 * float(q40 + q50)
                floor = float(np.clip(floor, q30, q60))
                floor = max(0.0, floor)
            else:
                floor = 8.0
        else:
            floor = float(floor)

        if snr_threshold is None:
            if snr_values.size:
                high = snr_values[snr_values >= floor]
                high = high if high.size else snr_values
                # Keep auto-threshold independent from the requested display
                # count (peaks). `peaks` should control only how many detected
                # peaks are shown, not which peaks are detected.
                anchor = np.sort(high)[::-1][: min(high.size, 40)]
                med, q75, q90 = np.percentile(anchor, [50, 75, 90])
                snr_threshold = float(
                    np.clip(max(med, 0.7 * q75, 2.5 * floor), max(2.5 * floor, floor), q90)
                )
            else:
                snr_threshold = max(4.0 * floor, 30.0)
        else:
            snr_threshold = float(snr_threshold)

        return floor, snr_threshold

    def x_ray_lookup(
        self, spec: str | list[str] | tuple[str, ...] | set[str]
    ) -> tuple[np.ndarray, np.ndarray, list[str]]:
        """Look up X-ray line energies, weights, and labels.

        Parameters
        ----------
        spec : str | sequence[str]
            One or more element/line specifiers.  Accepted formats include
            element names (``'Fe'``), element + shell (``'Fe K'``), and
            element + line (``'Fe Ka1'``).  Comma-separated strings are
            split automatically.

        Returns
        -------
        energies : ndarray
            1-D array of line energies in keV, sorted by energy.
        weights : ndarray
            Corresponding tabulated line weights (0--1).
        labels : list[str]
            Human-readable labels such as ``'FeKa1'``.

        Raises
        ------
        ValueError
            If no lines match the specifier(s).
        """
        info = type(self)._ensure_element_info()
        ordered = type(self)._ordered_element_keys(info)
        specs = type(self)._normalize_specs(spec, param_name="spec")

        rows: list[tuple[str, float, float]] = []
        for raw in specs:
            compact = re.sub(r"[\s_-]+", "", str(raw).strip())
            if not compact:
                continue
            element = next((k for k in ordered if compact.lower().startswith(k.lower())), None)
            if element is None:
                raise ValueError(f"Could not resolve element from specifier '{raw}'")
            suffix = compact[len(element) :]
            for line_name, line_info in type(self)._iter_selected_lines(
                element, suffix, raw_spec=str(raw)
            ):
                if not isinstance(line_info, dict):
                    continue
                try:
                    energy = float(line_info.get("energy (keV)", line_info.get("energy")))
                except (TypeError, ValueError):
                    continue
                try:
                    weight = float(line_info.get("weight", 0.0))
                except (TypeError, ValueError):
                    weight = 0.0
                rows.append(
                    (f"{element}{type(self)._canonical_line_name(line_name)}", energy, weight)
                )

        if not rows:
            raise ValueError(f"No X-ray lines matched specifier(s): {specs}")

        unique = sorted(
            {(lbl, round(float(e), 12), round(float(w), 12)) for lbl, e, w in rows},
            key=lambda t: (t[1], -t[2], t[0]),
        )
        return (
            np.asarray([e for _, e, _ in unique], dtype=float),
            np.asarray([w for _, _, w in unique], dtype=float),
            [lbl for lbl, _, _ in unique],
        )

    def generage_spectrum_images(self, elements=None, width=0.15, return_maps=False):
        """Generate spectrum images by integrating around X-ray line energies.

        For each matched X-ray line, sums the spectral intensity within an
        energy window of ``line_energy +/- width`` at every spatial pixel.
        Results are cached in ``self._spectrum_images`` for later use by
        :meth:`show_spectrum_images` and :meth:`quantify_composition_cliff_lorimer`.

        Parameters
        ----------
        elements : str | sequence[str] | None, optional
            Element/line specifiers (see :meth:`x_ray_lookup`).  If ``None``,
            uses ``self.model_elements``.
        width : float, optional
            Half-width of the integration window in keV.
        return_maps : bool, optional
            If ``True``, return ``(maps, labels)``.

        Returns
        -------
        tuple[ndarray, list[str]] | None
            Only returned when *return_maps* is ``True``.
        """
        if elements is None:
            if self.model_elements is None:
                raise ValueError("elements must be specified")
            elements = list(self.model_elements)

        energies, _, labels = self.x_ray_lookup(elements)
        keep = (energies > self.energy_axis.min()) & (energies < self.energy_axis.max())
        energies = energies[keep]
        labels = [label for label, ok in zip(labels, keep) if ok]

        mask = (self.energy_axis[:, None] > energies[None, :] - width) & (
            self.energy_axis[:, None] < energies[None, :] + width
        )
        # scan_row, scan_col, n_energy = self.array.shape
        # maps = (mask.astype(self.array.dtype).T @ self.array.reshape(n, -1)).reshape(
        #    mask.shape[1], scan_row, scan_col
        # )

        scan_row, scan_col, n_energy = self.array.shape
        maps = (
            mask.astype(self.array.dtype).T @ self.array.reshape(-1, n_energy).transpose()
        ).reshape(mask.shape[1], scan_row, scan_col)

        self._spectrum_images = {
            **getattr(self, "_spectrum_images", {}),
            **dict(zip(labels, maps)),
        }

        images, titles = self.show_spectrum_images(x_ray_lines=elements, return_maps=True)

        if return_maps:
            return images, titles

    def Integrate(self, spec, width=0.15, return_maps=False, show=True, **kwargs):
        """Integrate the spectrum around specified X-ray lines.

        Sums spectral intensity within ``line_energy +/- width`` for each
        selector.  By default, displays the resulting map(s).

        Parameters
        ----------
        spec : str | sequence[str]
            Element/line specifiers (see :meth:`x_ray_lookup`), e.g.
            ``'Fe Ka'`` or ``['Cu', 'Zn']``.
        width : float, optional
            Half-width of the integration window in keV.
        return_maps : bool, optional
            If ``True``, return the integrated maps.
        show : bool, optional
            If ``True``, display the maps.
        **kwargs
            Forwarded to the plotting function (e.g. ``cmap``, ``roi``).

        Returns
        -------
        ndarray | dict[str, ndarray]
            Single map when one selector is given, otherwise a dict keyed by
            selector string.
        """
        width = float(width)
        specs = type(self)._normalize_specs(spec, param_name="spec")
        arr = np.asarray(self.array, dtype=float)
        energy_axis = np.asarray(self.energy_axis, dtype=float)
        energy_min, energy_max = float(energy_axis.min()), float(energy_axis.max())

        selector_masks, integrated_maps = {}, {}
        for selector in map(str, specs):
            line_energies, _, _ = self.x_ray_lookup(selector.strip())
            line_energies = line_energies[
                (line_energies >= energy_min) & (line_energies <= energy_max)
            ]
            if not len(line_energies):
                raise ValueError(
                    f"No X-ray lines for selector '{selector}' are within the dataset energy range"
                )

            mask = np.any(
                (energy_axis[:, None] >= line_energies[None, :] - width)
                & (energy_axis[:, None] <= line_energies[None, :] + width),
                axis=1,
            )
            selector_masks[selector] = mask
            integrated_maps[selector] = arr[:, :, mask].sum(axis=2)

        if show:
            cmap = kwargs.pop("cmap", "magma")
            if len(integrated_maps) == 1:
                selector = next(iter(integrated_maps))
                self.show_energy_window_map(
                    energy_window=[energy_min, energy_max],
                    roi=kwargs.pop("roi", None),
                    roi_cal=kwargs.pop("roi_cal", None),
                    mask=selector_masks[selector],
                    data_type=kwargs.pop("data_type", "eds"),
                    cmap=cmap,
                    show=True,
                )
            else:
                show_2d(
                    list(integrated_maps.values()),
                    title=list(integrated_maps),
                    cmap=cmap,
                    scalebar={"sampling": self.sampling[1], "units": self.units[1]},
                    **kwargs,
                )

        return (
            integrated_maps
            if return_maps or len(integrated_maps) != 1
            else next(iter(integrated_maps.values()))
        )

    def integrate(self, spec, width=0.15, return_maps=False, show=True, **kwargs):
        """Convenience wrapper for Integrate."""
        return self.Integrate(spec=spec, width=width, return_maps=return_maps, show=show, **kwargs)

    def show_spectrum_images(
        self, x_ray_lines=None, return_fig=False, return_maps=False, method="integration", **kwargs
    ):
        """Display cached spectrum images.

        Parameters
        ----------
        x_ray_lines : str | sequence[str] | None, optional
            Selectors to filter which images are shown.  If ``None``, one
            panel per element is displayed.
        return_fig : bool, optional
            If ``True``, return ``(fig, ax)``.
        method : {"integration", "fit"}, optional
            Which cache to read from: integration-based maps or PyTorch
            fit-based maps.
        **kwargs
            Forwarded to :func:`show_2d` (e.g. ``cmap``).

        Returns
        -------
        tuple[Figure, Axes] | None
            Only returned when *return_fig* is ``True``.

        Raises
        ------
        ValueError
            If no cached spectrum images exist for the chosen *method*.
        """
        spectrum_images = self._get_spectrum_images(method)
        if not spectrum_images:
            raise ValueError("No spectrum images found. Run generage_spectrum_images(...) first.")

        line_map = {str(k): np.asarray(v) for k, v in spectrum_images.items()}
        labels = list(line_map)
        labels_by_element = type(self)._group_labels_by_element(labels)

        def sum_maps(lbls):
            return np.sum([line_map[lbl] for lbl in lbls], axis=0)

        specs = type(self)._normalize_specs(x_ray_lines, param_name="x_ray_lines", allow_none=True)
        if not specs:
            titles = sorted(labels_by_element)
            images = [sum_maps(labels_by_element[t]) for t in titles]
        else:
            selected = [
                type(self)._select_labels(
                    str(raw), labels=labels, labels_by_element=labels_by_element
                )
                for raw in specs
            ]
            if any(not s for s in selected):
                bad = next(raw for raw, s in zip(specs, selected) if not s)
                raise ValueError(f"No spectrum images matched selector '{bad}'")
            images = [line_map[s[0]] if len(s) == 1 else sum_maps(s) for s in selected]
            titles = [s[0] if len(s) == 1 else str(raw).strip() for raw, s in zip(specs, selected)]

        fig, ax = show_2d(
            images,
            title=titles,
            cmap=kwargs.pop("cmap", "magma"),
            scalebar={"sampling": self.sampling[1], "units": self.units[1]},
            returnfig=True,
            **kwargs,
        )

        if return_fig and return_maps:
            return (fig, ax), (images, titles)
        elif return_fig:
            return fig, ax
        elif return_maps:
            return images, titles

    def _build_pytorch_spectrum_images(
        self, abundance_maps: np.ndarray, element_names: list[str] | tuple[str, ...]
    ) -> dict[str, np.ndarray]:
        """Convert per-element abundance maps into per-line spectrum images using weights."""
        maps = np.asarray(abundance_maps)
        if maps.ndim != 3:
            return {}

        line_maps = {}
        for i, element_name in enumerate(element_names):
            if i >= maps.shape[0]:
                break
            try:
                _, line_weights, line_labels = self.x_ray_lookup(str(element_name))
            except ValueError:
                continue
            element_map = np.asarray(maps[i], dtype=float)
            for weight, label in zip(line_weights, line_labels):
                line_maps[str(label)] = element_map * float(weight)
        return line_maps

    def quantify_composition_cliff_lorimer(
        self, k_factors, method="integration", return_maps=False, verbose=True
    ):
        """Quantify elemental composition using the Cliff-Lorimer thin-film method.

        Parameters
        ----------
        k_factors : dict[str, float]
            Mapping of element/line selectors to their k-factors, e.g.
            ``{'Fe K': 1.0, 'Cu K': 1.45}``.  At least two elements are
            required.
        method : {"integration", "fit"}, optional
            Which cached spectrum images to use for intensity extraction.
        return_maps : bool, optional
            If ``True``, include per-pixel atomic-percent and weight-percent
            maps in the returned dict.
        verbose : bool, optional
            If ``True``, print the quantification summary table.

        Returns
        -------
        dict
            Keys include ``atomic_percent``, ``weight_percent``,
            ``intensities``, ``weighted_intensities``, and
            ``summary_table``.  When *return_maps* is ``True``, also
            includes ``atomic_percent_maps`` and ``weight_percent_maps``.

        Raises
        ------
        ValueError
            If *k_factors* is empty, fewer than two elements are matched, or
            spectrum images are missing.
        """
        if not k_factors:
            raise ValueError("k_factors must be a non-empty dict")
        spectrum_images = self._get_spectrum_images(method)
        if not spectrum_images:
            raise ValueError("No spectrum images available for quantification")

        ordered_elements = type(self)._ordered_element_keys(type(self)._ensure_element_info())
        line_map = {str(k): np.asarray(v, dtype=float) for k, v in spectrum_images.items()}
        labels = list(line_map)
        labels_by_element = type(self)._group_labels_by_element(labels)

        def match(selector: str) -> list[str]:
            return type(self)._select_labels(
                selector, labels=labels, labels_by_element=labels_by_element
            )

        intensities, weighted_intensities = {}, {}
        selector_maps = {} if return_maps else None
        intensity_maps = {} if return_maps else None
        weighted_intensity_maps = {} if return_maps else None

        for selector, k_raw in k_factors.items():
            k_val = float(k_raw)
            sel_labels = match(str(selector).strip())
            if not sel_labels:
                raise ValueError(f"No spectrum images matched selector {selector!r}")

            matched_elements = {
                type(self)._resolve_element_from_label(lbl, ordered_elements) for lbl in sel_labels
            } - {None}
            if len(matched_elements) != 1:
                raise ValueError(
                    f"Selector {selector!r} matched multiple elements: {sorted(matched_elements)}"
                )
            element = next(iter(matched_elements))

            grouped_map = np.sum([line_map[lbl] for lbl in sel_labels], axis=0)
            intensity = float(grouped_map.sum())
            weighted = float(k_val * intensity)
            intensities[element] = intensities.get(element, 0.0) + intensity
            weighted_intensities[element] = weighted_intensities.get(element, 0.0) + weighted

            if return_maps:
                weighted_map = grouped_map * k_val
                selector_maps[str(selector)] = grouped_map
                intensity_maps[element] = intensity_maps.get(element, 0) + grouped_map
                weighted_intensity_maps[element] = (
                    weighted_intensity_maps.get(element, 0) + weighted_map
                )

        if len(weighted_intensities) < 2:
            raise ValueError("At least two elements are required for Cliff-Lorimer quantification")

        weighted_sum = sum(weighted_intensities.values())
        atomic_percent = {
            el: 100.0 * val / weighted_sum if weighted_sum > 0 else 0.0
            for el, val in weighted_intensities.items()
        }

        if type(self).atomic_weights is None:
            type(self).load_atomic_weights()
        atomic_weights = type(self).atomic_weights or {}
        missing = [el for el in atomic_percent if el not in atomic_weights]
        if missing:
            raise ValueError(f"Atomic weights not found for elements: {missing}")

        weight_sum = sum(
            (atomic_percent[el] / 100.0) * float(atomic_weights[el]) for el in atomic_percent
        )
        weight_percent = {
            el: (atomic_percent[el] / 100.0) * float(atomic_weights[el]) / weight_sum * 100.0
            if weight_sum > 0
            else 0.0
            for el in atomic_percent
        }

        ordered = sorted(weighted_intensities, key=weighted_intensities.get, reverse=True)
        table_text = "\n".join(
            [
                "Element  Intensity      Weighted Intensity    Atomic %    Weight %",
                "-------  -------------  --------------------  ----------  ----------",
                *[
                    f"{el:<7}  {intensities[el]:>13.3f}  {weighted_intensities[el]:>20.3f}  {atomic_percent[el]:>10.3f}  {weight_percent[el]:>10.3f}"
                    for el in ordered
                ],
            ]
        )
        result = {
            "intensities": intensities,
            "weighted_intensities": weighted_intensities,
            "atomic_percent": atomic_percent,
            "weight_percent": weight_percent,
            "summary_table": table_text,
        }
        if verbose:
            print(table_text)

        if return_maps:
            weighted_stack = np.stack(list(weighted_intensity_maps.values()), axis=0)
            weighted_sum_map = weighted_stack.sum(axis=0)
            atomic_percent_maps = {
                el: np.divide(
                    wmap * 100.0,
                    weighted_sum_map,
                    out=np.zeros_like(weighted_sum_map, dtype=float),
                    where=weighted_sum_map > 0,
                )
                for el, wmap in weighted_intensity_maps.items()
            }
            mass_maps = {
                el: atomic_percent_maps[el] / 100.0 * float(atomic_weights[el])
                for el in atomic_percent_maps
            }
            mass_sum_map = np.sum(np.stack(list(mass_maps.values()), axis=0), axis=0)
            weight_percent_maps = {
                el: np.divide(
                    mmap * 100.0,
                    mass_sum_map,
                    out=np.zeros_like(mass_sum_map, dtype=float),
                    where=mass_sum_map > 0,
                )
                for el, mmap in mass_maps.items()
            }
            result.update(
                {
                    "selector_maps": selector_maps,
                    "intensity_maps": intensity_maps,
                    "weighted_intensity_maps": weighted_intensity_maps,
                    "atomic_percent_maps": atomic_percent_maps,
                    "weight_percent_maps": weight_percent_maps,
                }
            )
        return result

    def clear_spectrum_images(self):
        """Clear cached integration-based spectrum images."""
        self._spectrum_images = {}

    def clear_spectrum_images_pytorch(self):
        """Clear cached PyTorch fit-based spectrum images."""
        self._spectrum_images_pytorch = {}

    def peak_autoid(
        self,
        roi=None,
        roi_cal=None,
        energy_range=None,
        elements=None,
        ignore_elements=None,
        ignore_range=None,
        tolerance=0.15,
        min_line_weight=0.0,
        mask=None,
        show_text=True,
        floor=None,
        snr_threshold=None,
        distance_threshold_for_sample=0.05,
        grid_peaks=None,
        peaks=15,
        mode=None,
        line=None,
        return_details=False,
    ):
        """Automatically identify elements from EDS peaks in the mean spectrum.

        Finds peaks in the spatially-averaged spectrum, matches them against a
        database of known X-ray line energies, and classifies elements as
        *detected* (high confidence) or *possible* (lower confidence).  Results
        are printed and overlaid on an interactive spectrum plot.

        Parameters
        ----------
        roi : sequence[int] | None, optional
            Pixel-coordinate ROI ``[y0, y1, x0, x1]`` used when computing the
            mean spectrum.  If ``None``, the full spatial extent is used.
        roi_cal : sequence[float] | None, optional
            Calibrated-coordinate ROI (same layout as *roi* but in physical
            units).
        energy_range : sequence[float] | None, optional
            Two-element energy window ``[emin, emax]`` in keV.  Peaks outside
            this range are ignored.
        elements : str | sequence[str] | None, optional
            Element or element-line specifiers to search for, e.g.
            ``'Fe'``, ``'Fe Ka'``, or ``['Cu', 'Zn K']``.  When provided,
            behaviour depends on *mode*.
        ignore_elements : str | sequence[str] | None, optional
            Elements to exclude from autodetection.
        ignore_range : sequence[float] | None, optional
            Energy range ``[emin, emax]`` whose peaks are ignored.  Defaults to
            ``[0, 0.25]`` keV to skip the noise floor.
        tolerance : float, optional
            Maximum energy difference in keV between a detected peak and a
            tabulated X-ray line for them to be considered a match.
            M-shell minor lines use ``tolerance * 0.5``.
        min_line_weight : float, optional
            Minimum tabulated line weight (0--1) for a line to be considered.
        mask : ndarray | None, optional
            Boolean spatial mask; only pixels where ``mask`` is ``True``
            contribute to the mean spectrum.
        show_text : bool, optional
            If ``True``, annotate matched peaks on the plot.
        floor : float | None, optional
            Minimum signal-to-noise ratio for a peak to be displayed.  If
            ``None``, estimated from robust middle quantiles (roughly between
            the 30th and 60th percentile of peak SNRs).
        snr_threshold : float | None, optional
            SNR above which a peak match counts as "strong" evidence for an
            element.  If ``None``, estimated automatically.
        distance_threshold_for_sample : float, optional
            Maximum energy distance (keV) for a match to qualify as a strong
            match (used together with *snr_threshold*).
        grid_peaks : dict | None, optional
            Mapping of ``{label: energy}`` for known grid/artifact peaks that
            should be flagged in the output.
        peaks : int, optional
            Maximum number of peaks to display.
        mode : {"elements_only", "elements_preferred", "autofill"} | None, optional
            Search strategy.  ``"elements_only"`` restricts matching to
            *elements*; ``"elements_preferred"`` boosts them but allows others;
            ``"autofill"`` (default when *elements* is ``None``) searches all
            elements.
        line : float | sequence[float] | None, optional
            Energy value(s) in keV for reference lines to draw on the spectrum
            plot, e.g. ``3.692`` or ``[3.692, 4.510]``.  Lines are drawn as
            dashed black vertical lines.
        return_details : bool, optional
            If ``True``, return a dict with detection details instead of the
            figure.

        Returns
        -------
        tuple[Figure, tuple[Axes, Axes]] | dict
            By default returns ``(fig, (ax_img, ax_spec))``.  When
            *return_details* is ``True``, returns a dict containing
            ``detected_elements``, ``element_confidence``, ``display_peaks``,
            ``peak_matches``, ``floor``, ``snr_threshold``, and the figure.
        """
        type(self)._ensure_element_info()
        all_info = type(self).element_info or {}
        grid_peaks = grid_peaks or {}
        ignore_range = [0, 0.25] if ignore_range is None else ignore_range
        ignored_elements = set(
            map(str, type(self)._normalize_specs(ignore_elements, allow_none=True) or [])
        )
        min_line_weight = max(float(min_line_weight), 0.0)

        requested = type(self)._parse_element_selectors(
            elements, allow_none=True, param_name="elements"
        )
        saved = {
            str(k): (set(map(str, v.keys())) if isinstance(v, dict) and v else None)
            for k, v in (getattr(self, "model_elements", {}) or {}).items()
        } or None
        edge_filters = requested if requested is not None else saved
        requested_elements = set(edge_filters) if edge_filters else None

        mode = (str(mode).strip().lower() if mode is not None else None) or (
            "elements_only" if requested_elements else "autofill"
        )
        search_elements = requested_elements if mode == "elements_only" else None
        preferred_elements = (
            set(map(str, requested_elements or [])) if mode == "elements_preferred" else set()
        )
        reference_elements = requested_elements

        fig, (ax_img, ax_spec) = self.show_mean_spectrum(
            roi=roi,
            roi_cal=roi_cal,
            energy_range=energy_range,
            mask=mask,
            data_type="eds",
            show=False,
        )
        spec = self.calculate_mean_spectrum(
            roi=roi,
            roi_cal=roi_cal,
            energy_range=energy_range,
            mask=mask,
        )

        E = np.asarray(self.energy_axis, dtype=float)

        # Keep the energy axis aligned with calculate_mean_spectrum filtering.
        if mask is not None:
            mask_arr = np.asarray(mask, dtype=bool)
            if mask_arr.shape != E.shape:
                raise ValueError(
                    f"Mask shape {mask_arr.shape} does not match energy axis shape {E.shape}."
                )
            E = E[mask_arr]

        if energy_range is not None:
            keep = (energy_range[0] <= E) & (E <= energy_range[1])
            E = E[keep]

        if len(spec) != len(E):
            raise ValueError(
                "Energy axis length does not match mean spectrum length after filtering. "
                f"Got len(E)={len(E)} and len(spec)={len(spec)}."
            )

        def in_ignore(energy):
            return len(ignore_range) == 2 and ignore_range[0] <= float(energy) <= ignore_range[1]

        peak_indices, props = find_peaks(spec, height=0, distance=5)
        peak_heights = props["peak_heights"]
        peak_proms = (
            peak_prominences(spec, peak_indices)[0]
            if len(peak_indices)
            else np.asarray([], dtype=float)
        )
        peak_width_samples = (
            peak_widths(spec, peak_indices, rel_height=0.5)[0]
            if len(peak_indices)
            else np.asarray([], dtype=float)
        )
        background_std = np.nanstd(spec[spec <= np.nanpercentile(spec, 50)])
        if not np.isfinite(background_std) or background_std <= 0:
            background_std = np.nanstd(spec)
        if not np.isfinite(background_std) or background_std <= 0:
            background_std = 1.0

        # Collapse shoulder peaks before SNR filtering.
        # Two adjacent peaks are treated as one if they are very close in energy
        # and the valley between them is shallow relative to the smaller peak.
        # This removes split-peak artifacts that tend to over-label broad peaks.
        def collapse_shoulder_peaks(indices, heights, prominences, widths):
            if len(indices) <= 1:
                return (
                    np.asarray(indices, dtype=int),
                    np.asarray(heights, dtype=float),
                    np.asarray(prominences, dtype=float),
                    np.asarray(widths, dtype=float),
                )

            energy_gap_limit = max(6.0 * float(self.sampling[2]), 0.14)
            min_valley_relief = 0.35
            min_height_ratio = 0.45

            keep = []
            i = 0
            while i < len(indices):
                best_idx = int(indices[i])
                best_h = float(heights[i])
                best_p = float(prominences[i])
                best_w = float(widths[i])
                j = i + 1

                while j < len(indices):
                    cand_idx = int(indices[j])
                    cand_h = float(heights[j])
                    cand_p = float(prominences[j])
                    cand_w = float(widths[j])
                    if float(E[cand_idx] - E[best_idx]) > energy_gap_limit:
                        break

                    lo, hi = sorted((best_idx, cand_idx))
                    if hi - lo <= 1:
                        valley = float(min(spec[lo], spec[hi]))
                    else:
                        valley = float(np.min(spec[lo : hi + 1]))

                    smaller = max(min(best_h, cand_h), 1e-12)
                    valley_relief = (smaller - valley) / smaller
                    height_ratio = min(best_h, cand_h) / max(best_h, cand_h)

                    # Not a clearly separated doublet -> merge shoulders.
                    if valley_relief < min_valley_relief or height_ratio < min_height_ratio:
                        if (cand_p > best_p) or (cand_p == best_p and cand_h > best_h):
                            best_idx, best_h, best_p, best_w = cand_idx, cand_h, cand_p, cand_w
                        j += 1
                        continue

                    break

                keep.append((best_idx, best_h, best_p, best_w))
                i = j

            out_idx = np.asarray([pk for pk, _, _, _ in keep], dtype=int)
            out_h = np.asarray([h for _, h, _, _ in keep], dtype=float)
            out_p = np.asarray([p for _, _, p, _ in keep], dtype=float)
            out_w = np.asarray([w for _, _, _, w in keep], dtype=float)
            order = np.argsort(out_idx)
            return out_idx[order], out_h[order], out_p[order], out_w[order]

        peak_indices, peak_heights, peak_proms, peak_width_samples = collapse_shoulder_peaks(
            peak_indices,
            peak_heights,
            peak_proms,
            peak_width_samples,
        )

        snr_values = np.asarray([height / background_std for height in peak_heights], dtype=float)
        floor, snr_threshold = type(self)._estimate_snr_thresholds(
            snr_values,
            floor,
            snr_threshold,
        )

        # Prominence filter in SNR units: suppress shoulder/noise artifacts that
        # may have acceptable height but do not form a distinct peak.
        prominence_snr = np.asarray(
            [float(p) / max(float(background_std), 1e-12) for p in peak_proms], dtype=float
        )

        def _local_noise_std(pk_idx):
            # Use local baseline variability so narrow doublets are not lost
            # when a wide energy range inflates global noise estimates.
            local_window = max(0.24, 12.0 * float(self.sampling[2]))
            mask_local = np.abs(E - float(E[int(pk_idx)])) <= local_window
            if int(np.count_nonzero(mask_local)) < 9:
                return float(background_std)

            y_local = np.asarray(spec[mask_local], dtype=float)
            if y_local.size < 9 or not np.all(np.isfinite(y_local)):
                return float(background_std)

            local_cut = float(np.nanpercentile(y_local, 70))
            base_local = y_local[y_local <= local_cut]
            if base_local.size < 5:
                base_local = y_local

            local_std = float(np.nanstd(base_local))
            if not np.isfinite(local_std) or local_std <= 0:
                local_std = float(background_std)
            return max(local_std, 1e-12)

        local_noise = np.asarray([_local_noise_std(int(i)) for i in peak_indices], dtype=float)
        local_snr_values = np.asarray(
            [float(h) / max(float(n), 1e-12) for h, n in zip(peak_heights, local_noise)],
            dtype=float,
        )
        local_prominence_snr = np.asarray(
            [float(p) / max(float(n), 1e-12) for p, n in zip(peak_proms, local_noise)], dtype=float
        )

        prominence_floor = max(2.2, 0.85 * float(floor))
        salience_snr = prominence_snr * np.sqrt(np.maximum(peak_width_samples, 1e-12))
        salience_floor = max(4.2, 2.0 * float(floor))
        local_salience_snr = local_prominence_snr * np.sqrt(np.maximum(peak_width_samples, 1e-12))

        adaptive_floor = max(2.0, 0.62 * float(floor))
        adaptive_prominence_floor = max(1.6, 0.62 * float(prominence_floor))
        adaptive_salience_floor = max(2.6, 0.62 * float(salience_floor))

        display_peaks_with_prom = [
            (
                int(i),
                float(h),
                float(E[i]),
                float(max(float(h / background_std), float(local_snr))),
                float(max(float(p_snr), float(local_p_snr))),
                float(max(float(sal), float(local_sal))),
            )
            for i, h, p_snr, sal, local_snr, local_p_snr, local_sal in zip(
                peak_indices,
                peak_heights,
                prominence_snr,
                salience_snr,
                local_snr_values,
                local_prominence_snr,
                local_salience_snr,
            )
            if (
                not in_ignore(E[i])
                and (
                    (
                        h / background_std >= floor
                        and p_snr >= prominence_floor
                        and sal >= salience_floor
                    )
                    or (
                        local_snr >= adaptive_floor
                        and local_p_snr >= adaptive_prominence_floor
                        and local_sal >= adaptive_salience_floor
                    )
                )
            )
        ]

        # Validate peaks as local Gaussian components (center/sigma/amplitude)
        # rather than raw single-bin maxima, then merge overlapping components.
        def _gauss_with_offset(x, amp, mu, sigma, offset):
            sigma = max(float(sigma), 1e-12)
            return float(offset) + float(amp) * np.exp(-0.5 * ((x - float(mu)) / sigma) ** 2)

        def _fit_local_gaussian(pk_idx):
            window = max(0.18, 10.0 * float(self.sampling[2]))
            x0 = float(E[pk_idx])
            mask_local = np.abs(E - x0) <= window
            if int(np.count_nonzero(mask_local)) < 7:
                return None

            x_local = np.asarray(E[mask_local], dtype=float)
            y_local = np.asarray(spec[mask_local], dtype=float)
            if not np.all(np.isfinite(y_local)):
                return None

            baseline = float(np.percentile(y_local, 20))
            peak_val = float(spec[pk_idx])
            amp0 = max(peak_val - baseline, 1e-9)
            sigma0 = max(0.04, 2.0 * float(self.sampling[2]))

            lo_sigma = max(1.5 * float(self.sampling[2]), 0.010)
            hi_sigma = 0.18
            bounds = (
                [0.0, x0 - 0.06, lo_sigma, baseline - abs(amp0)],
                [max(amp0 * 5.0, 1e-6), x0 + 0.06, hi_sigma, baseline + abs(amp0)],
            )

            try:
                popt, _ = curve_fit(
                    _gauss_with_offset,
                    x_local,
                    y_local,
                    p0=[amp0, x0, sigma0, baseline],
                    bounds=bounds,
                    maxfev=4000,
                )
            except Exception:
                return None

            amp, mu, sigma, offset = map(float, popt)
            if amp <= 0 or not np.isfinite(mu) or not np.isfinite(sigma):
                return None

            y_hat = _gauss_with_offset(x_local, amp, mu, sigma, offset)
            ss_res = float(np.sum((y_local - y_hat) ** 2))
            ss_tot = float(np.sum((y_local - float(np.mean(y_local))) ** 2))
            r2 = 1.0 - ss_res / max(ss_tot, 1e-12)
            amp_snr = amp / max(float(background_std), 1e-12)

            return {
                "idx": int(pk_idx),
                "mu": float(mu),
                "sigma": float(sigma),
                "amp": float(amp),
                "amp_snr": float(amp_snr),
                "r2": float(r2),
                "area": float(amp * sigma),
            }

        gaussian_validation_gate = max(2.2 * float(floor), 0.25 * float(snr_threshold))
        strong_keep_idx = {
            int(pk_idx)
            for pk_idx, _, _, snr, _, _ in display_peaks_with_prom
            if float(snr) >= gaussian_validation_gate
        }

        gauss_components = []
        for pk_idx, _, _, snr, _, _ in display_peaks_with_prom:
            if float(snr) >= gaussian_validation_gate:
                continue
            fit = _fit_local_gaussian(int(pk_idx))
            if fit is None:
                continue
            # Keep only physically plausible and sufficiently Gaussian components.
            if fit["r2"] < 0.58:
                continue
            if fit["amp_snr"] < max(2.0, 0.75 * float(floor)):
                continue
            if fit["sigma"] < max(1.5 * float(self.sampling[2]), 0.010) or fit["sigma"] > 0.18:
                continue
            gauss_components.append(fit)

        gaussian_validated = bool(gauss_components)
        if gauss_components:
            # Merge overlapping Gaussian components and keep the stronger one.
            gauss_components.sort(key=lambda comp: comp["mu"])
            merged = []
            for comp in gauss_components:
                if not merged:
                    merged.append(comp)
                    continue
                prev = merged[-1]
                # Keep neighbouring components separate unless they are truly
                # unresolved by both center spacing and valley separation.
                center_gap = abs(float(comp["mu"]) - float(prev["mu"]))
                overlap_thresh = 1.15 * min(float(prev["sigma"]), float(comp["sigma"]))

                prev_idx = int(prev["idx"])
                comp_idx = int(comp["idx"])
                lo, hi = sorted((prev_idx, comp_idx))
                if hi - lo <= 1:
                    valley = float(min(spec[lo], spec[hi]))
                else:
                    valley = float(np.min(spec[lo : hi + 1]))
                smaller_amp = max(min(float(prev["amp"]), float(comp["amp"])), 1e-12)
                valley_relief = (smaller_amp - valley) / smaller_amp

                unresolved_pair = center_gap <= overlap_thresh and valley_relief < 0.22
                if unresolved_pair:
                    if (comp["area"] > prev["area"]) or (
                        comp["area"] == prev["area"] and comp["amp_snr"] > prev["amp_snr"]
                    ):
                        merged[-1] = comp
                else:
                    merged.append(comp)

            keep_idx = {int(comp["idx"]) for comp in merged}
            keep_idx.update(strong_keep_idx)
            display_peaks_with_prom = [
                item for item in display_peaks_with_prom if int(item[0]) in keep_idx
            ]
        else:
            # If weak-peak Gaussian fitting did not validate any component,
            # still keep strong visual peaks.
            if strong_keep_idx:
                display_peaks_with_prom = [
                    item for item in display_peaks_with_prom if int(item[0]) in strong_keep_idx
                ]

        # Prune weak shoulder-like bumps near a much stronger neighbouring peak.
        # This prevents over-detecting pseudo-peaks on the flanks of broad peaks.
        if len(display_peaks_with_prom) > 1 and not gaussian_validated:
            by_energy = sorted(display_peaks_with_prom, key=lambda item: item[2])
            shoulder_window = max(8.0 * float(self.sampling[2]), 0.22)
            weak_snr_ratio = 0.45
            weak_prom_ratio = 0.65
            local_prom_floor = max(3.5, 1.10 * float(floor))

            pruned = []
            for idx, h, en, snr, p_snr, sal in by_energy:
                strongest_neighbor = None
                for o_idx, o_h, o_en, o_snr, o_p_snr, o_sal in by_energy:
                    if o_idx == idx:
                        continue
                    if abs(float(o_en) - float(en)) > shoulder_window:
                        continue
                    if strongest_neighbor is None or o_snr > strongest_neighbor[0]:
                        strongest_neighbor = (float(o_snr), float(o_p_snr), float(o_en))

                if strongest_neighbor is None:
                    pruned.append((idx, h, en, snr, p_snr, sal))
                    continue

                nbr_snr, nbr_prom, _ = strongest_neighbor
                is_weak_shoulder = (
                    float(snr) < weak_snr_ratio * max(nbr_snr, 1e-12)
                    and float(p_snr) < weak_prom_ratio * max(nbr_prom, 1e-12)
                    and float(p_snr) < local_prom_floor
                )
                if not is_weak_shoulder:
                    pruned.append((idx, h, en, snr, p_snr, sal))

            display_peaks_with_prom = pruned

        display_peaks = [(idx, h, en, snr) for idx, h, en, snr, _, _ in display_peaks_with_prom]
        display_peaks.sort(key=lambda item: item[3], reverse=True)

        def candidate_matches(peak_energy, snr, allowed_elements=None):
            matches = []
            for element_name, lines in all_info.items():
                if allowed_elements is not None and element_name not in allowed_elements:
                    continue
                for line_name, line_info in lines.items():
                    if not type(self)._line_allowed_for_element(
                        element_name, line_name, edge_filters
                    ):
                        continue
                    line_weight = float(line_info.get("weight", 0.5))
                    line_energy = float(line_info["energy (keV)"])
                    shell = type(self)._line_shell(line_name)
                    tol = (
                        tolerance * 0.5
                        if shell == "M" and ("Ma" not in line_name and "Mb" not in line_name)
                        else tolerance
                    )
                    distance = abs(peak_energy - line_energy)
                    if line_weight < min_line_weight or distance > tol:
                        continue
                    score = type(self)._peak_confidence(
                        snr, line_weight, distance, tolerance
                    ) * type(self)._shell_preference_factor(shell)
                    matches.append(
                        {
                            "element": str(element_name),
                            "line": str(line_name),
                            "weight": line_weight,
                            "distance": distance,
                            "score": float(score),
                            "shell": shell,
                        }
                    )
            matches.sort(key=lambda m: m["score"], reverse=True)
            return matches

        peak_matches = []
        for peak_idx, height, peak_energy, snr in display_peaks:
            matches = candidate_matches(peak_energy, snr, search_elements)
            if not matches:
                continue
            best = matches[0]
            peak_matches.append(
                (
                    peak_idx,
                    height,
                    peak_energy,
                    snr,
                    best["element"],
                    f"{best['element']} {best['line']}",
                    best["distance"],
                    best["line"],
                    best["weight"],
                    best["score"],
                )
            )

        energy_min = float(np.min(E)) if len(E) else float(self.origin[2])
        energy_max = float(np.max(E)) if len(E) else energy_min

        def observable_shells_for_element(element):
            shells = set()
            for line_name, line_info in (all_info.get(str(element), {}) or {}).items():
                if not type(self)._line_allowed_for_element(str(element), line_name, edge_filters):
                    continue
                shell = type(self)._line_shell(line_name)
                if shell not in {"K", "L", "M"}:
                    continue
                try:
                    line_energy = float(line_info.get("energy (keV)", line_info.get("energy")))
                except (TypeError, ValueError):
                    continue
                if energy_min <= line_energy <= energy_max:
                    shells.add(shell)
            return shells

        def strongest_observable_line(element, shell_name):
            candidates = []
            for line_name, line_info in (all_info.get(str(element), {}) or {}).items():
                if not type(self)._line_allowed_for_element(str(element), line_name, edge_filters):
                    continue
                if type(self)._line_shell(line_name) != shell_name:
                    continue
                try:
                    line_energy = float(line_info.get("energy (keV)", line_info.get("energy")))
                    line_weight = float(line_info.get("weight", 0.0))
                except (TypeError, ValueError):
                    continue
                if energy_min <= line_energy <= energy_max:
                    candidates.append((line_weight, line_energy, str(line_name)))
            return max(candidates, default=None)

        def shell_has_observable_support(element, shell_name):
            strongest = strongest_observable_line(element, shell_name)
            if strongest is None:
                return True

            _, target_energy, _ = strongest
            support_window = max(float(tolerance), 3.0 * float(self.sampling[2]), 0.04)

            for _, _, peak_energy, _ in display_peaks:
                dist_to_target = abs(float(peak_energy) - float(target_energy))
                if dist_to_target > support_window:
                    continue
                # Nearby spectral support exists for this shell line.
                return True

            local_idx = np.where(np.abs(E - float(target_energy)) <= support_window)[0]
            if local_idx.size == 0:
                return False

            local_snr = float(np.nanmax(spec[local_idx]) / max(float(background_std), 1e-9))
            weak_bump_threshold = max(2.5, 0.35 * float(snr_threshold))
            if local_snr < weak_bump_threshold:
                return False

            return True

        element_stats, line_evidence = {}, {}
        for (
            _,
            _,
            peak_energy,
            snr,
            element,
            _,
            distance,
            line_name,
            line_weight,
            conf,
        ) in peak_matches:
            if search_elements is not None and element not in search_elements:
                continue
            shell = type(self)._line_shell(line_name)
            stats = element_stats.setdefault(
                element,
                {
                    "raw_conf": 0.0,
                    "shells": set(),
                    "lines": set(),
                    "strong_matches": 0,
                    "match_count": 0,
                    "best_match_conf": 0.0,
                    "best_match_snr": 0.0,
                    "best_match_energy": 0.0,
                    "best_match_distance": float("inf"),
                    "best_match_weight": 0.0,
                    "best_match_shell": "?",
                },
            )
            label = f"{element} {line_name}"
            evidence = line_evidence.setdefault(
                label,
                {
                    "match_count": 0,
                    "strong_matches": 0,
                    "best_conf": 0.0,
                    "best_snr": 0.0,
                    "energies": [],
                },
            )

            stats["raw_conf"] += float(conf)
            stats["shells"].add(shell)
            stats["lines"].add(line_name)
            stats["match_count"] += 1
            stats["strong_matches"] += int(
                snr > snr_threshold and distance < distance_threshold_for_sample
            )
            if conf > stats["best_match_conf"]:
                stats.update(
                    {
                        "best_match_conf": float(conf),
                        "best_match_snr": float(snr),
                        "best_match_energy": float(peak_energy),
                        "best_match_distance": float(distance),
                        "best_match_weight": float(line_weight),
                        "best_match_shell": shell,
                    }
                )

            evidence["match_count"] += 1
            evidence["energies"].append(float(peak_energy))
            evidence["strong_matches"] += int(
                snr > snr_threshold and distance < distance_threshold_for_sample
            )
            if conf > evidence["best_conf"]:
                evidence["best_conf"] = float(conf)
                evidence["best_snr"] = float(snr)

        # Collect all candidate elements across every display peak (not just best-match winners)
        all_candidate_shells: dict[str, set] = {}
        for peak_idx, height, peak_energy, snr in display_peaks:
            for m in candidate_matches(peak_energy, snr, search_elements):
                shell = m["shell"]
                if shell in {"K", "L", "M"}:
                    all_candidate_shells.setdefault(m["element"], set()).add(shell)

        shell_hierarchy = ["K", "L", "M"]  # descending energy order

        demoted_elements = set()
        for element, shells in all_candidate_shells.items():
            # Prefer shells that actually won first-pass matches for this element.
            # Using all candidate shells can falsely trigger higher-shell checks
            # (e.g. Cu candidate L-lines) even when the element is only evidenced by K-lines.
            observed_shells = set((element_stats.get(element, {}) or {}).get("shells", set())) & {
                "K",
                "L",
                "M",
            }
            matched_shells = observed_shells if observed_shells else (shells & {"K", "L", "M"})
            observable = observable_shells_for_element(element)
            eliminate = False
            for matched_shell in matched_shells:
                shell_idx = shell_hierarchy.index(matched_shell)
                # Every higher-energy shell that is observable must have spectral support.
                # Also verify that the supporting shell is genuine by checking its own
                # strong secondary lines — prevents a coincidental neighbouring peak
                # (e.g. Cu Kb1,3 near Os La1) from falsely satisfying the L-shell check.
                for higher_shell in shell_hierarchy[:shell_idx]:
                    if higher_shell not in observable:
                        continue
                    if not shell_has_observable_support(element, higher_shell):
                        eliminate = True
                        break
                if eliminate:
                    break
            if eliminate:
                demoted_elements.add(str(element))

        element_confidence = {}
        # --- Intensity ratio check and multi-peak pattern boost ---
        for element, stats in element_stats.items():
            valid_shells = {shell for shell in stats["shells"] if shell in {"K", "L", "M"}}
            shell_bonus = float(np.sqrt(max(1, len(valid_shells))))
            line_bonus = 1.0 + 0.30 * float(np.log1p(max(0, len(stats["lines"]) - 1)))
            strong_bonus = 1.0 + 0.40 * float(np.log1p(stats["strong_matches"]))
            major_bonus = 1.20 if {"K", "L"} & valid_shells else 1.0

            # Intensity ratio logic
            element_peak_intensities = {}
            for (
                _,
                height,
                peak_energy,
                snr,
                el,
                _,
                distance,
                line_name,
                line_weight,
                conf,
            ) in peak_matches:
                if el == element:
                    element_peak_intensities.setdefault(line_name, []).append(float(height))
            # Only consider if at least 2 lines detected
            if len(element_peak_intensities) >= 2:
                observed = []
                expected = []
                for line_name, intensities in element_peak_intensities.items():
                    observed.append(max(intensities))
                    weight = all_info.get(element, {}).get(line_name, {}).get("weight", None)
                    try:
                        expected.append(float(weight) if weight is not None else 0.0)
                    except Exception:
                        expected.append(0.0)
                obs_sum = sum(observed)
                exp_sum = sum(expected)
                if obs_sum > 0 and exp_sum > 0:
                    observed_norm = [x / obs_sum for x in observed]
                    expected_norm = [x / exp_sum for x in expected]
                    ratio_score = 1.0 - (
                        sum(abs(o - e) for o, e in zip(observed_norm, expected_norm)) / 2.0
                    )
                    ratio_factor = 1.0
                    if ratio_score > 0.7:
                        ratio_factor = 1.15 + 0.25 * (ratio_score - 0.7)
                    elif ratio_score < 0.4:
                        ratio_factor = 0.7 + 0.5 * ratio_score
                else:
                    ratio_factor = 1.0
            else:
                ratio_factor = 1.0

            # --- Strong pattern boost: if both main lines for K, L, or M are matched, multiply confidence by 3 (dominates score) ---
            matched_lines = set(element_peak_intensities.keys())
            k_lines = {"Ka1", "Kb1"}
            l_lines = {"La1", "Lb1"}
            m_lines = {"Ma1", "Mb1"}
            pattern_factor = 1.0
            if k_lines.issubset(matched_lines):
                pattern_factor = 3.0
            elif l_lines.issubset(matched_lines):
                pattern_factor = 2.5
            elif m_lines.issubset(matched_lines):
                pattern_factor = 2.0

            element_confidence[element] = (
                stats["raw_conf"]
                * shell_bonus
                * line_bonus
                * strong_bonus
                * major_bonus
                * ratio_factor
                * pattern_factor
            )

        detected_elements = set()
        if element_confidence:
            conf_values = np.asarray(list(element_confidence.values()), dtype=float)
            poisson_mdl_snr = 3.0
            cutoff = max(float(np.percentile(conf_values, 45)), 0.30 * float(conf_values.max()))
            for element, confidence in element_confidence.items():
                stats = element_stats[element]
                lines = set(stats["lines"])
                # Criterion 1: Both main lines matched (pattern match) → always autodetect
                strong_pattern = (
                    {"Ka1", "Kb1"}.issubset(lines)
                    or {"La1", "Lb1"}.issubset(lines)
                    or {"Ma1", "Mb1"}.issubset(lines)
                ) and confidence > 0
                # Criterion 2: High confidence above cutoff and sufficient SNR
                high_confidence = (
                    confidence >= cutoff and stats["best_match_snr"] >= poisson_mdl_snr
                )
                if strong_pattern or high_confidence:
                    detected_elements.add(element)

        dominant_elements = set()
        if element_confidence:
            conf_values = np.asarray(list(element_confidence.values()), dtype=float)
            conf_floor = max(float(np.median(conf_values)) if conf_values.size else 0.0, 1e-9)
            conf_p80 = float(np.percentile(conf_values, 80)) if conf_values.size > 1 else 0.0
            for element, confidence in element_confidence.items():
                stats = element_stats.get(element, {})
                repeat_support = (
                    int(stats.get("match_count", 0)) >= 2
                    or int(stats.get("strong_matches", 0)) >= 1
                )
                if confidence >= conf_p80 and confidence >= 1.8 * conf_floor and repeat_support:
                    dominant_elements.add(element)

        anchor_elements = {
            element
            for element in detected_elements
            if element in element_stats
            and element_stats[element].get("best_match_energy", 0.0) >= 6.0
            and element_stats[element].get("best_match_weight", 0.0) >= 0.8
        }
        max_detected_conf = max(
            [element_confidence.get(el, 0.0) for el in detected_elements], default=0.0
        )

        def prior_boost(element):
            prior = float(element_confidence.get(element, 0.0)) / max(
                float(max_detected_conf), 1e-9
            )
            factor = 1.0 + 0.5 * prior
            if prior >= 0.90:
                factor *= 1.9
            elif prior >= 0.75:
                factor *= 1.5
            elif prior >= 0.55:
                factor *= 1.2
            return prior, factor

        def consistency_boost(element, line_name, peak_energy):
            is_detected = element in detected_elements
            is_dominant = element in dominant_elements
            if is_dominant:
                scale = 1.0
            elif is_detected:
                scale = 0.80
            else:
                scale = 0.65
            # First, check evidence for this exact line
            evidence = line_evidence.get(f"{element} {line_name}")
            if evidence and any(
                abs(float(peak_energy) - float(prev)) <= 0.04
                for prev in evidence.get("energies", [])
            ):
                best_conf = float(evidence.get("best_conf", 0.0))
                best_snr = float(evidence.get("best_snr", 0.0))
                strong = int(evidence.get("strong_matches", 0))
                line_weight = float(
                    (all_info.get(element, {}).get(line_name, {}) or {}).get("weight", 0.5)
                )
                tier = 1.0 + 0.7 * max(0.0, line_weight - 0.35)
                if strong >= 1 and best_conf >= 1.4:
                    return min(3.2, scale * 2.4 * tier)
                if best_conf >= 1.1 and best_snr >= max(floor, 0.75 * snr_threshold):
                    return min(2.6, scale * 1.9 * tier)
                if best_conf >= 0.8:
                    return min(2.0, scale * 1.5 * tier)
                return min(1.5, scale * 1.2 * tier)
            # Element was matched via a different line — boost secondary lines of this element
            stats = element_stats.get(element, {})
            elem_conf = float(element_confidence.get(element, 0.0))
            elem_strong = int(stats.get("strong_matches", 0))
            line_weight = float(
                (all_info.get(element, {}).get(line_name, {}) or {}).get("weight", 0.5)
            )
            tier = 1.0 + 0.5 * max(0.0, line_weight - 0.35)
            if elem_strong >= 1 and elem_conf >= 1.4:
                return min(2.4, scale * 1.8 * tier)
            if elem_conf >= 1.1:
                return min(2.0, scale * 1.5 * tier)
            if elem_conf >= 0.8:
                return min(1.6, scale * 1.2 * tier)
            return min(1.3, scale * 1.1 * tier)

        def dominant_boost(element):
            if element not in dominant_elements:
                return 1.0
            prior, _ = prior_boost(element)
            stats = element_stats.get(element, {})
            repeat_support = max(
                int(stats.get("strong_matches", 0)), max(0, int(stats.get("match_count", 0)) - 1)
            )
            base = 2.30 if prior >= 0.90 else 1.85 if prior >= 0.75 else 1.45
            if repeat_support >= 2:
                base *= 1.10
            return min(base, 2.60)

        def reranked_matches(peak_energy, snr, allowed_elements=None, top_k=None):
            # Compute which elements have both main lines matched (pattern boost)
            element_to_lines = {}
            for _, _, _, _, el, _, _, ln, _, _ in peak_matches:
                element_to_lines.setdefault(el, set()).add(ln)

            def _has_main_line(lines, target):
                # Accept compact aliases from x_ray_lines.csv such as La1,2 or Kb1,3.
                for ln in lines:
                    name = str(ln)
                    if name == target:
                        return True
                    if target in {"Ka1", "Kb1", "La1", "Lb1", "Ma1", "Mb1"} and name.startswith(
                        target + ","
                    ):
                        return True
                return False

            def _canonical_aliases(target):
                if target == "La1":
                    return ("La1", "La1,2")
                if target == "Lb1":
                    return ("Lb1",)
                if target == "Ka1":
                    return ("Ka1",)
                if target == "Kb1":
                    return ("Kb1", "Kb1,3")
                return (target,)

            def _line_evidence_strength(element, target):
                best = 0.0
                for alias in _canonical_aliases(target):
                    ev = line_evidence.get(f"{element} {alias}")
                    if not ev:
                        continue
                    best_conf = float(ev.get("best_conf", 0.0))
                    strong = float(ev.get("strong_matches", 0))
                    count = float(ev.get("match_count", 0))
                    score = best_conf + 0.45 * strong + 0.15 * count
                    if score > best:
                        best = score
                return best

            def _l_support_strength(element):
                return _line_evidence_strength(element, "La1") + _line_evidence_strength(
                    element, "Lb1"
                )

            candidates = candidate_matches(peak_energy, snr, allowed_elements)

            # For weak peaks, also consider a relaxed-distance pass for already
            # detected/dominant elements. This keeps context-consistent lines in
            # play (e.g. Te continuation) even when local calibration/noise shifts
            # push them slightly beyond the strict tolerance window.
            weak_peak = float(snr) < max(2.5 * float(floor), 0.30 * float(snr_threshold))
            if weak_peak:
                relaxed_tol = max(float(tolerance), 0.30)
                context_elements = set(map(str, detected_elements | dominant_elements))
                if context_elements:
                    for element_name, lines in all_info.items():
                        element_name = str(element_name)
                        if element_name not in context_elements:
                            continue
                        if allowed_elements is not None and element_name not in allowed_elements:
                            continue
                        for line_name, line_info in lines.items():
                            if not type(self)._line_allowed_for_element(
                                element_name, line_name, edge_filters
                            ):
                                continue
                            line_weight = float(line_info.get("weight", 0.5))
                            line_energy = float(line_info["energy (keV)"])
                            shell = type(self)._line_shell(line_name)
                            tol = (
                                relaxed_tol * 0.5
                                if shell == "M"
                                and ("Ma" not in line_name and "Mb" not in line_name)
                                else relaxed_tol
                            )
                            distance = abs(float(peak_energy) - line_energy)
                            if line_weight < min_line_weight or distance > tol:
                                continue
                            score = type(self)._peak_confidence(
                                snr, line_weight, distance, relaxed_tol
                            ) * type(self)._shell_preference_factor(shell)
                            candidates.append(
                                {
                                    "element": element_name,
                                    "line": str(line_name),
                                    "weight": line_weight,
                                    "distance": distance,
                                    "score": float(score),
                                    "shell": shell,
                                }
                            )

            # De-duplicate exact element/line candidates, keeping highest score.
            if candidates:
                uniq = {}
                for c in candidates:
                    key = (str(c["element"]), str(c["line"]))
                    prev = uniq.get(key)
                    if prev is None or float(c["score"]) > float(prev["score"]):
                        uniq[key] = c
                candidates = list(uniq.values())
                candidates.sort(key=lambda m: m["score"], reverse=True)

                # Performance guard: the precedence logic below is O(n^2) over
                # candidates. Keep the strongest candidates, but always retain
                # context-important elements (confirmed/dominant/preferred).
                max_candidates = 48
                if len(candidates) > max_candidates:
                    context_elements = set(
                        map(str, detected_elements | dominant_elements | preferred_elements)
                    )
                    trimmed = list(candidates[:max_candidates])
                    if context_elements:
                        kept_keys = {(str(c["element"]), str(c["line"])) for c in trimmed}
                        for c in candidates[max_candidates:]:
                            key = (str(c["element"]), str(c["line"]))
                            if key in kept_keys:
                                continue
                            if str(c["element"]) in context_elements:
                                trimmed.append(c)
                                kept_keys.add(key)
                    candidates = trimmed

            element_has_l_support = {}
            element_has_l_pair = {}
            for el, lines in element_to_lines.items():
                has_la = _has_main_line(lines, "La1")
                has_lb = _has_main_line(lines, "Lb1")
                element_has_l_support[str(el)] = has_la or has_lb
                element_has_l_pair[str(el)] = has_la and has_lb

            # Guard against boosted confirmed elements stealing a peak from a
            # much-closer strong K/L candidate (e.g. Cu Ka around 8 keV).
            distance_anchor = None
            for candidate in candidates:
                if candidate["shell"] not in {"K", "L"}:
                    continue
                if float(candidate["weight"]) < 0.30:
                    continue
                if float(candidate["score"]) < 0.45:
                    continue
                distance_anchor = max(float(candidate["distance"]), 1e-9)
                break

            scored = []
            for match in candidates:
                element, line_name, shell = match["element"], match["line"], match["shell"]
                is_demoted = str(element) in demoted_elements

                minor_l_penalty = 1.0

                # Logical guard for L-series continuation: do not let an orphan
                # minor L-line assignment (Ll/Lg/Lb2) outrank a closer candidate
                # from an element that already shows L-series support (La/Lb).
                if (
                    shell == "L"
                    and line_name in {"Ll", "Lg1", "Lb2,15"}
                    and not element_has_l_support.get(str(element), False)
                ):
                    supported_closer_exists = False
                    for other in candidates:
                        other_el = str(other["element"])
                        if other_el == str(element):
                            continue
                        if other["shell"] != "L":
                            continue
                        if not element_has_l_support.get(other_el, False):
                            continue
                        if float(other["distance"]) < float(match["distance"]):
                            supported_closer_exists = True
                            break
                    if supported_closer_exists:
                        minor_l_penalty *= 0.55

                # Stricter logical precedence: an orphan minor L-line must not
                # outrank a closer L-line from an element with an established
                # La/Lb pair in this spectrum.
                if (
                    shell == "L"
                    and line_name in {"Ll", "Lg1", "Lb2,15"}
                    and not element_has_l_pair.get(str(element), False)
                ):
                    paired_closer_exists = False
                    for other in candidates:
                        other_el = str(other["element"])
                        if other_el == str(element):
                            continue
                        if other["shell"] != "L":
                            continue
                        if not element_has_l_pair.get(other_el, False):
                            continue
                        if float(other["distance"]) <= float(match["distance"]):
                            paired_closer_exists = True
                            break
                    if paired_closer_exists:
                        minor_l_penalty *= 0.70

                # Evidence-strength precedence for minor L-lines:
                # if another element has materially stronger La/Lb evidence and
                # a comparable-or-better distance match, do not keep the weaker
                # minor-L candidate as a possible winner.
                if shell == "L" and line_name in {"Ll", "Lg1", "Lb2,15"}:
                    this_el = str(element)
                    this_dist = float(match["distance"])
                    this_support = _l_support_strength(this_el)
                    beaten_by_stronger = False
                    for other in candidates:
                        other_el = str(other["element"])
                        if other_el == this_el or other["shell"] != "L":
                            continue
                        other_support = _l_support_strength(other_el)
                        if other_support <= max(0.4, this_support + 0.30):
                            continue
                        # Accept up to 30 eV slack so support can break near ties.
                        if float(other["distance"]) <= this_dist + 0.03:
                            beaten_by_stronger = True
                            break
                    if beaten_by_stronger:
                        minor_l_penalty *= 0.60

                prior, prior_factor = prior_boost(element)
                pref = 1.35 if element in preferred_elements else 1.0
                anchor = 1.15 if element in anchor_elements and shell in {"K", "L"} else 1.0
                dom = dominant_boost(element)
                # Pattern boost: if both main lines for K, L, or M are matched by detected peaks, boost candidate score
                lines_matched = element_to_lines.get(element, set())
                has_k_pair = _has_main_line(lines_matched, "Ka1") and _has_main_line(
                    lines_matched, "Kb1"
                )
                has_l_pair = _has_main_line(lines_matched, "La1") and _has_main_line(
                    lines_matched, "Lb1"
                )
                has_m_pair = _has_main_line(lines_matched, "Ma1") and _has_main_line(
                    lines_matched, "Mb1"
                )
                pattern_factor = 1.0
                if has_k_pair:
                    pattern_factor = 3.0
                elif has_l_pair:
                    pattern_factor = 2.5
                elif has_m_pair:
                    pattern_factor = 2.0

                if shell == "M":
                    prior_factor = 1.0 + 0.3 * prior
                    dom = min(dom, 1.30)

                # Guard against introducing new singleton elements on weak peaks.
                # If an element is not already detected/dominant and only appears
                # as an isolated line, require a very tight distance match.
                if (
                    weak_peak
                    and element not in detected_elements
                    and element not in dominant_elements
                ):
                    matched_lines_for_el = element_to_lines.get(element, set())
                    # Consider an element supported if it already has any matched
                    # line in the current spectrum, or strong line evidence from
                    # first-pass matching. This avoids dropping context-consistent
                    # secondary lines (e.g. Cu Kb1,3 after Cu Ka1 is matched).
                    element_line_strength = 0.0
                    for ln in matched_lines_for_el:
                        ev = line_evidence.get(f"{element} {ln}")
                        if not ev:
                            continue
                        best_conf = float(ev.get("best_conf", 0.0))
                        strong = float(ev.get("strong_matches", 0))
                        count = float(ev.get("match_count", 0))
                        element_line_strength = max(
                            element_line_strength, best_conf + 0.4 * strong + 0.1 * count
                        )

                    has_support = len(matched_lines_for_el) >= 1 or element_line_strength >= 0.8
                    if not has_support:
                        if float(match["distance"]) > 0.035:
                            continue
                        prior_factor *= 0.65
                # For confirmed elements (detected or dominant), the line_weight prior is irrelevant —
                # we already know the element is present. Use weight=1.0 and score purely on distance
                # so that e.g. Cu Kb1 (weight=0.17) beats Os La1 (weight=1.0) when Cu is confirmed
                # and Cu Kb1 is closer to the measured peak.
                confirmed = element in detected_elements or element in dominant_elements
                if confirmed:
                    sigma = max(float(tolerance) / 3.0, 1e-9)
                    distance_factor = np.exp(-0.5 * (float(match["distance"]) / sigma) ** 2)
                    base_score = (
                        np.log1p(max(float(snr), 0.0))
                        * 1.0
                        * distance_factor
                        * type(self)._shell_preference_factor(shell)
                    )
                    # Once an element is clearly present, prefer physically
                    # consistent continuation lines over introducing new
                    # elements for nearby ambiguous peaks.
                    continuation = consistency_boost(element, line_name, peak_energy)
                    base_score = base_score * max(1.0, min(float(continuation), 1.8))
                else:
                    base_score = match["score"]
                    consistency = consistency_boost(element, line_name, peak_energy)
                    # Non-confirmed elements should not gain an aggressive
                    # boost that steals peaks from already-confirmed elements.
                    base_score = base_score * min(1.0, float(consistency))
                score = (
                    base_score
                    * prior_factor
                    * pref
                    * anchor
                    * dom
                    * pattern_factor
                    * minor_l_penalty
                )

                # If there is a strong nearby K/L anchor, damp long-distance
                # takeovers that are caused mainly by cross-peak boosts.
                if (
                    confirmed
                    and distance_anchor is not None
                    and float(match["distance"]) > distance_anchor
                ):
                    ratio = float(match["distance"]) / distance_anchor
                    if ratio >= 2.0:
                        score *= ratio**-1.6

                scored.append({**match, "score": float(score), "demoted": bool(is_demoted)})

            # Ranking-only policy: keep shell-inconsistent elements as options,
            # but place them behind more plausible (non-demoted) candidates.
            scored.sort(key=lambda m: (bool(m.get("demoted", False)), -float(m["score"])))
            if mode == "elements_preferred" and preferred_elements:
                preferred = [m for m in scored if m["element"] in preferred_elements]
                scored = (
                    preferred + [m for m in scored if m["element"] not in preferred_elements]
                    if preferred
                    else scored
                )

            unique, seen = [], set()
            for match in scored:
                label = f"{match['element']} {match['line']}"
                if label in seen:
                    continue
                seen.add(label)
                unique.append(match)

            if top_k is None or len(unique) <= 1:
                return unique
            selected = [unique[0]]
            used_elements = {unique[0]["element"]}
            for match in unique[1:]:
                if match["element"] in used_elements:
                    continue
                selected.append(match)
                used_elements.add(match["element"])
                if len(selected) >= int(top_k):
                    return selected
            for match in unique[1:]:
                if match not in selected:
                    selected.append(match)
                if len(selected) >= int(top_k):
                    break
            return selected

        rematch_allowed = {
            str(match[4]) for match in peak_matches if str(match[4]) not in ignored_elements
        }
        rematch_allowed.update(map(str, detected_elements))
        rematch_allowed.update(preferred_elements)

        refined_peak_matches = []
        for peak_idx, height, peak_energy, snr in display_peaks:
            best = reranked_matches(peak_energy, snr, search_elements, top_k=1)
            best = best[0] if best else None
            if best is None:
                continue
            refined_peak_matches.append(
                (
                    peak_idx,
                    height,
                    peak_energy,
                    snr,
                    best["element"],
                    f"{best['element']} {best['line']}",
                    best["distance"],
                    best["line"],
                    best["weight"],
                    best["score"],
                )
            )
        peak_matches = refined_peak_matches

        # Backfill element_confidence for elements that only appear after the
        # unrestricted re-rank (e.g. not in search_elements so never entered
        # element_stats in the first pass).  Use the same base formula as
        # _peak_confidence so the displayed value is meaningful.
        for (
            _,
            height,
            peak_energy,
            snr,
            element,
            _,
            distance,
            line_name,
            line_weight,
            _,
        ) in peak_matches:
            if element in element_confidence:
                continue
            sigma = max(float(tolerance) / 3.0, 1e-9)
            dist_factor = float(np.exp(-0.5 * (float(distance) / sigma) ** 2))
            raw = float(
                np.log1p(max(float(snr), 0.0)) * max(float(line_weight), 0.0) * dist_factor
            )
            shell = type(self)._line_shell(str(line_name))
            valid_shells = {shell} & {"K", "L", "M"}
            major_bonus = 1.20 if {"K", "L"} & valid_shells else 1.0
            element_confidence[element] = raw * major_bonus

        matched_elements = {str(match[4]) for match in peak_matches}
        detected_elements = {
            str(el)
            for el in detected_elements
            if str(el) in matched_elements and str(el) not in ignored_elements
        }
        if mode == "elements_preferred":
            detected_elements.update(
                str(el) for el in preferred_elements if str(el) in matched_elements
            )
        refined_match_by_idx = {int(match[0]): match for match in peak_matches}
        plot_peaks = display_peaks[:peaks]
        plot_peak_indices = {int(pk_idx) for pk_idx, _, _, _ in plot_peaks}

        final_matches_by_element: dict[str, set[str]] = {}
        for _, _, _, _, element, _, _, line_name, _, _ in peak_matches:
            if element not in ignored_elements:
                final_matches_by_element.setdefault(element, set()).add(str(line_name))

        # For display purposes (table + plot), restrict to elements/lines seen in plot_peaks
        plot_matches_by_element: dict[str, set[str]] = {}
        for pk_idx, _, _, _, element, _, _, line_name, _, _ in peak_matches:
            if int(pk_idx) in plot_peak_indices and element not in ignored_elements:
                plot_matches_by_element.setdefault(element, set()).add(str(line_name))

        candidate_elements = sorted(
            str(el) for el in final_matches_by_element if str(el) not in detected_elements
        )
        possible_elements = set(candidate_elements)

        plot_all_identified = set(
            el
            for el in (set(detected_elements) | set(candidate_elements))
            if el in plot_matches_by_element
        )
        if plot_all_identified:
            det_rows = []
            for element in sorted(map(str, plot_all_identified)):
                conf = element_confidence.get(element, 0.0)
                lines_matched = sorted(map(str, plot_matches_by_element.get(element, set())))
                if element in detected_elements:
                    status = "Dominant" if element in dominant_elements else "Detected"
                else:
                    status = "Possible"
                det_rows.append(
                    (element, status, conf, ", ".join(lines_matched) if lines_matched else "-")
                )
            det_rows.sort(
                key=lambda r: (0 if r[1] == "Dominant" else 1 if r[1] == "Detected" else 2, -r[2])
            )
            print(f"\n{'Element':<10} {'Confidence':<12} {'Matched Lines'}")
            print("-" * 50)
            for el, status, conf, lines_str in det_rows:
                print(f"{el:<10} {conf:<12.3f} {lines_str}")
            print("-" * 50)
        else:
            print("\nDetected: None")

        elements_for_color = set(detected_elements) | {str(match[4]) for match in peak_matches}
        if search_elements is not None:
            elements_for_color.update(map(str, search_elements))
        palette = [
            "#1f77b4",
            "#d62728",
            "#2ca02c",
            "#9467bd",
            "#ff7f0e",
            "#8c564b",
            "#e377c2",
            "#17becf",
            "#bcbd22",
            "#7f7f7f",
            "#003f5c",
            "#7a5195",
            "#ef5675",
            "#ffa600",
            "#2f4b7c",
        ]
        element_color_map = {
            el: palette[i % len(palette)] for i, el in enumerate(sorted(elements_for_color))
        }
        y_min = float(np.nanmin(spec)) if len(spec) else 0.0
        y_max = float(np.nanmax(spec)) if len(spec) else 1.0
        y_scale = max(max(1e-9, y_max - y_min), abs(y_max), abs(y_min), 1e-6)
        y_dot = -0.04 * y_scale

        def infer_requested_color(peak_energy):
            if reference_elements is None:
                return None
            best_element, best_distance = None, float("inf")
            for element in reference_elements:
                for line_name, line_info in (all_info.get(str(element), {}) or {}).items():
                    if not type(self)._line_allowed_for_element(
                        str(element), line_name, edge_filters
                    ):
                        continue
                    try:
                        distance = abs(float(peak_energy) - float(line_info.get("energy (keV)")))
                    except (TypeError, ValueError):
                        continue
                    if distance < best_distance:
                        best_distance, best_element = distance, str(element)
            return best_element

        table_rows = []
        for peak_idx, height, peak_energy, snr in plot_peaks:
            match = refined_match_by_idx.get(int(peak_idx))
            color = (
                element_color_map.get(match[4], "red")
                if match is not None
                else element_color_map.get(str(infer_requested_color(peak_energy)), "red")
            )

            if not in_ignore(peak_energy):
                # Only plot solid lines for matched peaks (autodetected or requested elements)
                if match is not None:
                    ax_spec.axvline(
                        peak_energy, color=color, linestyle="-", alpha=0.5, linewidth=1.5
                    )
                else:
                    ax_spec.plot(
                        [peak_energy],
                        [y_dot],
                        marker="|",
                        markersize=4,
                        color="gray",
                        alpha=0.8,
                        linestyle="None",
                    )

                if show_text and match is not None:
                    for grid_element, grid_energy in grid_peaks.items():
                        if abs(peak_energy - grid_energy) < 0.1:
                            ax_spec.text(
                                peak_energy,
                                height * 0.7,
                                f"{grid_element}\n(grid)",
                                ha="center",
                                va="bottom",
                                fontsize=8,
                                color="gray",
                                style="italic",
                            )
                            print(f"Peak at {peak_energy} keV may come from the grid.")
                            break

            def label_with_energy_and_ratio(label):
                # label is like 'Fe Ka', want to append (energy, ratio) from all_info and observed/expected
                if not label or label == "-" or label == "Unmatched" or label == "Unknown":
                    return label
                parts = label.split()
                if len(parts) < 2:
                    return label
                element, line = parts[0], parts[1].replace("*", "")
                line_info = all_info.get(element, {}).get(line, {})
                ref_energy = None
                if isinstance(line_info, dict):
                    ref_energy = line_info.get("energy (keV)", line_info.get("energy"))
                try:
                    ref_energy = float(ref_energy)
                except (TypeError, ValueError):
                    ref_energy = None
                label_core = label.rstrip("*")
                star = "*" if label.endswith("*") else ""
                if ref_energy is not None:
                    return f"{label_core} ({ref_energy:.3f}){star}"
                else:
                    return label

            if match is None:
                table_rows.append(
                    (
                        peak_energy,
                        height,
                        snr,
                        "Unmatched" if search_elements is not None else "Unknown",
                        "-",
                        "-",
                    )
                )
                continue

            # Best match for the table MUST be the same element/line shown on the spectrum
            # (from refined_match_by_idx). Preserve elements_only filtering for alternatives.
            best_label = f"{match[4]} {match[7]}"
            ranked = reranked_matches(peak_energy, snr, search_elements, top_k=3)
            labels = [
                (f"{m['element']} {m['line']}", float(m["score"]), m["element"], m["line"])
                for m in ranked
            ]
            # If the spectrum winner appears in ranked, use that ordering; otherwise prepend it.
            if not any(lbl.lower() == best_label.lower() for lbl, _, _, _ in labels):
                labels = [(best_label, 0.0, match[4], match[7])] + labels

            def fmt(label):
                label = (
                    f"{label}*"
                    if requested_elements and str(label).split()[0] in requested_elements
                    else label
                )
                return label

            remaining = [
                (label, score, elem, line)
                for label, score, elem, line in labels
                if label.lower() != best_label.lower()
            ]

            table_rows.append(
                (
                    peak_energy,
                    height,
                    snr,
                    label_with_energy_and_ratio(fmt(best_label)),
                    label_with_energy_and_ratio(fmt(remaining[0][0]))
                    if len(remaining) > 0
                    else "-",
                    label_with_energy_and_ratio(fmt(remaining[1][0]))
                    if len(remaining) > 1
                    else "-",
                )
            )

        current_bottom, current_top = ax_spec.get_ylim()
        padded_bottom = min(current_bottom, y_min - 0.10 * y_scale)
        padded_top = max(current_top, y_max + 0.18 * y_scale)
        ax_spec.set_ylim(bottom=padded_bottom, top=padded_top)

        label_candidates = []
        top_label_y = 0.99
        peak_label_y = 0.92
        # Plot reference lines (dotted) ONLY for explicitly requested elements, not for autodetected/possible
        if requested_elements:
            energy_min, energy_max = float(np.min(E)), float(np.max(E))
            matched_by_element = {}
            for _, _, peak_energy, _, element, _, _, _, _, _ in peak_matches:
                matched_by_element.setdefault(str(element), []).append(float(peak_energy))

            for element in sorted(requested_elements):
                candidates = []
                for line_name, line_info in (all_info.get(str(element), {}) or {}).items():
                    if not type(self)._line_allowed_for_element(
                        str(element), line_name, edge_filters
                    ):
                        continue
                    try:
                        line_energy = float(line_info.get("energy (keV)"))
                        line_weight = float(line_info.get("weight", 0.0))
                    except (TypeError, ValueError):
                        continue
                    if energy_min <= line_energy <= energy_max:
                        candidates.append((str(line_name), line_energy, line_weight))
                candidates = sorted(
                    [c for c in candidates if c[2] >= 0.05] or candidates,
                    key=lambda item: item[2],
                    reverse=True,
                )[:6]
                for line_name, line_energy, _ in candidates:
                    if in_ignore(line_energy):
                        continue
                    # Skip if already matched by a detected peak
                    if any(
                        abs(line_energy - matched_energy) <= max(0.05, 0.5 * tolerance)
                        for matched_energy in matched_by_element.get(str(element), [])
                    ):
                        continue
                    color = element_color_map.get(str(element), "gray")
                    style = "--"
                    alpha = 0.5
                    ax_spec.axvline(
                        line_energy, color="gray", linestyle=style, alpha=alpha, linewidth=1.2
                    )
                    label_candidates.append(
                        (
                            float(line_energy),
                            f"{element} {line_name}",
                            color,
                            style,
                            float(top_label_y),
                            "axes_top",
                            8,
                            "normal",
                            0.8,
                        )
                    )

        if show_text and peak_matches:
            label_allowed = set(detected_elements) | possible_elements
            if requested_elements:
                label_allowed.update(str(el) for el in requested_elements)
            for pk_idx, _height, peak_energy, _, element, match_str, _, _, _, _ in peak_matches:
                if int(pk_idx) not in plot_peak_indices:
                    continue
                is_requested = requested_elements is not None and element in requested_elements
                if element not in label_allowed or in_ignore(peak_energy):
                    continue
                label = f"{element} {match_str.split()[-1]}" + ("*" if is_requested else "")
                label_candidates.append(
                    (
                        float(peak_energy),
                        label,
                        element_color_map.get(element, "black"),
                        "-",
                        float(peak_label_y),
                        "axes_peak",
                        10,
                        "bold",
                        1.0,
                    )
                )

        legend_handles, legend_labels = [], set()
        if show_text and label_candidates:
            label_candidates.sort(key=lambda item: item[0])
            drawn_texts = []
            for (
                peak_energy,
                label_text,
                color,
                linestyle,
                y_value,
                y_mode,
                font_size,
                font_weight,
                alpha_value,
            ) in label_candidates:
                common = dict(
                    ha="center",
                    fontsize=font_size,
                    color=color,
                    weight=font_weight,
                    rotation=90,
                    alpha=alpha_value,
                )
                if y_mode in {"axes_top", "axes_peak"}:
                    txt = ax_spec.text(
                        peak_energy,
                        y_value,
                        label_text,
                        va="top",
                        transform=ax_spec.get_xaxis_transform(),
                        clip_on=True,
                        **common,
                    )
                else:
                    txt = ax_spec.text(peak_energy, y_value, label_text, va="bottom", **common)
                # Prioritize data-peak labels over top reference labels if collisions occur.
                priority = 1 if y_mode in {"data", "axes_peak"} else 0
                drawn_texts.append((txt, label_text, color, linestyle, priority))

            if drawn_texts:
                fig.canvas.draw()
                ax_bbox = ax_spec.get_window_extent()
                renderer = fig.canvas.get_renderer()
                kept_bboxes = []
                # Keep higher-priority labels first, then by x-position for stable layout.
                drawn_texts.sort(key=lambda item: (-item[4], item[0].get_position()[0]))
                for txt, label_text, color, linestyle, _ in drawn_texts:
                    txt_bbox = txt.get_window_extent(renderer=renderer)
                    out_of_bounds = (
                        txt_bbox.x0 < ax_bbox.x0
                        or txt_bbox.x1 > ax_bbox.x1
                        or txt_bbox.y0 < ax_bbox.y0
                        or txt_bbox.y1 > ax_bbox.y1
                    )
                    overlaps_kept = any(txt_bbox.overlaps(prev_bbox) for prev_bbox in kept_bboxes)

                    if out_of_bounds or overlaps_kept:
                        txt.remove()
                        key = (label_text, str(color), linestyle)
                        if key not in legend_labels:
                            legend_labels.add(key)
                            legend_handles.append(
                                Line2D(
                                    [0],
                                    [0],
                                    color=color,
                                    linestyle=linestyle,
                                    linewidth=1.5,
                                    label=label_text,
                                )
                            )
                    else:
                        kept_bboxes.append(txt_bbox)

        if legend_handles:
            overlap_legend = ax_spec.legend(
                handles=legend_handles, loc="upper right", fontsize=8, title="Overlapping Labels"
            )
            ax_spec.add_artist(overlap_legend)

        if line is not None:
            x_min, x_max = ax_spec.get_xlim()
            _ref_energies = [line] if isinstance(line, (int, float)) else list(line)
            for ref_energy in _ref_energies:
                try:
                    ref_energy = float(ref_energy)
                except (TypeError, ValueError):
                    continue
                # Do not let out-of-window reference lines change autoscaled limits.
                if x_min <= ref_energy <= x_max:
                    ax_spec.axvline(
                        ref_energy, color="black", linestyle="--", linewidth=1.2, zorder=3
                    )
            ax_spec.set_xlim(x_min, x_max)

        fig.tight_layout()
        plt.show()

        sorted_table_rows = sorted(table_rows, key=lambda item: item[0])
        print(
            f"{'Energy (keV)':<12} {'Intensity':<12} {'SNR':<8} {'Best Match':<22} {'Alt 2':<22} {'Alt 3':<22}"
        )
        print("-" * 105)
        for peak_energy, height, snr, best_match, alt_2, alt_3 in sorted_table_rows:
            print(
                f"{peak_energy:<12.3f} {height:<12.2f} {snr:<8.1f} {best_match:<22} {alt_2:<22} {alt_3:<22}"
            )
        print("-" * 105)
        print(
            f"{len(plot_peaks)} of {len(display_peaks)} peaks above "
            f"floor={floor:.1f}, snr_threshold={snr_threshold:.1f} displayed.\n"
        )

        if return_details:
            return {
                "figure": fig,
                "axes": (ax_img, ax_spec),
                "detected_elements": sorted(detected_elements),
                "element_confidence": element_confidence,
                "display_peaks": display_peaks,
                "peak_matches": peak_matches,
                "floor": floor,
                "snr_threshold": snr_threshold,
            }
        return fig, (ax_img, ax_spec)

    def _fit_mean_model_pytorch(
        self,
        energy_axis,
        spectrum_raw,
        elements_to_fit,
        peak_width,
        polynomial_background_degree,
        num_iters,
        optimizer,
        lr,
        loss_name,
        normalize_target,
        default_lr_adam,
        default_lr_lbfgs,
        verbose=False,
    ):
        """Fit a single mean spectrum using the PyTorch EDS model."""
        target = spectrum_raw
        spectrum_offset = torch.tensor(0.0, dtype=spectrum_raw.dtype, device=spectrum_raw.device)
        spectrum_scale = torch.tensor(1.0, dtype=spectrum_raw.dtype, device=spectrum_raw.device)
        if normalize_target:
            spectrum_offset = spectrum_raw.min()
            spectrum_scale = torch.clamp(spectrum_raw.max() - spectrum_offset, min=1e-8)
            target = (spectrum_raw - spectrum_offset) / spectrum_scale

        background = PolynomialBackground(
            energy_axis,
            degree=polynomial_background_degree,
        )
        peaks = GaussianPeaks(
            energy_axis,
            peak_width=peak_width,
            elements_to_fit=elements_to_fit,
        )
        model = EDSModel(peaks, background)
        model = model.to(device=energy_axis.device, dtype=energy_axis.dtype)
        if len(model.peak_model.element_names) == 0:
            raise ValueError("No elements found in the selected energy range/elements_to_fit.")

        optimizer_name = optimizer.lower()
        if optimizer_name == "adam":
            if lr is None:
                lr = default_lr_adam
            optimizer_obj = torch.optim.Adam(model.parameters(), lr=lr)
        elif optimizer_name == "lbfgs":
            if lr is None:
                lr = default_lr_lbfgs
            optimizer_obj = torch.optim.LBFGS(
                model.parameters(),
                lr=lr,
                line_search_fn="strong_wolfe",
            )
        else:
            raise ValueError("optimizer must be 'lbfgs' or 'adam'")

        loss_iter = []
        for i in range(num_iters):
            if optimizer_name == "lbfgs":

                def closure():
                    optimizer_obj.zero_grad()
                    predicted = model()
                    loss = eds_data_loss(predicted, target, loss=loss_name)
                    loss.backward()
                    return loss

                loss = optimizer_obj.step(closure)
                if not torch.is_tensor(loss):
                    with torch.no_grad():
                        loss = eds_data_loss(model(), target, loss=loss_name)
            else:
                optimizer_obj.zero_grad()
                predicted = model()
                loss = eds_data_loss(predicted, target, loss=loss_name)
                loss.backward()
                optimizer_obj.step()

            loss_iter.append(float(loss.detach().cpu().item()))
            if verbose and ((i + 1) % max(1, num_iters // 10) == 0 or i == 0):
                print(f"iter {i + 1:4d}/{num_iters}: loss={loss_iter[-1]:.6g}")

        with torch.no_grad():
            final_pred_target = model()
            if normalize_target:
                final_pred_raw = final_pred_target * spectrum_scale + spectrum_offset
            else:
                final_pred_raw = final_pred_target

        return {
            "model": model,
            "loss_history": np.asarray(loss_iter),
            "final_pred_raw": final_pred_raw.detach(),
            "spectrum_offset": spectrum_offset.detach(),
            "spectrum_scale": spectrum_scale.detach(),
        }

    def fit_spectrum_mean_pytorch(
        self,
        energy_range=None,
        elements_to_fit=None,
        peak_width=0.1,
        num_iters=1000,
        lr=None,
        polynomial_background_degree=3,
        optimizer="lbfgs",
        device=None,
    ):
        """Fit the spatially-summed mean EDS spectrum and display results.

        A convenience wrapper around :meth:`_fit_mean_model_pytorch` that
        handles device selection, energy windowing, and result visualization.

        Parameters
        ----------
        energy_range : sequence[float] | None, optional
            Two-element energy interval ``[emin, emax]`` in keV.  If ``None``,
            the full energy axis is used.
        elements_to_fit : sequence[str] | None, optional
            Element symbols to include in the fit.  If ``None``, uses keys
            from ``self.model_elements``.
        peak_width : float, optional
            Initial FWHM-like peak width in keV.
        num_iters : int, optional
            Number of optimization iterations.
        lr : float | None, optional
            Learning rate.  If ``None``, an optimizer-specific default is used.
        polynomial_background_degree : int, optional
            Degree of the polynomial background basis.
        optimizer : {"adam", "lbfgs"}, optional
            Optimizer to use.
        device : str | torch.device | None, optional
            Torch device.  If ``None``, uses CUDA when available.

        Returns
        -------
        dict
            Keys include ``loss_history``, ``fitted_spectrum``,
            ``input_spectrum``, ``background_spectrum``, ``concentrations``,
            ``element_names``, ``peak_widths``, ``energy_axis``, and
            ``fit_range``.
        """
        optimizer_name = str(optimizer).lower()
        if optimizer_name not in {"adam", "lbfgs"}:
            raise ValueError("optimizer must be 'lbfgs' or 'adam'")

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA device requested but torch.cuda.is_available() is False.")

        if elements_to_fit is None:
            if not self.model_elements:
                raise ValueError("elements_to_fit must be specified")
            elements_to_fit = list(self.model_elements.keys())
            print(f"using model_elements {elements_to_fit}")

        energy_axis_np = self.energy_axis.copy()
        energy_axis = torch.tensor(energy_axis_np, dtype=torch.float32, device=device)
        spectra = torch.tensor(self.array, dtype=torch.float32, device=device)

        if energy_range is not None:
            ind = (energy_axis >= energy_range[0]) & (energy_axis <= energy_range[1])
            energy_axis = energy_axis[ind]
            spectra = spectra[:, :, ind]
        else:
            energy_range = [float(energy_axis.min().item()), float(energy_axis.max().item())]

        print("fitting spectrum globally")
        spectrum_raw = spectra.sum((0, 1))
        mean_fit = self._fit_mean_model_pytorch(
            energy_axis=energy_axis,
            spectrum_raw=spectrum_raw,
            elements_to_fit=elements_to_fit,
            peak_width=peak_width,
            polynomial_background_degree=polynomial_background_degree,
            num_iters=num_iters,
            optimizer=optimizer_name,
            lr=lr,
            loss_name="mse",
            normalize_target=True,
            default_lr_adam=1e-3,
            default_lr_lbfgs=1.0,
            verbose=True,
        )

        model = mean_fit["model"]
        loss_history = mean_fit["loss_history"]
        spectrum_offset = mean_fit["spectrum_offset"]
        spectrum_scale = mean_fit["spectrum_scale"]
        with torch.no_grad():
            final_pred = mean_fit["final_pred_raw"].cpu().numpy()
            shell_concs = (
                nn.functional.softplus(model.peak_model.concentrations).detach().cpu().numpy()
            )
            shell_names = list(model.peak_model.shell_group_names)
            shell_element_indices = (
                model.peak_model.shell_group_element_indices.detach().cpu().numpy()
            )
            concs = np.zeros(len(model.peak_model.element_names), dtype=np.float32)
            np.add.at(concs, shell_element_indices, shell_concs)
            final_fwhm = (
                torch.nn.functional.softplus(model.peak_model.peak_width_by_peak)
                .detach()
                .cpu()
                .numpy()
            )
            background_fit = (
                (model.background_model().detach() * spectrum_scale + spectrum_offset)
                .cpu()
                .numpy()
            )

        print(
            f"\nFinal: width median={np.median(final_fwhm):.3f} keV, "
            f"min={final_fwhm.min():.3f}, max={final_fwhm.max():.3f}"
        )

        top_n = max(10, len(elements_to_fit) if elements_to_fit is not None else 0)
        sorted_indices = np.argsort(concs)[::-1]
        print("\nTop elements:")
        for i, idx in enumerate(sorted_indices[:top_n], 1):
            elem = model.peak_model.element_names[idx]
            conc = concs[idx]
            print(f"{i:2d}. {elem:2s}: {conc:.3f}")

        shell_top_n = max(10, min(len(shell_names), top_n))
        shell_sorted_indices = np.argsort(shell_concs)[::-1]
        print("\nTop edge groups:")
        for i, idx in enumerate(shell_sorted_indices[:shell_top_n], 1):
            shell_name = shell_names[idx]
            shell_conc = shell_concs[idx]
            print(f"{i:2d}. {shell_name:>6s}: {shell_conc:.3f}")

        energy_axis_plot = energy_axis.detach().cpu().numpy()
        spectrum_raw_plot = spectrum_raw.detach().cpu().numpy()
        fig, ax = plt.subplots(2, 1, figsize=(10, 6))
        ax[0].plot(np.arange(loss_history.shape[0]), loss_history, color="k")
        ax[0].set_title("loss")
        ax[0].set_xlabel("iterations")
        ax[0].set_ylabel("loss")
        ax[0].set_yscale("log")

        ax[1].plot(energy_axis_plot, spectrum_raw_plot, "k-", label="Data", linewidth=1)
        ax[1].plot(energy_axis_plot, final_pred, "r-", label="Fit", linewidth=2)
        ax[1].plot(
            energy_axis_plot,
            background_fit,
            "b--",
            label="Background",
            linewidth=1.5,
        )
        ax[1].set_xlim(energy_range[0], energy_range[1])
        ax[1].legend()
        ax[1].set_title("fit spectrum")
        ax[1].set_xlabel("Energy (keV)")
        ax[1].set_ylabel("Counts")
        plt.tight_layout()
        plt.show()

        return {
            "loss_history": loss_history,
            "fitted_spectrum": final_pred,
            "input_spectrum": spectrum_raw_plot,
            "background_spectrum": background_fit,
            "concentrations": concs,
            "element_names": model.peak_model.element_names,
            "edge_concentrations": shell_concs,
            "edge_names": shell_names,
            "edge_element_indices": shell_element_indices,
            "peak_widths": final_fwhm,
            "energy_axis": energy_axis_plot,
            "fit_range": energy_range,
        }

    def fit_spectrum_pytorch(
        self,
        energy_range=None,
        elements_to_fit=None,
        peak_width=0.1,
        num_iters=300,
        num_iters_global=200,
        polynomial_background_degree=3,
        optimizer_global="lbfgs",
        optimizer_local="lbfgs",
        loss_global=None,
        loss_local="poisson",
        freeze_peak_width=True,
        spatial_lambda=0.0,
        min_total_counts=0.0,
        verbose=True,
        fit_mean_only=False,
        show_plot=True,
        lr_global=None,
        lr_local=None,
        device=None,
        constrain_background=0.1,
    ):
        """Fit EDS spectra using a PyTorch model.

        Supports two workflows:
        - Mean-only fitting (`fit_mean_only=True`): fit a single spectrum formed by
          summing over all spatial pixels.
        - Global + local fitting (`fit_mean_only=False`): fit a global mean model,
          then refine concentrations/background per pixel across the full cube.

        Parameters
        ----------
        energy_range : sequence[float] | None, optional
            Two-element energy interval ``[emin, emax]`` in keV used for fitting.
            If ``None``, the full energy axis is used.
        elements_to_fit : sequence[str] | None, optional
            Element symbols (or model-supported element labels) to include in the
            fit. If ``None``, uses keys from ``self.model_elements``.
        peak_width : float, optional
            Initial peak width (FWHM-like parameter in keV) for model peaks.
        num_iters : int, optional
            Number of optimization iterations for mean-only mode, or local
            per-pixel refinement iterations in full-cube mode.
        num_iters_global : int, optional
            Number of iterations for the global/mean stage in full-cube mode.
        polynomial_background_degree : int, optional
            Degree of polynomial background basis.
        optimizer_global : {"adam", "lbfgs"}, optional
            Optimizer for the global/mean stage.
        optimizer_local : {"adam", "lbfgs"}, optional
            Optimizer for per-pixel local fitting.
        loss_global : {"poisson", "mse"} | None, optional
            Global-stage data term. If ``None``, defaults to ``"mse"`` for
            mean-only mode and ``"poisson"`` otherwise.
        loss_local : {"poisson", "mse"}, optional
            Local-stage data term (ignored when ``fit_mean_only=True``).
        freeze_peak_width : bool, optional
            If ``True``, keep peak widths fixed during local fitting.
        spatial_lambda : float, optional
            L2 spatial smoothness weight applied to abundance maps during local
            fitting. Must be non-negative.
        min_total_counts : float, optional
            Minimum per-pixel integrated counts required for a pixel to
            participate in local fitting.
        verbose : bool, optional
            If ``True``, print optimization progress.
        fit_mean_only : bool, optional
            If ``True``, run only the mean-spectrum fit and skip per-pixel
            refinement.
        show_plot : bool, optional
            If ``True``, display diagnostic plots.
        lr_global : float | None, optional
            Learning rate for the global optimizer. If ``None``, an optimizer-
            specific default is used.
        lr_local : float | None, optional
            Learning rate for the local optimizer. If ``None``, an optimizer-
            specific default is used.
        device : str | torch.device | None, optional
            Torch device for fitting (for example ``"cpu"`` or ``"cuda"``).
            If ``None``, uses CUDA when available, otherwise CPU.
        constrain_background : float, optional
            Background prior weight used in local fitting to keep per-pixel
            background coefficients close to the globally optimized background.
            Set to ``0`` to disable. This is only used when
            ``fit_mean_only=False``.

        Returns
        -------
        dict
            Fit results. Contents depend on the selected mode.

            Mean-only mode (``fit_mean_only=True``) returns keys:
            ``loss_history``, ``fitted_spectrum``, ``input_spectrum``,
            ``background_spectrum``, ``concentrations``, ``element_names``,
            ``edge_concentrations``, ``edge_names``, ``edge_element_indices``,
            ``peak_widths``, ``energy_axis``, ``fit_range``.

            Full-cube mode (``fit_mean_only=False``) returns keys:
            ``abundance_maps``, ``element_names``, ``peak_widths``,
            ``loss_history``, ``global_loss_history``, ``valid_pixel_mask``,
            ``energy_axis``, ``input_spectrum``, ``fitted_spectrum``,
            ``background_spectrum``, ``input_spectrum_all_pixels``,
            ``fitted_spectrum_all_pixels``, ``background_spectrum_all_pixels``,
            ``fit_range``.

        Raises
        ------
        TypeError
            If ``constrain_background`` is not numeric (for example ``bool``).
        ValueError
            If optimizer/loss names are invalid, ``spatial_lambda < 0``, CUDA is
            requested but unavailable, ``constrain_background < 0``, or no pixels
            satisfy ``min_total_counts``.
        """

        def _normalize_choice(name, param_name, allowed_values):
            name_norm = str(name).lower()
            if name_norm not in allowed_values:
                allowed_display = "', '".join(sorted(allowed_values))
                raise ValueError(f"{param_name} must be '{allowed_display}'")
            return name_norm

        effective_optimizer_global = _normalize_choice(
            optimizer_global, "optimizer_global", {"adam", "lbfgs"}
        )
        effective_optimizer_local = _normalize_choice(
            optimizer_local, "optimizer_local", {"adam", "lbfgs"}
        )
        effective_loss_global = (
            _normalize_choice(loss_global, "loss_global", {"poisson", "mse"})
            if loss_global is not None
            else ("mse" if fit_mean_only else "poisson")
        )
        effective_loss_local = (
            _normalize_choice(loss_local, "loss_local", {"poisson", "mse"})
            if not fit_mean_only
            else None
        )

        if spatial_lambda < 0:
            raise ValueError("spatial_lambda must be >= 0")

        if isinstance(constrain_background, bool):
            raise TypeError("constrain_background must be a non-negative float.")
        try:
            background_prior_lambda = float(constrain_background)
        except (TypeError, ValueError) as exc:
            raise TypeError("constrain_background must be a non-negative float.") from exc
        if background_prior_lambda < 0:
            raise ValueError("constrain_background must be >= 0")

        if elements_to_fit is None:
            if not self.model_elements:
                raise ValueError("elements_to_fit must be specified")
            elements_to_fit = list(self.model_elements.keys())
            if verbose:
                print(f"using model_elements {elements_to_fit}")

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA device requested but torch.cuda.is_available() is False.")

        effective_lr_global = lr_global
        effective_lr_local = lr_local

        energy_axis_np = self.energy_axis.copy()
        energy_axis = torch.tensor(energy_axis_np, dtype=torch.float32, device=device)
        spectra = torch.tensor(self.array, dtype=torch.float32, device=device)

        if energy_range is not None:
            ind = (energy_axis >= energy_range[0]) & (energy_axis <= energy_range[1])
            energy_axis = energy_axis[ind]
            spectra = spectra[:, :, ind]
        else:
            energy_range = [float(energy_axis.min().item()), float(energy_axis.max().item())]

        if fit_mean_only:
            if verbose:
                print("fitting spectrum globally")
            spectrum_raw = spectra.sum((0, 1))
            mean_fit = self._fit_mean_model_pytorch(
                energy_axis=energy_axis,
                spectrum_raw=spectrum_raw,
                elements_to_fit=elements_to_fit,
                peak_width=peak_width,
                polynomial_background_degree=polynomial_background_degree,
                num_iters=num_iters,
                optimizer=effective_optimizer_global,
                lr=effective_lr_global,
                loss_name=effective_loss_global,
                normalize_target=True,
                default_lr_adam=1e-3,
                default_lr_lbfgs=1.0,
                verbose=verbose,
            )

            model = mean_fit["model"]
            loss_history = mean_fit["loss_history"]
            spectrum_offset = mean_fit["spectrum_offset"]
            spectrum_scale = mean_fit["spectrum_scale"]
            with torch.no_grad():
                final_pred = mean_fit["final_pred_raw"].cpu().numpy()
                shell_concs = (
                    nn.functional.softplus(model.peak_model.concentrations).detach().cpu().numpy()
                )
                shell_element_indices = (
                    model.peak_model.shell_group_element_indices.detach().cpu().numpy()
                )
                concs = np.zeros(len(model.peak_model.element_names), dtype=np.float32)
                np.add.at(concs, shell_element_indices, shell_concs)
                final_fwhm = (
                    torch.nn.functional.softplus(model.peak_model.peak_width_by_peak)
                    .detach()
                    .cpu()
                    .numpy()
                )
                background_fit = (
                    (model.background_model().detach() * spectrum_scale + spectrum_offset)
                    .cpu()
                    .numpy()
                )

            print(
                f"\nFinal: width median={np.median(final_fwhm):.3f} keV, "
                f"min={final_fwhm.min():.3f}, max={final_fwhm.max():.3f}"
            )

            top_n = max(10, len(elements_to_fit) if elements_to_fit is not None else 0)
            sorted_indices = np.argsort(concs)[::-1]
            print("\nTop elements:")
            for i, idx in enumerate(sorted_indices[:top_n], 1):
                elem = model.peak_model.element_names[idx]
                conc = concs[idx]
                print(f"{i:2d}. {elem:2s}: {conc:.3f}")

            if show_plot:
                energy_axis_plot = energy_axis.detach().cpu().numpy()
                spectrum_raw_plot = spectrum_raw.detach().cpu().numpy()
                fig, ax = plt.subplots(2, 1, figsize=(10, 6))
                ax[0].plot(np.arange(loss_history.shape[0]), loss_history, color="k")
                ax[0].set_title("loss")
                ax[0].set_xlabel("iterations")
                ax[0].set_ylabel("loss")
                ax[0].set_yscale("log")

                ax[1].plot(energy_axis_plot, spectrum_raw_plot, "k-", label="Data", linewidth=1)
                ax[1].plot(energy_axis_plot, final_pred, "r-", label="Fit", linewidth=2)
                ax[1].plot(
                    energy_axis_plot,
                    background_fit,
                    "b--",
                    label="Background",
                    linewidth=1.5,
                )
                ax[1].set_xlim(energy_range[0], energy_range[1])
                ax[1].legend()
                ax[1].set_title("fit spectrum")
                ax[1].set_xlabel("Energy (keV)")
                ax[1].set_ylabel("Counts")
                plt.tight_layout()
                plt.show()

            return {
                "loss_history": loss_history,
                "fitted_spectrum": final_pred,
                "input_spectrum": spectrum_raw.detach().cpu().numpy(),
                "background_spectrum": background_fit,
                "concentrations": concs,
                "element_names": model.peak_model.element_names,
                "edge_concentrations": shell_concs,
                "edge_names": list(model.peak_model.shell_group_names),
                "edge_element_indices": shell_element_indices,
                "peak_widths": final_fwhm,
                "energy_axis": energy_axis.detach().cpu().numpy(),
                "fit_range": energy_range,
            }

        scan_row, scan_col, n_energy = spectra.shape
        n_pixels = scan_row * scan_col
        spectra_flat = spectra.reshape(n_pixels, n_energy)

        total_counts = spectra_flat.sum(dim=1)
        valid_pixel_mask = total_counts >= float(min_total_counts)
        if not torch.any(valid_pixel_mask):
            raise ValueError("No pixels satisfy min_total_counts. Lower threshold and retry.")

        mean_spectrum = spectra_flat[valid_pixel_mask].mean(dim=0)

        if verbose:
            print("fitting spectrum globally")
        global_fit = self._fit_mean_model_pytorch(
            energy_axis=energy_axis,
            spectrum_raw=mean_spectrum,
            elements_to_fit=elements_to_fit,
            peak_width=peak_width,
            polynomial_background_degree=polynomial_background_degree,
            num_iters=num_iters_global,
            optimizer=effective_optimizer_global,
            lr=effective_lr_global,
            loss_name=effective_loss_global,
            normalize_target=True,
            default_lr_adam=1e-3,
            default_lr_lbfgs=1.0,
            verbose=verbose,
        )
        global_model = global_fit["model"]
        global_loss_history = global_fit["loss_history"]
        global_scale = global_fit["spectrum_scale"].detach()
        global_offset = global_fit["spectrum_offset"].detach()
        global_fitted_spectrum = global_fit["final_pred_raw"].detach().cpu().numpy()

        n_elements = len(global_model.peak_model.element_names)
        with torch.no_grad():
            global_conc_shell = (
                nn.functional.softplus(global_model.peak_model.concentrations).detach()
                * global_scale
            )
            shell_element_indices = global_model.peak_model.shell_group_element_indices
            global_conc = torch.zeros(
                n_elements,
                dtype=global_conc_shell.dtype,
                device=global_conc_shell.device,
            )
            global_conc.index_add_(0, shell_element_indices, global_conc_shell)
            global_bg_coeffs = global_model.background_model.coeffs.detach() * global_scale
            if global_bg_coeffs.numel() > 0:
                global_bg_coeffs = global_bg_coeffs.clone()
                global_bg_coeffs[0] = global_bg_coeffs[0] + global_offset
            global_peak_width_params = global_model.peak_model.peak_width_by_peak.detach().clone()

        peak_energies = global_model.peak_model.peak_energies
        peak_weights = global_model.peak_model.peak_weights
        peak_element_indices = global_model.peak_model.peak_element_indices
        energy_step = float(global_model.peak_model.energy_step)

        background_basis = polynomial_energy_basis(
            energy_axis, degree=polynomial_background_degree
        )

        mean_total = torch.clamp(mean_spectrum.sum(), min=1e-8)
        pixel_scales = (total_counts / mean_total).unsqueeze(1)
        conc_init = torch.clamp(
            global_conc.unsqueeze(0) * pixel_scales,
            min=1e-3,
        )
        conc_init = torch.clamp(
            conc_init * (1.0 + 0.02 * torch.randn_like(conc_init)),
            min=1e-3,
        )

        conc_logits = nn.Parameter(inverse_softplus(conc_init))
        bg_coeffs_init = global_bg_coeffs.unsqueeze(0).repeat(n_pixels, 1) * pixel_scales
        bg_coeffs = nn.Parameter(bg_coeffs_init.clone())

        if freeze_peak_width:
            peak_width_params = global_peak_width_params
        else:
            peak_width_params = nn.Parameter(global_peak_width_params.clone())

        if freeze_peak_width:
            element_basis = build_element_basis(
                energy_axis=energy_axis,
                peak_energies=peak_energies,
                peak_weights=peak_weights,
                peak_element_indices=peak_element_indices,
                peak_width_by_peak=peak_width_params,
                n_elements=n_elements,
                energy_step=energy_step,
            )

        trainable_params = [conc_logits, bg_coeffs]
        if not freeze_peak_width:
            trainable_params.append(peak_width_params)

        local_lr = (
            effective_lr_local
            if effective_lr_local is not None
            else (0.05 if effective_optimizer_local == "adam" else 1.0)
        )

        if effective_optimizer_local == "adam":
            adam_param_groups = [{"params": [conc_logits], "lr": local_lr}]
            adam_param_groups.append({"params": [bg_coeffs], "lr": local_lr})
            if not freeze_peak_width:
                adam_param_groups.append({"params": [peak_width_params], "lr": local_lr})
            local_opt = torch.optim.Adam(adam_param_groups)
        else:
            local_opt = torch.optim.LBFGS(
                trainable_params,
                lr=local_lr,
                line_search_fn="strong_wolfe",
            )

        loss_history = []

        def _forward_model():
            basis = (
                element_basis
                if freeze_peak_width
                else build_element_basis(
                    energy_axis=energy_axis,
                    peak_energies=peak_energies,
                    peak_weights=peak_weights,
                    peak_element_indices=peak_element_indices,
                    peak_width_by_peak=peak_width_params,
                    n_elements=n_elements,
                    energy_step=energy_step,
                )
            )
            conc = nn.functional.softplus(conc_logits)  # (P, n_elements)
            peaks_pred = conc @ basis.t()
            bg_pred = bg_coeffs @ background_basis
            predicted = torch.clamp(peaks_pred + bg_pred, min=1e-8, max=1e8)
            return predicted, conc, bg_pred

        def _background_regularization():
            if background_prior_lambda <= 0:
                return bg_coeffs.new_tensor(0.0)

            coeff_init_eval = bg_coeffs_init[valid_pixel_mask]
            coeff_eval = bg_coeffs[valid_pixel_mask]
            coeff_scale = torch.clamp(coeff_init_eval.abs().mean(), min=1e-8)
            reg_prior = ((coeff_eval - coeff_init_eval) / coeff_scale).pow(2).mean()
            return background_prior_lambda * reg_prior

        def _local_loss(pred_local, conc_local):
            local_scale = torch.clamp(global_scale, min=1e-8)
            pred_eval = pred_local[valid_pixel_mask] / local_scale
            target_eval = spectra_flat[valid_pixel_mask] / local_scale

            loss_data = eds_data_loss(
                pred_eval,
                target_eval,
                loss=effective_loss_local,
            )
            loss_total = loss_data + _background_regularization()

            if spatial_lambda <= 0:
                return loss_total

            conc_maps = conc_local.view(scan_row, scan_col, n_elements).permute(2, 0, 1)
            conc_maps = conc_maps / torch.clamp(global_scale, min=1e-8)
            loss_smooth = abundance_smoothness_l2(conc_maps)
            return loss_total + spatial_lambda * loss_smooth

        if verbose:
            print("fitting spectrum position-by-position")
        for i in range(num_iters):
            if effective_optimizer_local == "lbfgs":

                def _local_closure():
                    local_opt.zero_grad()
                    pred_local, conc_local, _bg_local = _forward_model()
                    loss_total = _local_loss(pred_local, conc_local)
                    loss_total.backward()
                    return loss_total

                loss_value = local_opt.step(_local_closure)
                if not torch.is_tensor(loss_value):
                    with torch.no_grad():
                        pred_local, conc_local, _bg_local = _forward_model()
                        loss_value = _local_loss(pred_local, conc_local)
            else:
                local_opt.zero_grad()
                pred_local, conc_local, _bg_local = _forward_model()
                loss_value = _local_loss(pred_local, conc_local)
                loss_value.backward()
                local_opt.step()

            loss_history.append(float(loss_value.detach().cpu().item()))
            if verbose and ((i + 1) % max(1, num_iters // 10) == 0 or i == 0):
                print(f"iter {i + 1:4d}/{num_iters}: loss={loss_history[-1]:.6g}")

        with torch.no_grad():
            pred_final, conc_final, bg_final = _forward_model()
            mean_input_spectrum = spectra_flat[valid_pixel_mask].mean(dim=0).cpu().numpy()
            mean_fitted_spectrum = pred_final[valid_pixel_mask].mean(dim=0).cpu().numpy()
            mean_background_spectrum = bg_final[valid_pixel_mask].mean(dim=0).cpu().numpy()
            mean_input_spectrum_all = spectra_flat.mean(dim=0).cpu().numpy()
            mean_fitted_spectrum_all = pred_final.mean(dim=0).cpu().numpy()
            mean_background_spectrum_all = bg_final.mean(dim=0).cpu().numpy()

            abundance_maps = (
                conc_final.view(scan_row, scan_col, n_elements).permute(2, 0, 1).cpu().numpy()
            )
            peak_widths = nn.functional.softplus(peak_width_params).detach().cpu().numpy()

        pytorch_spectrum_images = self._build_pytorch_spectrum_images(
            abundance_maps=abundance_maps,
            element_names=list(global_model.peak_model.element_names),
        )
        if hasattr(self, "_spectrum_images_pytorch"):
            self._spectrum_images_pytorch = {
                **self._spectrum_images_pytorch,
                **pytorch_spectrum_images,
            }
        else:
            self._spectrum_images_pytorch = {}
            self._spectrum_images_pytorch = {
                **self._spectrum_images_pytorch,
                **pytorch_spectrum_images,
            }

        loss_history_array = np.asarray(loss_history)
        energy_axis_np = energy_axis.cpu().numpy()

        if show_plot:
            fig, ax = plt.subplots(1, 1, figsize=(8, 4))
            global_x = np.arange(global_loss_history.shape[0])
            local_x = np.arange(loss_history_array.shape[0]) + global_loss_history.shape[0]
            ax.plot(
                global_x,
                global_loss_history,
                "b-",
                label="global",
            )
            ax.plot(
                local_x,
                loss_history_array,
                "r-",
                label="local",
            )
            ax.axvline(
                x=global_loss_history.shape[0] - 0.5,
                color="gray",
                linestyle="--",
                linewidth=1.0,
                label="switch",
            )
            ax.set_title("loss")
            ax.set_xlabel("iterations")
            ax.set_ylabel("loss")
            ax.set_yscale("log")
            ax.legend()
            plt.tight_layout()
            plt.show()

            fig, ax = plt.subplots(1, 1, figsize=(10, 4))
            ax.plot(energy_axis_np, mean_input_spectrum, "k-", label="Data", linewidth=1)
            ax.plot(
                energy_axis_np,
                global_fitted_spectrum,
                color="cyan",
                label="Global fit",
                linewidth=2.5,
            )
            ax.plot(energy_axis_np, mean_fitted_spectrum, "r-", label="Fit", linewidth=2.5)
            ax.plot(
                energy_axis_np,
                mean_background_spectrum,
                "b--",
                label="Background",
                linewidth=2.5,
            )
            ax.set_xlim(energy_range[0], energy_range[1])
            ax.legend()
            ax.set_title("fit spectrum after local fitting (valid-pixel averaged)")
            ax.set_xlabel("Energy (keV)")
            ax.set_ylabel("Counts")
            plt.tight_layout()
            plt.show()

            self.show_spectrum_images(method="fit")

        return {
            "abundance_maps": abundance_maps,
            "element_names": global_model.peak_model.element_names,
            "peak_widths": peak_widths,
            "loss_history": loss_history_array,
            "global_loss_history": np.asarray(global_loss_history),
            "valid_pixel_mask": valid_pixel_mask.view(scan_row, scan_col).cpu().numpy(),
            "energy_axis": energy_axis_np,
            "input_spectrum": mean_input_spectrum,
            "fitted_spectrum": mean_fitted_spectrum,
            "background_spectrum": mean_background_spectrum,
            "input_spectrum_all_pixels": mean_input_spectrum_all,
            "fitted_spectrum_all_pixels": mean_fitted_spectrum_all,
            "background_spectrum_all_pixels": mean_background_spectrum_all,
            "fit_range": energy_range,
            "spectrum_images_pytorch": self._spectrum_images_pytorch,
        }

    def calculate_background_polynomial(
        self,
        spectrum,
        energy_axis=None,
        degree=3,
        percentile=10,
        window_size=50,
    ):
        """
        Fit an EDS continuum background with a polynomial power series in energy.

        A rolling low-percentile envelope is used as the fit target so sharp
        characteristic X-ray peaks do not dominate the continuum fit.
        """

        spectrum = np.asarray(spectrum, dtype=float)
        if spectrum.ndim != 1:
            raise ValueError("spectrum must be a 1D array")
        if spectrum.size == 0:
            raise ValueError("spectrum must contain at least one channel")

        if energy_axis is None:
            energy_axis = np.asarray(self.energy_axis, dtype=float)
            if energy_axis.shape != spectrum.shape:
                energy_axis = float(self.origin[2]) + float(self.sampling[2]) * np.arange(
                    spectrum.size, dtype=float
                )
        else:
            energy_axis = np.asarray(energy_axis, dtype=float)
        if energy_axis.shape != spectrum.shape:
            raise ValueError("energy_axis must have the same shape as spectrum")

        if isinstance(degree, bool):
            raise TypeError("degree must be a non-negative integer")
        try:
            degree = int(degree)
        except (TypeError, ValueError) as exc:
            raise TypeError("degree must be a non-negative integer") from exc
        if degree < 0:
            raise ValueError("degree must be >= 0")

        try:
            percentile = float(percentile)
        except (TypeError, ValueError) as exc:
            raise TypeError("percentile must be a number between 0 and 100") from exc
        if percentile < 0 or percentile > 100:
            raise ValueError("percentile must be between 0 and 100")

        if isinstance(window_size, bool):
            raise TypeError("window_size must be a positive integer")
        try:
            window_size = int(window_size)
        except (TypeError, ValueError) as exc:
            raise TypeError("window_size must be a positive integer") from exc
        if window_size < 1:
            raise ValueError("window_size must be >= 1")
        window_size = min(window_size, spectrum.size)

        finite = np.isfinite(spectrum) & np.isfinite(energy_axis)
        if np.count_nonzero(finite) < degree + 1:
            raise ValueError("not enough finite spectrum points for the requested degree")

        half_window = window_size // 2
        envelope = np.full_like(spectrum, np.nan, dtype=float)
        for channel in range(spectrum.size):
            start = max(0, channel - half_window)
            end = min(spectrum.size, channel + half_window + 1)
            values = spectrum[start:end]
            values = values[np.isfinite(values)]
            if values.size:
                envelope[channel] = np.percentile(values, percentile)

        fit_mask = finite & np.isfinite(envelope)
        if np.count_nonzero(fit_mask) < degree + 1:
            raise ValueError("not enough background fit points for the requested degree")

        fit_energy = energy_axis[fit_mask]
        fit_counts = envelope[fit_mask]
        energy_min = float(np.min(fit_energy))
        energy_span = float(np.max(fit_energy) - energy_min)
        if energy_span <= 0:
            if degree != 0:
                raise ValueError("energy_axis must span more than one value for degree > 0")
            return np.full_like(spectrum, max(float(np.median(fit_counts)), 0.0), dtype=float)

        # Scaling improves conditioning; this remains a polynomial in energy.
        def scaled_energy(energy):
            return 2.0 * (np.asarray(energy, dtype=float) - energy_min) / energy_span - 1.0

        def polynomial_background(energy, *coefficients):
            energy_scaled = scaled_energy(energy)
            background = np.zeros_like(energy_scaled, dtype=float)
            for power, coefficient in enumerate(coefficients):
                background += coefficient * (energy_scaled**power)
            return background

        scaled_fit_energy = scaled_energy(fit_energy)
        initial_coefficients = np.polynomial.polynomial.polyfit(
            scaled_fit_energy,
            fit_counts,
            deg=degree,
        )
        try:
            coefficients, _ = curve_fit(
                polynomial_background,
                fit_energy,
                fit_counts,
                p0=initial_coefficients,
                maxfev=10000,
            )
        except (RuntimeError, ValueError, FloatingPointError):
            coefficients = initial_coefficients

        background = polynomial_background(energy_axis, *coefficients)
        finite_counts = spectrum[finite]
        max_count = max(float(np.max(finite_counts)), float(np.max(fit_counts)), 0.0)
        background = np.nan_to_num(background, nan=0.0, posinf=max_count, neginf=0.0)
        return np.maximum(background, 0.0)

    def calculate_background_powerlaw(self, spectrum, *args, **kwargs):
        """Compatibility wrapper for the EDS polynomial background fit."""
        return self.calculate_background_polynomial(spectrum, *args, **kwargs)
