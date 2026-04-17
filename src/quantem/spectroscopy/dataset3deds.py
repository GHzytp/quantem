import re
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from matplotlib.lines import Line2D
from numpy.typing import NDArray
from scipy.signal import find_peaks

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
    where the data consists of a 3D array with dimensions (energy, scan_y, scan_x).
    The first dimension represents the energy, while the latter
    two dimensions represent real space sampling.

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
        """Initialize a 3D EDS dataset.

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

        self.dataset_type = "eds"

    @staticmethod
    def _normalize_specs(specs, param_name="spec", allow_none=False):
        if specs is None and allow_none:
            return None
        if isinstance(specs, str):
            return [s.strip() for s in specs.split(",") if s.strip()]
        if isinstance(specs, (list, tuple, set)):
            out = []
            for item in specs:
                out.extend([s.strip() for s in str(item).split(",") if s.strip()])
            return out
        raise TypeError(
            f"{param_name} must be {'None, ' if allow_none else ''}a string or a sequence of strings"
        )

    @staticmethod
    def _normalize_token(text):
        return re.sub(r"[^a-z0-9]", "", str(text).lower())

    @staticmethod
    def _ordered_element_keys(all_info):
        return sorted([str(key) for key in all_info], key=lambda k: (-len(k), k))

    @classmethod
    def _resolve_element_from_label(cls, label, ordered_elements):
        label_str = str(label)
        for element_name in ordered_elements:
            if label_str.startswith(element_name):
                return element_name
        m = re.match(r"^[A-Z][a-z]?", label_str)
        return m.group(0) if m else None

    @classmethod
    def _ensure_element_info(cls):
        """Load and return the element->lines info dict."""
        if cls.element_info is None:
            cls.load_element_info()
        return cls.element_info or {}

    @classmethod
    def _parse_element_selectors(cls, specs, *, allow_none=False, param_name="spec"):
        """Parse selectors like 'Fe', 'FeK', 'FeKa1' into {element: None|set[tokens]}."""
        tokens = cls._normalize_specs(specs, param_name=param_name, allow_none=allow_none)
        if tokens is None:
            return None

        info = cls._ensure_element_info()
        ordered = cls._ordered_element_keys(info)
        out: dict[str, set[str] | None] = {}

        for raw in tokens:
            compact = re.sub(r"[\s_-]+", "", str(raw).strip())
            if not compact:
                continue

            element = next((k for k in ordered if compact.lower().startswith(k.lower())), None)
            if element is None:
                raise ValueError(f"Could not resolve element from specifier '{raw}'")

            suffix = compact[len(element) :]
            if not suffix:
                out[element] = None
            else:
                out.setdefault(element, set())
                if out[element] is not None:
                    out[element].add(str(suffix))

        return out or None

    @staticmethod
    def _canonical_line_name(line_name: str) -> str:
        return str(line_name).split("__", 1)[0]

    @classmethod
    def _iter_selected_lines(cls, element: str, suffix: str, *, raw_spec: str):
        """Yield (line_name, line_info) for an element based on suffix matching."""
        info = cls._ensure_element_info()
        lines = info.get(element) or {}
        if not isinstance(lines, dict) or not lines:
            raise ValueError(f"No X-ray lines found for element '{element}'")

        if not suffix:
            yield from lines.items()
            return

        suffix_norm = cls._normalize_token(suffix)
        if not suffix_norm:
            raise ValueError(f"Could not parse line/edge token from '{raw_spec}'")

        exact, prefix = [], []
        for ln, li in lines.items():
            base = cls._canonical_line_name(str(ln))
            ln_norm = cls._normalize_token(base)
            if ln_norm == suffix_norm:
                exact.append((ln, li))
            if ln_norm.startswith(suffix_norm):
                prefix.append((ln, li))

        chosen = exact or prefix
        if not chosen:
            raise ValueError(
                f"No X-ray lines matched specifier '{raw_spec}' for element '{element}'"
            )
        yield from chosen

    @classmethod
    def _group_labels_by_element(cls, labels: list[str]):
        info = cls._ensure_element_info()
        ordered = cls._ordered_element_keys(info)
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
        """Select labels from available spectrum-image labels using selector semantics."""
        sel = str(selector).strip()
        if not sel:
            return []

        lower_map = {lbl.lower(): lbl for lbl in labels}
        if sel.lower() in lower_map:
            return [lower_map[sel.lower()]]

        elem_map = {elem.lower(): elem for elem in labels_by_element}
        if sel.lower() in elem_map:
            return list(labels_by_element[elem_map[sel.lower()]])

        compact = cls._normalize_token(sel)
        return [lbl for lbl in labels if cls._normalize_token(lbl).startswith(compact)]

    @staticmethod
    def _line_shell(line_name: str) -> str:
        line_upper = str(line_name).upper()
        if line_upper.startswith("K"):
            return "K"
        if line_upper.startswith("L"):
            return "L"
        if line_upper.startswith("M"):
            return "M"
        return "?"

    @staticmethod
    def _peak_confidence(
        snr_value: float, line_weight: float, distance_value: float, tolerance: float
    ) -> float:
        # Use a Gaussian distance likelihood: exp(-0.5 * (d/sigma)^2).
        # sigma = tolerance / 3 so the hard cutoff sits at ~3sigma,
        # giving a steep penalty for distance while tolerance stays as a
        # generous search window.  This prevents heavy-element Ma1 lines
        # (weight 1.0) from overshadowing lighter-element K-lines at much
        # closer distances (e.g. Os Ma1 vs P Ka1 near 2 keV).
        sigma = max(float(tolerance) / 3.0, 1e-9)
        snr_term = np.log1p(max(float(snr_value), 0.0))
        weight_term = max(float(line_weight), 0.0)
        distance_term = np.exp(-0.5 * (float(distance_value) / sigma) ** 2)
        return snr_term * weight_term * distance_term

    @staticmethod
    def _line_matches_selector(line_name: str, selector: str) -> bool:
        line = str(line_name).strip().lower()
        token = str(selector).strip().lower()
        if token in {"k", "l", "m"}:
            return line.startswith(token)
        return token in line

    @classmethod
    def _line_allowed_for_element(
        cls, element_name: str, line_name: str, edge_filters=None
    ) -> bool:
        if edge_filters is None:
            return True
        selectors = edge_filters.get(str(element_name))
        if selectors is None:
            return True
        return any(cls._line_matches_selector(line_name, token) for token in selectors)

    def x_ray_lookup(
        self, spec: str | list[str] | tuple[str, ...] | set[str]
    ) -> tuple[np.ndarray, np.ndarray, list[str]]:
        """Lookup EDS X-ray lines for element, shell, or specific line specifiers."""
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

                energy_raw = line_info.get("energy (keV)", line_info.get("energy"))
                try:
                    energy = float(energy_raw)
                except (TypeError, ValueError):
                    continue

                try:
                    weight = float(line_info.get("weight", 0.0))
                except (TypeError, ValueError):
                    weight = 0.0

                canonical = type(self)._canonical_line_name(str(line_name))
                rows.append((f"{element}{canonical}", energy, weight))

        if not rows:
            raise ValueError(f"No X-ray lines matched specifier(s): {specs}")

        rows.sort(key=lambda t: (t[1], -t[2], t[0]))
        seen = set()
        unique = []
        for lbl, e, w in rows:
            k = (lbl, round(float(e), 12), round(float(w), 12))
            if k in seen:
                continue
            seen.add(k)
            unique.append((lbl, e, w))

        energies = np.asarray([e for _, e, _ in unique], dtype=float)
        weights = np.asarray([w for _, _, w in unique], dtype=float)
        labels = [lbl for lbl, _, _ in unique]
        return energies, weights, labels

    def generage_spectrum_images(self, elements=None, width=0.15, return_maps=False, show=True):
        if elements is None:
            if self.model_elements is None:
                raise ValueError("elements must be specified")
            elements = list(self.model_elements.keys())
            print(f"using model_elements {elements}")

        energies, _, labels = self.x_ray_lookup(elements)
        energy_max = self.energy_axis.max()
        energy_min = self.energy_axis.min()
        ind = np.logical_and(energies > energy_min, energies < energy_max)
        energies = energies[ind]
        labels = [label for label, keep in zip(labels, ind) if keep]

        energy_axis = self.energy_axis.copy()
        energy_axis_2d = energy_axis[:, None]
        energies_2d = (energies)[None, :]

        mask = (energy_axis_2d > energies_2d - width) & (energy_axis_2d < energies_2d + width)

        N, H, W = self.array.shape
        K = mask.shape[1]
        eds2 = self.array.reshape(N, -1)
        w = mask.astype(self.array.dtype)

        maps = (w.T @ eds2).reshape(K, H, W)

        existing = getattr(self, "_spectrum_images", {})
        self._spectrum_images = {**existing, **dict(zip(labels, maps))}

        if show:
            self.show_spectrum_images(x_ray_lines=elements)

        if return_maps:
            return maps, labels

    def Integrate(self, spec, width=0.15, return_maps=False, show=True, **kwargs):
        """Integrate selected X-ray lines and return combined map(s).

        This method builds per-selector energy masks (union of line windows)
        and integrates the dataset over those masks. For single-selector
        plotting it reuses ``show_energy_window_map`` with the computed mask.

        Parameters
        ----------
        spec : str | list[str] | tuple[str, ...] | set[str]
            Selector(s) like ``"Au"``, ``"AuK"``, ``"AuKa1"``.
            Element selectors (e.g. ``"Au"``) integrate all available lines.
        width : float, optional
            Half-width in keV for line-window integration, by default 0.15.
        return_maps : bool, optional
            If True, always return ``dict[selector, map]``.
            If False and a single selector is provided, return only that 2D map.
        show : bool, optional
            If True, display the integrated map(s).
        """
        try:
            width = float(width)
        except (TypeError, ValueError):
            raise ValueError("width must be a positive finite number")
        if not np.isfinite(width) or width <= 0:
            raise ValueError("width must be a positive finite number")

        specs = type(self)._normalize_specs(spec, param_name="spec")
        arr = np.asarray(self.array, dtype=float)
        energy_axis = np.asarray(self.energy_axis, dtype=float)
        energy_min = float(np.min(energy_axis))
        energy_max = float(np.max(energy_axis))

        integrated_maps = {}
        selector_masks = {}
        for raw in specs:
            selector = str(raw).strip()
            line_energies, _, _ = self.x_ray_lookup(selector)
            in_range = np.logical_and(line_energies >= energy_min, line_energies <= energy_max)
            line_energies = np.asarray(line_energies[in_range], dtype=float)
            if line_energies.size == 0:
                raise ValueError(
                    f"No X-ray lines for selector '{selector}' are within the dataset energy range"
                )

            selector_mask = np.zeros(energy_axis.shape, dtype=bool)
            for line_energy in line_energies:
                selector_mask |= np.logical_and(
                    energy_axis >= (float(line_energy) - width),
                    energy_axis <= (float(line_energy) + width),
                )

            if not np.any(selector_mask):
                raise ValueError(
                    f"No energy channels selected for selector '{selector}'. Try increasing width."
                )

            selector_masks[selector] = selector_mask
            integrated_maps[selector] = arr[selector_mask, :, :].sum(axis=0)

        if show:
            if "roi_px" in kwargs or "roi_cal" in kwargs:
                raise ValueError(
                    "Use roi (pixel) or roi_units (calibrated). roi_px/roi_cal are not supported"
                )
            if len(integrated_maps) == 1:
                selector = next(iter(integrated_maps.keys()))
                self.show_energy_window_map(
                    energy_window=[energy_min, energy_max],
                    roi=kwargs.pop("roi", None),
                    roi_units=kwargs.pop("roi_units", None),
                    mask=selector_masks[selector],
                    data_type=kwargs.pop("data_type", "eds"),
                    cmap=kwargs.pop("cmap", "magma"),
                    show=True,
                )
            else:
                show_2d(
                    list(integrated_maps.values()),
                    title=list(integrated_maps.keys()),
                    cmap=kwargs.pop("cmap", "magma"),
                    scalebar={"sampling": self.sampling[1], "units": self.units[1]},
                    **kwargs,
                )

        if return_maps or len(integrated_maps) != 1:
            return integrated_maps
        return next(iter(integrated_maps.values()))

    def integrate(self, spec, width=0.15, return_maps=False, show=True, **kwargs):
        """Lowercase alias for :meth:`Integrate`."""
        return self.Integrate(
            spec=spec,
            width=width,
            return_maps=return_maps,
            show=show,
            **kwargs,
        )

    def show_spectrum_images(
        self, x_ray_lines=None, return_fig=False, method="integration", **kwargs
    ):
        """Plot cached spectrum-image maps."""
        spectrum_images = (
            getattr(self, "_spectrum_images", None)
            if method == "integration"
            else getattr(self, "_spectrum_images_pytorch", None)
            if method == "fit"
            else None
        )
        if spectrum_images is None:
            raise ValueError(
                f"Method {method!r} is not supported, please choose 'integration' or 'fit'"
            )
        if not isinstance(spectrum_images, dict) or not spectrum_images:
            raise ValueError("No spectrum images found. Run generage_spectrum_images(...) first.")

        line_map = {str(k): np.asarray(v) for k, v in spectrum_images.items()}
        labels = list(line_map)
        labels_by_element = type(self)._group_labels_by_element(labels)

        def _sum_maps(lbls):
            return np.sum(np.stack([line_map[lbl] for lbl in lbls], axis=0), axis=0)

        specs = type(self)._normalize_specs(x_ray_lines, param_name="x_ray_lines", allow_none=True)
        images, titles = [], []

        if not specs:
            for element in sorted(labels_by_element):
                lbls = labels_by_element[element]
                if lbls:
                    images.append(_sum_maps(lbls))
                    titles.append(element)
        else:
            for raw in specs:
                selected = type(self)._select_labels(
                    str(raw), labels=labels, labels_by_element=labels_by_element
                )
                if not selected:
                    raise ValueError(
                        f"No spectrum images matched selector '{raw}'. "
                        f"Available examples: {', '.join(sorted(labels)[:10])}"
                    )
                images.append(line_map[selected[0]] if len(selected) == 1 else _sum_maps(selected))
                titles.append(selected[0] if len(selected) == 1 else str(raw).strip())

        if not images:
            raise ValueError("No spectrum images selected for plotting")

        cmap = kwargs.pop("cmap", "magma")
        fig, ax = show_2d(
            images,
            title=titles,
            cmap=cmap,
            scalebar={"sampling": self.sampling[1], "units": self.units[1]},
            returnfig=True,
            **kwargs,
        )
        if return_fig:
            return fig, ax

    def _build_pytorch_spectrum_images(
        self, abundance_maps: np.ndarray, element_names: list[str] | tuple[str, ...]
    ) -> dict[str, np.ndarray]:
        """Build per-line maps from fitted per-element abundance maps."""
        maps = np.asarray(abundance_maps)
        if maps.ndim != 3:
            return {}

        line_maps = {}
        for element_index, element_name in enumerate(element_names):
            if element_index >= maps.shape[0]:
                break

            element_map = np.asarray(maps[element_index], dtype=float)
            try:
                _, line_weights, line_labels = self.x_ray_lookup(str(element_name))
            except ValueError:
                continue

            for weight, label in zip(line_weights, line_labels):
                try:
                    weight_value = float(weight)
                except (TypeError, ValueError):
                    continue
                line_maps[str(label)] = element_map * weight_value

        return line_maps

    def quantify_composition_cliff_lorimer(
        self,
        k_factors,
        method="integration",
        return_maps=False,
        verbose=True,
    ):
        """Quantify composition from cached spectrum maps using Cliff-Lorimer.

        Parameters
        ----------
        k_factors : dict
            Mapping of selector -> k-factor.
        method : {"integration", "fit"}, optional
            Source map set to use.
        return_maps : bool, optional
            If True, include per-element/per-selector map outputs.
        verbose : bool, optional
            If True, print a small scalar text table (no maps).
        """
        if not isinstance(k_factors, dict) or not k_factors:
            raise ValueError("k_factors must be a non-empty dict")

        spectrum_images = (
            getattr(self, "_spectrum_images", None)
            if method == "integration"
            else getattr(self, "_spectrum_images_pytorch", None)
            if method == "fit"
            else None
        )
        if spectrum_images is None:
            raise ValueError(
                f"Method {method!r} is not supported, please choose 'integration' or 'fit'"
            )
        if not isinstance(spectrum_images, dict) or not spectrum_images:
            raise ValueError("No spectrum images available for quantification")

        type(self)._ensure_element_info()
        ordered_elements = type(self)._ordered_element_keys(type(self).element_info or {})

        line_map = {str(k): np.asarray(v, dtype=float) for k, v in spectrum_images.items()}
        labels = list(line_map)
        labels_by_element = type(self)._group_labels_by_element(labels)

        def _match(selector: str) -> list[str]:
            return type(self)._select_labels(
                selector, labels=labels, labels_by_element=labels_by_element
            )

        intensities: dict[str, float] = {}
        weighted_intensities: dict[str, float] = {}
        selector_maps = {} if return_maps else None
        intensity_maps = {} if return_maps else None
        weighted_intensity_maps = {} if return_maps else None

        for selector, k_raw in k_factors.items():
            try:
                k_val = float(k_raw)
            except (TypeError, ValueError):
                raise ValueError(f"k_factors[{selector!r}] must be numeric")
            if not np.isfinite(k_val) or k_val <= 0:
                raise ValueError(f"k_factors[{selector!r}] must be a positive finite number")

            sel_labels = _match(str(selector).strip())
            if not sel_labels:
                raise ValueError(
                    f"No spectrum images matched selector {selector!r}. "
                    f"Available examples: {', '.join(sorted(labels)[:10])}"
                )

            matched_elements = {
                type(self)._resolve_element_from_label(lbl, ordered_elements) for lbl in sel_labels
            }
            matched_elements = {e for e in matched_elements if e is not None}
            if len(matched_elements) != 1:
                raise ValueError(
                    f"Selector {selector!r} matched multiple elements: {sorted(matched_elements)}. "
                    "Use selectors like 'AuK' or 'AuKa1'."
                )
            element = next(iter(matched_elements))

            grouped_map = np.sum(np.stack([line_map[lbl] for lbl in sel_labels], axis=0), axis=0)
            intensity = float(np.sum(grouped_map))
            weighted = float(k_val * intensity)

            intensities[element] = float(intensities.get(element, 0.0)) + intensity
            weighted_intensities[element] = (
                float(weighted_intensities.get(element, 0.0)) + weighted
            )

            if return_maps:
                weighted_map = grouped_map * k_val
                selector_maps[str(selector)] = grouped_map
                if element in intensity_maps:
                    intensity_maps[element] = intensity_maps[element] + grouped_map
                    weighted_intensity_maps[element] = (
                        weighted_intensity_maps[element] + weighted_map
                    )
                else:
                    intensity_maps[element] = grouped_map.copy()
                    weighted_intensity_maps[element] = weighted_map.copy()

        if len(weighted_intensities) < 2:
            raise ValueError("At least two elements are required for Cliff-Lorimer quantification")

        weighted_sum = float(sum(weighted_intensities.values()))
        atomic_percent = {
            el: (100.0 * val / weighted_sum if weighted_sum > 0 else 0.0)
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
            el: (((atomic_percent[el] / 100.0) * float(atomic_weights[el]) / weight_sum) * 100.0)
            if weight_sum > 0
            else 0.0
            for el in atomic_percent
        }

        result = {
            "intensities": intensities,
            "weighted_intensities": weighted_intensities,
            "atomic_percent": atomic_percent,
            "weight_percent": weight_percent,
        }

        ordered_elements = sorted(
            weighted_intensities.keys(),
            key=lambda element_name: weighted_intensities[element_name],
            reverse=True,
        )
        table_lines = [
            "Element  Intensity      Weighted Intensity    Atomic %    Weight %",
            "-------  -------------  --------------------  ----------  ----------",
        ]
        for element_name in ordered_elements:
            table_lines.append(
                f"{element_name:<7}  "
                f"{intensities[element_name]:>13.3f}  "
                f"{weighted_intensities[element_name]:>20.3f}  "
                f"{atomic_percent[element_name]:>10.3f}  "
                f"{weight_percent[element_name]:>10.3f}"
            )
        table_text = "\n".join(table_lines)
        result["summary_table"] = table_text
        if verbose:
            print(table_text)

        if return_maps:
            weighted_stack = np.stack(list(weighted_intensity_maps.values()), axis=0)
            weighted_sum_map = np.sum(weighted_stack, axis=0)
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
                el: (atomic_percent_maps[el] / 100.0) * float(atomic_weights[el])
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
        if self._spectrum_images is not None:
            self._spectrum_images = {}

    def clear_spectrum_images_pytorch(self):
        if self._spectrum_images_pytorch is not None:
            self._spectrum_images = {}

    def peak_autoid(
        self,
        roi=None,
        roi_units=None,
        energy_range=None,
        elements=None,
        refline=None,
        ignore_elements=None,
        ignore_range=None,
        threshold=5.0,
        tolerance=0.15,
        min_line_weight=0.0,
        mask=None,
        show_text=True,
        snr_min=None,
        snr_threshold=None,
        distance_threshold_for_sample=0.05,
        grid_peaks=None,
        peaks=15,
        mode=None,
        return_details=False,
    ):
        """Auto-detect and label EDS peaks from the mean ROI spectrum.

        This method calls ``show_mean_spectrum`` to generate the baseline plot, then
        overlays peak markers and element labels from X-ray line matching.

        Parameters
        ----------
        min_line_weight : float, optional
            Minimum X-ray line weight required for a line to be eligible during
            matching and ranking. Set to 0 to allow all lines.
                mode : str | None, optional
                        Controls how element constraints are applied:
                        - "autofill": auto-ID over all elements while optionally overlaying
                            requested/saved-model element references.
                        - "elements_preferred": keep autofill enabled, but bias final
                            matching toward requested/saved-model elements.
                        - "elements_only": if elements are provided explicitly or via saved
                            model, only those elements are considered during matching.
                        - None: choose context-aware default. If no requested/saved-model
                            elements are present, defaults to "autofill". If requested or
                            saved-model elements exist, defaults to "elements_only".
        return_details : bool, optional
            If True, return full internal results (matches/confidence/peaks).
            If False (default), return only figure and axes.
        """
        type(self)._ensure_element_info()

        if grid_peaks is None:
            grid_peaks = {}

        requested_edge_filters = type(self)._parse_element_selectors(
            elements, allow_none=True, param_name="elements"
        )
        saved_model_edge_filters = {
            str(k): (set(map(str, v.keys())) if isinstance(v, dict) and v else None)
            for k, v in (getattr(self, "model_elements", {}) or {}).items()
        } or None

        if saved_model_edge_filters is not None and requested_edge_filters is not None:
            merged_edge_filters = dict(saved_model_edge_filters)
            for element_name, selectors in requested_edge_filters.items():
                if element_name not in merged_edge_filters:
                    merged_edge_filters[element_name] = selectors
                    continue

                existing = merged_edge_filters[element_name]
                if existing is None or selectors is None:
                    merged_edge_filters[element_name] = None
                else:
                    merged_edge_filters[element_name] = set(existing).union(set(selectors))
            requested_edge_filters = merged_edge_filters
        elif requested_edge_filters is None and saved_model_edge_filters is not None:
            requested_edge_filters = saved_model_edge_filters

        if requested_edge_filters is not None:
            elements = list(requested_edge_filters.keys())

        requested_elements = set(elements) if elements is not None else None

        mode_normalized = None if mode is None else str(mode).strip().lower()
        valid_modes = {"autofill", "elements_only", "elements_preferred"}
        if mode_normalized is not None and mode_normalized not in valid_modes:
            raise ValueError("mode must be one of: autofill, elements_only, elements_preferred")

        if mode_normalized is None:
            mode_normalized = "elements_only" if requested_elements is not None else "autofill"

        if mode_normalized == "elements_only" and requested_elements is None:
            raise ValueError("mode='elements_only' requires elements to be specified or saved")
        if mode_normalized == "elements_preferred" and requested_elements is None:
            raise ValueError(
                "mode='elements_preferred' requires elements to be specified or saved"
            )

        match_elements = requested_elements if mode_normalized == "elements_only" else None
        preferred_elements_set = (
            set(str(element_name) for element_name in requested_elements)
            if mode_normalized == "elements_preferred" and requested_elements is not None
            else set()
        )
        reference_elements = (
            requested_elements
            if mode_normalized in {"elements_only", "autofill", "elements_preferred"}
            and requested_elements is not None
            else None
        )

        if isinstance(ignore_elements, str):
            ignore_elements = [ignore_elements]
        if ignore_elements is not None and not isinstance(ignore_elements, (list, tuple, set)):
            raise TypeError("ignore_elements must be None, a string, or a sequence of strings")
        try:
            min_line_weight = float(min_line_weight)
        except (TypeError, ValueError) as exc:
            raise TypeError("min_line_weight must be a number") from exc
        if min_line_weight < 0:
            raise ValueError("min_line_weight must be >= 0")
        ignored_elements = (
            {str(element_name) for element_name in ignore_elements}
            if ignore_elements is not None
            else set()
        )

        fig, (ax_img, ax_spec) = self.show_mean_spectrum(
            roi=roi,
            roi_units=roi_units,
            energy_range=energy_range,
            mask=mask,
            data_type="eds",
            show=False,
        )

        spec = self.calculate_mean_spectrum(
            roi=roi,
            roi_units=roi_units,
            energy_range=energy_range,
            ignore_range=ignore_range,
            mask=mask,
        )
        dE = float(self.sampling[0])
        E0 = float(self.origin[0]) if hasattr(self, "origin") else 0.0
        E = E0 + dE * np.arange(self.shape[0])

        if energy_range is not None:
            indices = np.where((energy_range[0] <= E) & (energy_range[1] >= E))[0]
            E = E[indices]

        if ignore_range is None:
            ignore_range = [0, 0.25]

        def _is_in_ignored_range(energy_value: float) -> bool:
            if ignore_range is None or len(ignore_range) != 2:
                return False
            min_ignore, max_ignore = ignore_range
            return float(min_ignore) <= float(energy_value) <= float(max_ignore)

        peak_indices, peak_properties = find_peaks(spec, height=0, distance=5)
        peak_heights = peak_properties["peak_heights"]

        background_std = np.nanstd(spec[spec <= np.nanpercentile(spec, 50)])
        if not np.isfinite(background_std) or background_std <= 0:
            background_std = np.nanstd(spec)
        if not np.isfinite(background_std) or background_std <= 0:
            background_std = 1.0

        initial_snrs = []
        for _, height in zip(peak_indices, peak_heights):
            initial_snrs.append(height / background_std)

        if len(initial_snrs) > 0:
            snr_values = np.asarray(initial_snrs, dtype=float)
            snr_values = snr_values[np.isfinite(snr_values)]
        else:
            snr_values = np.asarray([], dtype=float)

        if snr_min is None:
            if snr_values.size > 0:
                sorted_snrs = np.sort(snr_values)
                target_rank = int(np.clip(2 * int(peaks), 12, 64))
                target_rank = min(target_rank, int(sorted_snrs.size))
                rank_cutoff = float(sorted_snrs[-target_rank])
                q30 = float(np.percentile(sorted_snrs, 30))
                q40 = float(np.percentile(sorted_snrs, 40))
                q50 = float(np.percentile(sorted_snrs, 50))
                adaptive_cutoff = min(q50, max(q30, 0.35 * rank_cutoff, 0.9 * q40))
                # Keep the display threshold permissive so moderate-SNR peaks remain visible.
                min_snr = float(np.clip(adaptive_cutoff, 7.0, 14.0))
            else:
                min_snr = 8.0
        else:
            min_snr = float(snr_min)

        if snr_threshold is None:
            if snr_values.size > 0:
                high_snr_pool = snr_values[snr_values >= min_snr]
                if high_snr_pool.size == 0:
                    high_snr_pool = snr_values
                sorted_high = np.sort(high_snr_pool)[::-1]
                anchor_count = int(np.clip(int(peaks), 10, 40))
                anchor_count = min(anchor_count, int(sorted_high.size))
                anchor_pool = sorted_high[:anchor_count]
                anchor_median = float(np.percentile(anchor_pool, 50))
                anchor_q75 = float(np.percentile(anchor_pool, 75))
                anchor_q90 = float(np.percentile(anchor_pool, 90))
                adaptive_threshold = max(anchor_median, 0.7 * anchor_q75, 2.5 * min_snr)
                # Anchor to the strongest displayed-peak regime so defaults do not
                # collapse toward the low-SNR bulk when peak counts are large.
                snr_threshold_for_sample = float(
                    np.clip(adaptive_threshold, max(2.5 * min_snr, min_snr), anchor_q90)
                )
            else:
                snr_threshold_for_sample = max(4.0 * min_snr, 30.0)
        else:
            snr_threshold_for_sample = float(snr_threshold)

        all_candidate_peaks = []
        significant_peaks = []
        for peak_idx, height in zip(peak_indices, peak_heights):
            peak_energy = E[peak_idx]

            if _is_in_ignored_range(peak_energy):
                continue

            snr = height / background_std
            all_candidate_peaks.append((peak_idx, height, peak_energy, snr))
            if snr >= min_snr:
                significant_peaks.append((peak_idx, height, peak_energy, snr))

        significant_peaks.sort(key=lambda item: item[3], reverse=True)

        display_peaks = significant_peaks[:peaks]

        all_info = type(self).element_info
        peak_matches = []

        def _line_shell(line_name):
            line_upper = str(line_name).upper()
            if line_upper.startswith("K"):
                return "K"
            if line_upper.startswith("L"):
                return "L"
            if line_upper.startswith("M"):
                return "M"
            return "?"

        def _shell_preference_factor(shell_name):
            # Keep K and L comparable; only downweight M lines because they
            # are more prone to overlap-driven false assignments.
            if shell_name in {"K", "L"}:
                return 1.0
            if shell_name == "M":
                return 0.72
            return 1.0

        def _peak_confidence(snr_value, line_weight, distance_value):
            return type(self)._peak_confidence(
                snr_value=snr_value,
                line_weight=line_weight,
                distance_value=distance_value,
                tolerance=tolerance,
            )

        def _line_matches_selector(line_name: str, selector: str) -> bool:
            line = str(line_name).strip().lower()
            token = str(selector).strip().lower()
            if token in {"k", "l", "m"}:
                return line.startswith(token)
            return token in line

        def _line_allowed_for_element(element_name, line_name, edge_filters=None):
            if edge_filters is None:
                return True

            selectors = edge_filters.get(str(element_name))
            if selectors is None:
                return True

            return any(_line_matches_selector(line_name, token) for token in selectors)

        def _best_line_match(peak_energy, peak_snr, allowed_elements=None, edge_filters=None):
            best_score = -float("inf")
            best_element = None
            best_line_name = None
            best_line_weight = 0.0
            best_distance = float("inf")

            if not all_info:
                return None

            for element_name, lines in all_info.items():
                if allowed_elements is not None and element_name not in allowed_elements:
                    continue

                for line_name, line_info in lines.items():
                    if not _line_allowed_for_element(element_name, line_name, edge_filters):
                        continue
                    line_energy = line_info["energy (keV)"]
                    line_weight = line_info.get("weight", 0.5)
                    distance = abs(peak_energy - line_energy)
                    shell = _line_shell(line_name)

                    is_m_line = "M" in line_name and not ("Ma" in line_name or "Mb" in line_name)
                    effective_tolerance = tolerance * 0.5 if is_m_line else tolerance

                    if line_weight >= min_line_weight and distance <= effective_tolerance:
                        # Use score-based ranking: confidence with weight and distance
                        # This matches rematcher logic and avoids distance-driven artifacts
                        score = _peak_confidence(peak_snr, line_weight, distance)
                        score *= _shell_preference_factor(shell)
                        if score > best_score:
                            best_score = score
                            best_element = element_name
                            best_line_name = line_name
                            best_line_weight = line_weight
                            best_distance = distance

            if best_element is None:
                return None

            return best_element, best_line_name, best_line_weight, best_distance

        search_elements = match_elements

        for peak_idx, height, peak_energy, snr in display_peaks:
            best_match_info = _best_line_match(
                peak_energy, snr, search_elements, requested_edge_filters
            )
            if best_match_info is not None:
                best_element, best_line_name, best_line_weight, best_distance = best_match_info
                best_element = str(best_element)
                best_line_name = str(best_line_name)
                best_match = f"{best_element} {best_line_name}"
                match_confidence = _peak_confidence(snr, best_line_weight, best_distance)
                peak_matches.append(
                    (
                        peak_idx,
                        height,
                        peak_energy,
                        snr,
                        best_element,
                        best_match,
                        best_distance,
                        best_line_name,
                        best_line_weight,
                        match_confidence,
                    )
                )

        detected_elements = set()
        detected_sample_peaks = {}
        element_confidence = {}
        element_stats = {}
        line_evidence = {}
        for (
            peak_idx,
            height,
            peak_energy,
            snr,
            element_name,
            match_str,
            distance,
            line_name,
            line_weight,
            match_confidence,
        ) in peak_matches:
            element_name = str(element_name)
            line_name = str(line_name)
            if search_elements is not None and element_name not in search_elements:
                continue

            shell = _line_shell(line_name)
            line_label = f"{element_name} {line_name}"
            if element_name not in element_stats:
                element_stats[element_name] = {
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
                }

            if line_label not in line_evidence:
                line_evidence[line_label] = {
                    "match_count": 0,
                    "strong_matches": 0,
                    "best_conf": 0.0,
                    "best_snr": 0.0,
                    "energies": [],
                }

            element_stats[element_name]["raw_conf"] += float(match_confidence)
            element_stats[element_name]["shells"].add(shell)
            element_stats[element_name]["lines"].add(str(line_name))
            element_stats[element_name]["match_count"] += 1
            if snr > snr_threshold_for_sample and distance < distance_threshold_for_sample:
                element_stats[element_name]["strong_matches"] += 1

            if float(match_confidence) > float(element_stats[element_name]["best_match_conf"]):
                element_stats[element_name]["best_match_conf"] = float(match_confidence)
                element_stats[element_name]["best_match_snr"] = float(snr)
                element_stats[element_name]["best_match_energy"] = float(peak_energy)
                element_stats[element_name]["best_match_distance"] = float(distance)
                element_stats[element_name]["best_match_weight"] = float(line_weight)
                element_stats[element_name]["best_match_shell"] = shell

            line_evidence[line_label]["match_count"] += 1
            line_evidence[line_label]["energies"].append(float(peak_energy))
            if snr > snr_threshold_for_sample and distance < distance_threshold_for_sample:
                line_evidence[line_label]["strong_matches"] += 1
            if float(match_confidence) > float(line_evidence[line_label]["best_conf"]):
                line_evidence[line_label]["best_conf"] = float(match_confidence)
                line_evidence[line_label]["best_snr"] = float(snr)

        for element_name, stats in element_stats.items():
            valid_shells = {shell for shell in stats["shells"] if shell in {"K", "L", "M"}}
            num_shells = len(valid_shells)
            num_lines = len(stats["lines"])
            has_major_shell = len(valid_shells.intersection({"K", "L"})) > 0

            # Shell bonus: each additional confirmed shell is independent evidence.
            # Two independent shells give P(element|K∧L) ∝ P_K · P_L, so the
            # log-likelihood adds linearly and the likelihood ratio scales as
            # sqrt(n_shells) in the geometric-mean sense (Jaynes, Prob. Theory).
            shell_bonus = float(np.sqrt(max(1, num_shells)))

            # Line bonus: each additional distinct matched line reduces the
            # probability of a chance coincidence. Diminishing returns are
            # captured by log1p (each doubling of evidence adds a fixed increment).
            line_bonus = 1.0 + 0.30 * float(np.log1p(max(0, num_lines - 1)))

            # Strong-match bonus: the Poisson false-peak rate at SNR > T scales
            # as erfc(T/√2); n independent strong peaks compound multiplicatively
            # so log1p(n) correctly models the diminishing-returns likelihood ratio.
            strong_bonus = 1.0 + 0.40 * float(np.log1p(stats["strong_matches"]))

            # K/L shell presence is strong physical prior: these are the primary
            # transitions expected in any sample above Z≈10.
            major_bonus = 1.20 if has_major_shell else 1.0

            confidence = stats["raw_conf"] * shell_bonus * line_bonus * strong_bonus * major_bonus
            element_confidence[element_name] = float(confidence)

        if element_confidence:
            conf_values = np.array(list(element_confidence.values()), dtype=float)
            # Absolute Poisson MDL floor (Currie 1968, Anal. Chem. 40:586):
            # a peak is physically detectable only when SNR >= 3σ of background.
            # The minimum confidence a single real peak can produce is:
            #   log1p(3.0) * 0.5 * exp(-0.5*(1σ distance)²) ≈ 0.334
            # (half-weight line, 1-sigma spatial offset, SNR=3).
            # No element below this floor can have a statistically detectable peak.
            poisson_mdl_snr = 3.0
            _mdl_conf_floor = float(np.log1p(poisson_mdl_snr)) * 0.5 * float(np.exp(-0.5))
            confidence_cutoff = max(
                float(np.percentile(conf_values, 45)),
                0.30 * float(conf_values.max()),
                _mdl_conf_floor,
            )

            for element_name, confidence in element_confidence.items():
                stats = element_stats[element_name]
                valid_shells = {shell for shell in stats["shells"] if shell in {"K", "L", "M"}}
                has_major_shell = len(valid_shells.intersection({"K", "L"})) > 0
                # is_supported: best peak must clear the 3σ Poisson MDL
                # (Currie 1968) — any peak below SNR=3 cannot be distinguished
                # from background noise regardless of fit quality.
                is_supported = (
                    confidence >= confidence_cutoff
                    and stats["best_match_snr"] >= poisson_mdl_snr
                    and (
                        stats["strong_matches"] >= 1
                        or stats["best_match_snr"] >= max(min_snr, 0.6 * snr_threshold_for_sample)
                    )
                )
                # is_near_cutoff_but_consistent: two independent peaks above the
                # 3σ MDL have a joint false-positive probability ≈ p₁·p₂ ≈ negligible
                # (akin to requiring both the critical level Lc and detection limit Ld
                # to be satisfied on independent lines — Currie 1968, §IV).
                is_near_cutoff_but_consistent = (
                    confidence >= 0.75 * confidence_cutoff
                    and stats["best_match_snr"] >= poisson_mdl_snr
                    and stats["match_count"] >= 2
                    and has_major_shell
                    and stats["best_match_snr"] >= max(min_snr, 0.5 * snr_threshold_for_sample)
                )
                is_high_energy_singleton_anchor = (
                    stats["match_count"] == 1
                    and stats["best_match_energy"] >= 6.0
                    and stats["best_match_weight"] >= 0.8
                    and stats["best_match_distance"] <= 0.35 * tolerance
                    and confidence >= 0.45 * confidence_cutoff
                )
                is_high_quality_singleton = (
                    stats["match_count"] == 1
                    and has_major_shell
                    and stats["best_match_snr"] >= max(min_snr, 0.6 * snr_threshold_for_sample)
                    and stats["best_match_weight"] >= 0.5
                    and stats["best_match_distance"] <= 0.30 * tolerance
                    and confidence >= 0.35 * confidence_cutoff
                )

                if (
                    is_supported
                    or is_near_cutoff_but_consistent
                    or is_high_energy_singleton_anchor
                    or is_high_quality_singleton
                ):
                    detected_elements.add(element_name)

        refined_peak_matches = []
        raw_match_by_idx = {int(match[0]): match for match in peak_matches}
        anchor_elements = {
            element_name
            for element_name in detected_elements
            if element_name in element_stats
            and element_stats[element_name].get("best_match_energy", 0.0) >= 6.0
            and element_stats[element_name].get("best_match_weight", 0.0) >= 0.8
        }
        max_detected_conf = (
            max([element_confidence.get(el, 0.0) for el in detected_elements])
            if len(detected_elements) > 0
            else 0.0
        )
        dominant_elements = set()
        if element_confidence:
            conf_values = np.array(list(element_confidence.values()), dtype=float)
            conf_median = float(np.median(conf_values)) if len(conf_values) > 0 else 0.0
            conf_p80 = float(np.percentile(conf_values, 80)) if len(conf_values) > 1 else 0.0
            conf_floor = max(conf_median, 1e-9)

            for element_name, confidence in element_confidence.items():
                stats = element_stats.get(str(element_name), {})
                has_repeat_support = (
                    int(stats.get("match_count", 0)) >= 2
                    or int(stats.get("strong_matches", 0)) >= 1
                )
                if (
                    float(confidence) >= conf_p80
                    and float(confidence) >= 1.8 * conf_floor
                    and has_repeat_support
                ):
                    dominant_elements.add(str(element_name))

        def _element_prior_factor(element_name, denom):
            prior = float(element_confidence.get(element_name, 0.0)) / denom
            prior_factor = 1.0 + 0.5 * prior

            # When an element is already strongly supported by the spectrum,
            # bias nearby ambiguous peaks toward that same element.
            if prior >= 0.90:
                prior_factor *= 1.9
            elif prior >= 0.75:
                prior_factor *= 1.5
            elif prior >= 0.55:
                prior_factor *= 1.2

            return prior, prior_factor

        def _line_consistency_boost(element_name, line_name, peak_energy, denom):
            # Use dominant_elements membership rather than a normalized-prior
            # threshold.  The normalized prior can be suppressed when one or
            # two elements have very large confidence (e.g. Au driving the
            # denominator high), causing legitimate dominant elements like Cu
            # to silently fail the old `prior >= 0.75` gate even though they
            # clearly belong in the sample.
            if str(element_name) not in dominant_elements:
                return 1.0

            line_label = f"{element_name} {line_name}"
            evidence = line_evidence.get(line_label)
            if evidence is None:
                return 1.0

            # Require support from another peak for this exact line so we
            # don't self-reinforce a single spurious match.
            has_other_peak_support = any(
                abs(float(peak_energy) - float(prev_energy)) >= 0.04
                for prev_energy in evidence.get("energies", [])
            )
            if not has_other_peak_support:
                return 1.0

            best_line_conf = float(evidence.get("best_conf", 0.0))
            best_line_snr = float(evidence.get("best_snr", 0.0))
            strong_line_matches = int(evidence.get("strong_matches", 0))

            # Scale by line weight: high-weight primary lines (K/L α, weight
            # ≈ 0.7–1.0) get proportionally stronger boosts than low-weight
            # secondary lines (weight < 0.4).  This prevents a low-weight
            # line from an equally-dominant element from outscoring a
            # confirmed primary line purely via prior amplification.
            line_info_entry = (all_info or {}).get(str(element_name), {}).get(str(line_name), {})
            line_weight = (
                float(line_info_entry.get("weight", 0.5))
                if isinstance(line_info_entry, dict)
                else 0.5
            )
            weight_tier = 1.0 + 0.7 * max(0.0, line_weight - 0.35)

            if strong_line_matches >= 1 and best_line_conf >= 1.4:
                return min(3.2, 2.4 * weight_tier)
            if best_line_conf >= 1.1 and best_line_snr >= max(
                min_snr, 0.75 * snr_threshold_for_sample
            ):
                return min(2.6, 1.9 * weight_tier)
            if best_line_conf >= 0.8:
                return min(2.0, 1.5 * weight_tier)
            return min(1.5, 1.2 * weight_tier)

        def _dominant_element_boost(element_name, denom):
            element_key = str(element_name)
            if element_key not in dominant_elements:
                return 1.0

            prior, _ = _element_prior_factor(element_key, denom)
            stats = element_stats.get(element_key, {})
            repeat_support = max(
                int(stats.get("strong_matches", 0)),
                max(0, int(stats.get("match_count", 0)) - 1),
            )

            if prior >= 0.90:
                base_boost = 2.30
            elif prior >= 0.75:
                base_boost = 1.85
            else:
                base_boost = 1.45

            if repeat_support >= 2:
                base_boost *= 1.10

            return min(base_boost, 2.60)

        def _best_supported_line_match_with_prior(
            peak_energy, snr, allowed_elements, edge_filters=None
        ):
            if not all_info or not allowed_elements:
                return None

            best_tuple = None
            best_score = -float("inf")
            best_preferred_tuple = None
            best_preferred_score = -float("inf")
            denom = max(float(max_detected_conf), 1e-9)

            for element_name, lines in all_info.items():
                if element_name not in allowed_elements:
                    continue

                prior, prior_factor = _element_prior_factor(element_name, denom)
                preferred_factor = 1.35 if str(element_name) in preferred_elements_set else 1.0

                for line_name, line_info in lines.items():
                    if not _line_allowed_for_element(element_name, line_name, edge_filters):
                        continue
                    line_energy = line_info["energy (keV)"]
                    line_weight = line_info.get("weight", 0.5)
                    distance = abs(peak_energy - line_energy)
                    shell = _line_shell(line_name)

                    is_m_line = "M" in line_name and not ("Ma" in line_name or "Mb" in line_name)
                    effective_tolerance = tolerance * 0.5 if is_m_line else tolerance

                    if line_weight < min_line_weight or distance > effective_tolerance:
                        continue

                    local_conf = _peak_confidence(snr, line_weight, distance)
                    anchor_boost = 1.0
                    if element_name in anchor_elements and shell in {"K", "L"}:
                        anchor_boost = 1.15

                    consistency_boost = _line_consistency_boost(
                        element_name, line_name, peak_energy, denom
                    )
                    dominant_boost = _dominant_element_boost(element_name, denom)

                    # M-lines are secondary transitions confirmed by K/L lines
                    # from the same element.  The large cascade boosts designed
                    # for K/L cross-peak propagation must not override a
                    # better-fitting primary K/L line from another element
                    # (e.g. P Ka1 vs Pt Ma1 near 2.03 keV: P Ka1 has higher
                    # local_conf but Pt is dominant via La1 at 9.47 keV).
                    if shell == "M":
                        eff_prior_factor = 1.0 + 0.3 * prior
                        eff_consistency = 1.0
                        eff_dominant = min(dominant_boost, 1.30)
                    else:
                        eff_prior_factor = prior_factor
                        eff_consistency = consistency_boost
                        eff_dominant = dominant_boost

                    score = (
                        local_conf
                        * eff_prior_factor
                        * anchor_boost
                        * preferred_factor
                        * eff_consistency
                        * eff_dominant
                    )
                    score *= _shell_preference_factor(shell)

                    if (
                        str(element_name) in preferred_elements_set
                        and score > best_preferred_score
                    ):
                        best_preferred_score = score
                        best_preferred_tuple = (element_name, line_name, line_weight, distance)

                    if score > best_score:
                        best_score = score
                        best_tuple = (element_name, line_name, line_weight, distance)

            if mode_normalized == "elements_preferred" and best_preferred_tuple is not None:
                return best_preferred_tuple

            return best_tuple

        def _top_supported_line_matches_with_prior(
            peak_energy, snr, allowed_elements, edge_filters=None, top_k=3
        ):
            if not all_info or top_k <= 0:
                return []

            scored_matches = []
            denom = max(float(max_detected_conf), 1e-9)

            for element_name, lines in all_info.items():
                if allowed_elements is not None and element_name not in allowed_elements:
                    continue

                prior, prior_factor = _element_prior_factor(element_name, denom)
                preferred_factor = 1.35 if str(element_name) in preferred_elements_set else 1.0

                for line_name, line_info in lines.items():
                    if not _line_allowed_for_element(element_name, line_name, edge_filters):
                        continue
                    line_energy = line_info["energy (keV)"]
                    line_weight = line_info.get("weight", 0.5)
                    distance = abs(peak_energy - line_energy)
                    shell = _line_shell(line_name)

                    is_m_line = "M" in line_name and not ("Ma" in line_name or "Mb" in line_name)
                    effective_tolerance = tolerance * 0.5 if is_m_line else tolerance

                    if line_weight < min_line_weight or distance > effective_tolerance:
                        continue

                    local_conf = _peak_confidence(snr, line_weight, distance)
                    anchor_boost = 1.0
                    if element_name in anchor_elements and shell in {"K", "L"}:
                        anchor_boost = 1.15

                    consistency_boost = _line_consistency_boost(
                        element_name, line_name, peak_energy, denom
                    )
                    dominant_boost = _dominant_element_boost(element_name, denom)

                    if shell == "M":
                        eff_prior_factor = 1.0 + 0.3 * prior
                        eff_consistency = 1.0
                        eff_dominant = min(dominant_boost, 1.30)
                    else:
                        eff_prior_factor = prior_factor
                        eff_consistency = consistency_boost
                        eff_dominant = dominant_boost

                    score = (
                        local_conf
                        * eff_prior_factor
                        * anchor_boost
                        * preferred_factor
                        * eff_consistency
                        * eff_dominant
                    )
                    score *= _shell_preference_factor(shell)
                    scored_matches.append(
                        (
                            float(score),
                            str(element_name),
                            str(line_name),
                            float(line_weight),
                            float(distance),
                        )
                    )

            scored_matches.sort(key=lambda item: item[0], reverse=True)

            if mode_normalized == "elements_preferred" and preferred_elements_set:
                preferred_scored_matches = [
                    item for item in scored_matches if str(item[1]) in preferred_elements_set
                ]
                if len(preferred_scored_matches) > 0:
                    non_preferred_scored_matches = [
                        item
                        for item in scored_matches
                        if str(item[1]) not in preferred_elements_set
                    ]
                    scored_matches = preferred_scored_matches + non_preferred_scored_matches

            unique = []
            seen_labels = set()
            for score, element_name, line_name, line_weight, distance in scored_matches:
                label = f"{element_name} {line_name}"
                if label in seen_labels:
                    continue
                seen_labels.add(label)
                unique.append((element_name, line_name, line_weight, distance, score))

            if len(unique) == 0:
                return []

            # Keep the top-scoring best match first, then prefer alternatives
            # from different elements before falling back to same-element lines.
            best_match = unique[0]
            selected = [best_match]
            selected_labels = {f"{best_match[0]} {best_match[1]}"}
            used_elements = {str(best_match[0])}

            for candidate in unique[1:]:
                element_name, line_name, _, _, _ = candidate
                label = f"{element_name} {line_name}"
                if label in selected_labels or str(element_name) in used_elements:
                    continue
                selected.append(candidate)
                selected_labels.add(label)
                used_elements.add(str(element_name))
                if len(selected) >= int(top_k):
                    return selected

            for candidate in unique[1:]:
                element_name, line_name, _, _, _ = candidate
                label = f"{element_name} {line_name}"
                if label in selected_labels:
                    continue
                selected.append(candidate)
                selected_labels.add(label)
                if len(selected) >= int(top_k):
                    break

            return selected

        first_pass_elements = {
            str(match[4])
            for match in peak_matches
            if len(match) > 4 and str(match[4]) not in ignored_elements
        }
        rematch_allowed_elements = set(str(element_name) for element_name in detected_elements)
        rematch_allowed_elements.update(first_pass_elements)
        if preferred_elements_set:
            rematch_allowed_elements.update(preferred_elements_set)

        for peak_idx, height, peak_energy, snr in display_peaks:
            match = raw_match_by_idx.get(int(peak_idx))

            if rematch_allowed_elements:
                alt_match_info = _best_supported_line_match_with_prior(
                    peak_energy, snr, rematch_allowed_elements, requested_edge_filters
                )
                if alt_match_info is not None:
                    alt_element, alt_line_name, alt_line_weight, alt_distance = alt_match_info
                    alt_element = str(alt_element)
                    alt_line_name = str(alt_line_name)
                    alt_match_str = f"{alt_element} {alt_line_name}"
                    alt_conf = _peak_confidence(snr, alt_line_weight, alt_distance)
                    match = (
                        peak_idx,
                        height,
                        peak_energy,
                        snr,
                        alt_element,
                        alt_match_str,
                        alt_distance,
                        alt_line_name,
                        alt_line_weight,
                        alt_conf,
                    )

            if match is not None:
                refined_peak_matches.append(match)

        peak_matches = refined_peak_matches

        # Keep confidence-based detected elements, but ensure they still have
        # at least one matched peak after rematching.
        matched_elements = {str(match[4]) for match in peak_matches}
        detected_elements = {
            str(element_name)
            for element_name in detected_elements
            if str(element_name) in matched_elements
        }
        if mode_normalized == "elements_preferred" and preferred_elements_set:
            detected_elements.update(
                str(element_name)
                for element_name in preferred_elements_set
                if str(element_name) in matched_elements
            )
        if ignored_elements:
            detected_elements = {
                str(element_name)
                for element_name in detected_elements
                if str(element_name) not in ignored_elements
            }

        refined_match_by_idx = {int(match[0]): match for match in peak_matches}

        final_matches_by_element: dict[str, set[str]] = {}
        for (
            _peak_idx,
            _height,
            _peak_energy,
            _snr,
            element_name,
            _match_str,
            _distance,
            line_name,
            _line_weight,
            _match_confidence,
        ) in peak_matches:
            element_key = str(element_name)
            if element_key in ignored_elements:
                continue
            final_matches_by_element.setdefault(element_key, set()).add(str(line_name))

        displayed_peak_count = len(display_peaks)
        total_over_snr_peak_count = len(significant_peaks)

        candidate_elements = sorted(
            str(element_name)
            for element_name in final_matches_by_element
            if str(element_name) not in detected_elements
        )
        possible_elements_set = set(candidate_elements)
        possible_line_labels = {
            f"{str(element_name)} {str(line_name)}"
            for (
                _,
                _,
                _,
                _,
                element_name,
                _,
                _,
                line_name,
                _,
                _,
            ) in peak_matches
            if str(element_name) in possible_elements_set
        }

        def _format_saved_model_elements(edge_filters):
            if edge_filters is None:
                return "None"

            formatted = []
            for element_name in sorted(str(name) for name in edge_filters):
                selectors = edge_filters.get(element_name)
                if selectors is None or len(selectors) == 0:
                    formatted.append(f"{element_name} [all]")
                    continue

                selector_names = sorted(str(token) for token in selectors)
                formatted.append(f"{element_name} [{', '.join(selector_names)}]")

            return "\n".join(formatted) if len(formatted) > 0 else "None"

        def _format_elements_with_lines(element_names):
            formatted = []
            for element_name in sorted(str(name) for name in element_names):
                line_names = sorted(
                    str(line_name)
                    for line_name in final_matches_by_element.get(str(element_name), set())
                )
                confidence_value = float(element_confidence.get(str(element_name), 0.0))
                confidence_suffix = f" conf={confidence_value:.2f}"
                if len(line_names) > 0:
                    formatted.append(
                        f"{element_name} [{', '.join(line_names)}]{confidence_suffix}"
                    )
                else:
                    formatted.append(f"{str(element_name)}{confidence_suffix}")
            return ", ".join(formatted)

        model_elements_header = "Saved Model Elements (Plotted):\n"
        print(
            f"\n{model_elements_header} {_format_saved_model_elements(saved_model_edge_filters)}"
        )

        if detected_elements:
            print(f"\nAutodetected: {_format_elements_with_lines(detected_elements)}")
        else:
            print("\nAutodetected: None")
        if dominant_elements:
            dominant_sorted = sorted(
                dominant_elements,
                key=lambda el: element_confidence.get(str(el), 0.0),
                reverse=True,
            )
            dominant_str = ", ".join(
                f"{el} (conf={element_confidence.get(str(el), 0.0):.2f})" for el in dominant_sorted
            )
            print(f"Dominant (strong prior): {dominant_str}")
        if candidate_elements:
            print(f"Possible: {_format_elements_with_lines(candidate_elements)}")
        else:
            print("Possible: None")

        # Stable per-element colors (same element color across K/L/M lines)
        elements_for_color = set(detected_elements)
        if search_elements is not None:
            elements_for_color.update(str(el) for el in search_elements)
        elements_for_color.update(str(match[4]) for match in peak_matches)

        sorted_elements_for_colors = sorted(elements_for_color)
        high_contrast_palette = [
            "#1f77b4",  # blue
            "#d62728",  # red
            "#2ca02c",  # green
            "#9467bd",  # purple
            "#ff7f0e",  # orange
            "#8c564b",  # brown
            "#e377c2",  # pink
            "#17becf",  # cyan
            "#bcbd22",  # olive
            "#7f7f7f",  # gray
            "#003f5c",  # dark blue
            "#7a5195",  # deep violet
            "#ef5675",  # strong rose
            "#ffa600",  # amber
            "#2f4b7c",  # slate blue
        ]
        color_palette = [
            high_contrast_palette[i % len(high_contrast_palette)]
            for i in range(max(1, len(sorted_elements_for_colors)))
        ]
        element_color_map = {
            element: color_palette[i] for i, element in enumerate(sorted_elements_for_colors)
        }

        table_rows = []
        matched_row_count = 0

        def _mark_autodetected_label(label_text):
            if not label_text or label_text == "-":
                return label_text
            element_token = str(label_text).split()[0]
            if element_token in detected_elements:
                return f"{label_text}*"
            return label_text

        def _format_label_with_score(label_text, score_value):
            if score_value is None:
                return str(label_text)
            return f"{label_text} ({float(score_value):.3f})"

        for peak_idx, height, peak_energy, snr in display_peaks:
            match = refined_match_by_idx.get(int(peak_idx))
            if match is not None:
                if search_elements is not None:
                    allowed_for_table = set(str(element_name) for element_name in search_elements)
                else:
                    allowed_for_table = {
                        str(element_name)
                        for element_name in (all_info or {}).keys()
                        if str(element_name) not in ignored_elements
                    }
                    if len(allowed_for_table) == 0:
                        allowed_for_table = None

                top_matches = _top_supported_line_matches_with_prior(
                    peak_energy,
                    snr,
                    allowed_for_table,
                    requested_edge_filters,
                    top_k=3,
                )
                best_label = f"{str(match[4])} {str(match[7])}"
                score_by_label = {
                    f"{element_name} {line_name}".lower(): float(score)
                    for element_name, line_name, _, _, score in top_matches
                }
                best_score = score_by_label.get(str(best_label).lower())
                best_match = _mark_autodetected_label(
                    _format_label_with_score(best_label, best_score)
                )
                alt_2 = "-"
                alt_3 = "-"

                if len(top_matches) > 0:
                    ranked_entries = [
                        (f"{element_name} {line_name}", float(score))
                        for element_name, line_name, _, _, score in top_matches
                    ]
                    ranked_entries = [
                        (label, score)
                        for label, score in ranked_entries
                        if str(label).lower() != str(best_label).lower()
                    ]
                    if len(ranked_entries) > 0:
                        alt_2 = _mark_autodetected_label(
                            _format_label_with_score(ranked_entries[0][0], ranked_entries[0][1])
                        )
                    if len(ranked_entries) > 1:
                        alt_3 = _mark_autodetected_label(
                            _format_label_with_score(ranked_entries[1][0], ranked_entries[1][1])
                        )

                table_rows.append((peak_energy, height, snr, best_match, alt_2, alt_3))
                matched_row_count += 1
            else:
                row_label = "Unmatched" if search_elements is not None else "Unknown"
                table_rows.append((peak_energy, height, snr, row_label, "-", "-"))

        sorted_table_rows = sorted(table_rows, key=lambda item: item[0])

        for (
            peak_idx,
            height,
            peak_energy,
            snr,
            element_name,
            match_str,
            distance,
            line_name,
            line_weight,
            match_confidence,
        ) in peak_matches:
            if element_name in detected_elements:
                detected_sample_peaks[peak_energy] = True

        filtered_sample_peaks = {}
        for peak_energy in detected_sample_peaks:
            for (
                peak_idx,
                height,
                matched_energy,
                snr,
                element_name,
                match_str,
                distance,
                line_name,
                line_weight,
                match_confidence,
            ) in peak_matches:
                if abs(matched_energy - peak_energy) < 0.001 and element_name in detected_elements:
                    filtered_sample_peaks[peak_energy] = True
                    break
        detected_sample_peaks = filtered_sample_peaks

        y_min = float(np.nanmin(spec)) if len(spec) > 0 else 0.0
        y_max = float(np.nanmax(spec)) if len(spec) > 0 else 1.0
        y_span = max(1e-9, y_max - y_min)
        y_scale = max(y_span, abs(y_max), 1.0)
        y_dot = -0.04 * y_scale

        def _infer_requested_element_for_color(peak_energy):
            if reference_elements is None or not all_info:
                return None

            best_element = None
            best_distance = float("inf")
            for element_name in reference_elements:
                element_key = str(element_name)
                lines_info = all_info.get(element_key, {})
                if not isinstance(lines_info, dict):
                    continue
                for line_name, line_info in lines_info.items():
                    if not _line_allowed_for_element(
                        element_key, line_name, requested_edge_filters
                    ):
                        continue
                    if not isinstance(line_info, dict):
                        continue
                    line_energy = line_info.get("energy (keV)")
                    if line_energy is None:
                        continue
                    try:
                        distance = abs(float(peak_energy) - float(line_energy))
                    except (TypeError, ValueError):
                        continue
                    if distance < best_distance:
                        best_distance = distance
                        best_element = element_key

            return best_element

        for peak_idx, height, peak_energy, snr in display_peaks:
            if _is_in_ignored_range(peak_energy):
                continue
            is_sample = detected_sample_peaks.get(peak_energy, False)
            match = refined_match_by_idx.get(int(peak_idx))
            is_possible = match is not None and str(match[4]) in possible_elements_set
            if match is not None:
                peak_element = match[4]
                line_color = element_color_map.get(peak_element, "red")
            else:
                inferred_element = _infer_requested_element_for_color(peak_energy)
                if inferred_element is not None:
                    line_color = element_color_map.get(str(inferred_element), "red")
                else:
                    line_color = "red"

            if is_sample:
                ax_spec.axvline(
                    peak_energy,
                    color=line_color,
                    linestyle="-",
                    alpha=0.5,
                    linewidth=1.5,
                )
            elif is_possible:
                ax_spec.axvline(
                    peak_energy,
                    color=line_color,
                    linestyle="--",
                    alpha=0.45,
                    linewidth=1.25,
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

            if show_text and is_sample:
                y_pos = height * 0.7
                is_grid_peak = False
                for grid_element, grid_energy in grid_peaks.items():
                    if abs(peak_energy - grid_energy) < 0.1:
                        ax_spec.text(
                            peak_energy,
                            y_pos,
                            f"{grid_element}\n(grid)",
                            ha="center",
                            va="bottom",
                            fontsize=8,
                            color="gray",
                            style="italic",
                        )
                        is_grid_peak = True
                        break
                if is_grid_peak:
                    print(f"Peak at {peak_energy} keV may come from the grid.")

        current_bottom, current_top = ax_spec.get_ylim()
        dot_padding = 0.02 * y_scale
        target_bottom = min(current_bottom, y_dot - dot_padding, -dot_padding)
        ax_spec.set_ylim(bottom=target_bottom, top=current_top)

        # If elements were explicitly requested, overlay reference X-ray lines from the
        # database even when they are not peak-matched by auto-id.
        all_label_candidates = []
        unified_axes_top_label_y = 0.92
        if reference_elements is not None:
            energy_min = float(np.min(E))
            energy_max = float(np.max(E))
            displayed_peak_energies = [
                float(peak_energy) for _, _, peak_energy, _ in display_peaks
            ]
            display_peak_tolerance = max(0.05, 0.5 * tolerance)
            existing_matches_by_element = {}
            for (
                peak_idx,
                height,
                peak_energy,
                snr,
                element_name,
                match_str,
                distance,
                line_name,
                line_weight,
                match_confidence,
            ) in peak_matches:
                element_key = str(element_name)
                if element_key not in existing_matches_by_element:
                    existing_matches_by_element[element_key] = []
                existing_matches_by_element[element_key].append(float(peak_energy))

            for element_name in sorted(reference_elements):
                element_key = str(element_name)
                lines_info = all_info.get(element_key, {}) if all_info is not None else {}
                if not isinstance(lines_info, dict) or len(lines_info) == 0:
                    continue

                candidate_lines = []
                for line_name, line_info in lines_info.items():
                    if not _line_allowed_for_element(
                        element_key, line_name, requested_edge_filters
                    ):
                        continue
                    if not isinstance(line_info, dict):
                        continue
                    energy_val = line_info.get("energy (keV)")
                    if energy_val is None:
                        continue
                    try:
                        line_energy = float(energy_val)
                    except (TypeError, ValueError):
                        continue
                    if not (energy_min <= line_energy <= energy_max):
                        continue

                    line_weight = float(line_info.get("weight", 0.0))
                    candidate_lines.append((line_name, line_energy, line_weight))

                if len(candidate_lines) == 0:
                    continue

                # Keep meaningful requested-element lines while avoiding excessive clutter.
                filtered_lines = [line for line in candidate_lines if line[2] >= 0.05]
                if len(filtered_lines) == 0:
                    filtered_lines = sorted(
                        candidate_lines, key=lambda item: item[2], reverse=True
                    )[:1]
                else:
                    filtered_lines = sorted(
                        filtered_lines, key=lambda item: item[2], reverse=True
                    )[:6]

                for line_name, line_energy, line_weight in filtered_lines:
                    if _is_in_ignored_range(line_energy):
                        continue

                    line_label = f"{element_key} {line_name}"
                    is_possible_line = line_label in possible_line_labels

                    # For possible elements, draw guides only near peaks that already
                    # passed snr_min. For reference-only requested elements, keep
                    # explicit refline guides even without nearby peaks.
                    if is_possible_line:
                        if not any(
                            abs(line_energy - peak_energy) <= display_peak_tolerance
                            for peak_energy in displayed_peak_energies
                        ):
                            continue

                    matched_energies = existing_matches_by_element.get(element_key, [])
                    if any(
                        abs(line_energy - matched_energy) <= max(0.05, 0.5 * tolerance)
                        for matched_energy in matched_energies
                    ):
                        continue

                    line_color = element_color_map.get(element_key, "black")
                    line_style = "--" if is_possible_line else ":"
                    line_alpha = 0.3 if is_possible_line else 0.35
                    ax_spec.axvline(
                        line_energy,
                        color=line_color,
                        linestyle=line_style,
                        alpha=line_alpha,
                        linewidth=1.2,
                    )
                    all_label_candidates.append(
                        (
                            float(line_energy),
                            f"{element_key} {line_name}",
                            line_color,
                            line_style,
                            float(unified_axes_top_label_y),
                            "axes_top",
                            8,
                            "normal",
                            0.8,
                        )
                    )

        legend_handles = []
        legend_labels = set()

        if show_text and peak_matches:
            labels_to_plot = []
            for (
                peak_idx,
                height,
                peak_energy,
                snr,
                element_name,
                match_str,
                distance,
                line_name,
                line_weight,
                match_confidence,
            ) in peak_matches:
                is_detected_peak = element_name in detected_elements and detected_sample_peaks.get(
                    peak_energy, False
                )
                is_possible_peak = str(element_name) in possible_elements_set
                if not (is_detected_peak or is_possible_peak):
                    continue

                line_name = match_str.split()[-1] if match_str else ""
                label_text = f"{element_name} {line_name}" if line_name else element_name
                if is_detected_peak:
                    label_text = f"{label_text}*"
                color = element_color_map.get(element_name, "black")
                linestyle = "-" if is_detected_peak else "--"
                labels_to_plot.append((peak_energy, label_text, color, height, linestyle))

            label_vertical_offset = max(0.03 * y_scale, 0.08)
            for peak_energy, label_text, color, height, linestyle in labels_to_plot:
                if _is_in_ignored_range(peak_energy):
                    continue
                if linestyle == "--":
                    all_label_candidates.append(
                        (
                            float(peak_energy),
                            label_text,
                            color,
                            linestyle,
                            float(unified_axes_top_label_y),
                            "axes_top",
                            9,
                            "normal",
                            0.9,
                        )
                    )
                else:
                    all_label_candidates.append(
                        (
                            float(peak_energy),
                            label_text,
                            color,
                            linestyle,
                            float(height + label_vertical_offset),
                            "data",
                            10,
                            "bold",
                            1.0,
                        )
                    )

        if show_text and all_label_candidates:
            all_label_candidates.sort(key=lambda item: item[0])

            overlap_threshold = max(0.16, 1.10 * float(tolerance))
            grouped_labels = []
            current_group = []

            for label in all_label_candidates:
                if not current_group:
                    current_group.append(label)
                    continue

                prev_energy = current_group[-1][0]
                energy_delta = abs(label[0] - prev_energy)
                should_group = energy_delta <= overlap_threshold

                if should_group:
                    current_group.append(label)
                else:
                    grouped_labels.append(current_group)
                    current_group = [label]

            if current_group:
                grouped_labels.append(current_group)

            for group in grouped_labels:
                if len(group) == 1:
                    (
                        peak_energy,
                        label_text,
                        color,
                        linestyle,
                        y_value,
                        y_mode,
                        font_size,
                        font_weight,
                        alpha_value,
                    ) = group[0]
                    if y_mode == "axes_top":
                        ax_spec.text(
                            peak_energy,
                            y_value,
                            label_text,
                            ha="center",
                            va="top",
                            fontsize=font_size,
                            color=color,
                            weight=font_weight,
                            rotation=90,
                            alpha=alpha_value,
                            transform=ax_spec.get_xaxis_transform(),
                            clip_on=True,
                        )
                    else:
                        ax_spec.text(
                            peak_energy,
                            y_value,
                            label_text,
                            ha="center",
                            va="bottom",
                            fontsize=font_size,
                            color=color,
                            weight=font_weight,
                            rotation=90,
                            alpha=alpha_value,
                        )
                else:
                    for _, label_text, color, linestyle, _, _, _, _, _ in group:
                        legend_key = (label_text, str(color), linestyle)
                        if legend_key in legend_labels:
                            continue
                        legend_labels.add(legend_key)
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

        overlap_legend = None
        if legend_handles:
            overlap_legend = ax_spec.legend(
                handles=legend_handles,
                loc="upper right",
                fontsize=8,
                title="Overlapping Labels",
            )
        if overlap_legend is not None:
            ax_spec.add_artist(overlap_legend)

        fig.tight_layout()
        plt.show()

        print(
            f"{'Energy (keV)':<12} {'Intensity':<12} {'SNR':<8} "
            f"{'Best Match':<22} {'Alt 2':<22} {'Alt 3':<22}"
        )
        print("-" * 105)
        for peak_energy, height, snr, best_match, alt_2, alt_3 in sorted_table_rows:
            print(
                f"{peak_energy:<12.3f} {height:<12.1f} {snr:<8.1f} "
                f"{best_match:<22} {alt_2:<22} {alt_3:<22}"
            )
        print("-" * 105)
        print(
            f"{displayed_peak_count} of {total_over_snr_peak_count} peaks above "
            f"snr_min={min_snr:.1f}, snr_threshold={snr_threshold_for_sample:.1f} displayed.\n"
        )

        if return_details:
            return {
                "figure": fig,
                "axes": (ax_img, ax_spec),
                "detected_elements": sorted(detected_elements),
                "element_confidence": element_confidence,
                "display_peaks": display_peaks,
                "peak_matches": peak_matches,
                "snr_min": min_snr,
                "snr_threshold": snr_threshold_for_sample,
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
            if self.model_elements is None:
                raise ValueError("elements_to_fit must be specified")
            elements_to_fit = list(self.model_elements.keys())
            print(f"using model_elements {elements_to_fit}")

        energy_axis_np = self.energy_axis.copy()
        energy_axis = torch.tensor(energy_axis_np, dtype=torch.float32, device=device)
        spectra = torch.tensor(self.array, dtype=torch.float32, device=device)

        if energy_range is not None:
            ind = (energy_axis >= energy_range[0]) & (energy_axis <= energy_range[1])
            energy_axis = energy_axis[ind]
            spectra = spectra[ind]
        else:
            energy_range = [float(energy_axis.min().item()), float(energy_axis.max().item())]

        print("fitting spectrum globally")
        spectrum_raw = spectra.sum((-1, -2))
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
            if self.model_elements is None:
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
            spectra = spectra[ind]
        else:
            energy_range = [float(energy_axis.min().item()), float(energy_axis.max().item())]

        if fit_mean_only:
            if verbose:
                print("fitting spectrum globally")
            spectrum_raw = spectra.sum((-1, -2))
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

        n_energy, n_y, n_x = spectra.shape
        n_pixels = n_y * n_x
        spectra_flat = spectra.permute(1, 2, 0).reshape(n_pixels, n_energy)

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

            conc_maps = conc_local.view(n_y, n_x, n_elements).permute(2, 0, 1)
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

            abundance_maps = conc_final.view(n_y, n_x, n_elements).permute(2, 0, 1).cpu().numpy()
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
            "valid_pixel_mask": valid_pixel_mask.view(n_y, n_x).cpu().numpy(),
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

    def calculate_background_powerlaw(self, spectrum):
        import numpy as np

        """
            From input spectrum, calculate power-law background typical for EDS Bremsstrahlung.
            Uses a conservative approach with heavy smoothing to avoid creating artifacts.
            
            Parameters
            ----------
            spectrum : ndarray
                1D spectrum
            energy_axis : ndarray
                Energy axis corresponding to spectrum
                
            Returns
            -------
            ndarray
                1D array representing the calculated background
            """
        from scipy.ndimage import gaussian_filter

        # Use a larger window for more conservative background estimation
        window_size = 15  # Larger window = smoother, less aggressive
        background = np.zeros_like(spectrum)
        half_window = window_size // 2

        # Estimate background from sliding minimum
        for i in range(len(spectrum)):
            start = max(0, i - half_window)
            end = min(len(spectrum), i + half_window + 1)
            # Use percentile instead of minimum for more robustness
            background[i] = np.percentile(spectrum[start:end], 10)

        # Apply heavy smoothing to avoid creating artificial features
        background = gaussian_filter(background, sigma=5.0)

        # Be very conservative - only subtract 80% of estimated background
        # This prevents over-subtraction that creates artificial peaks
        background = background * 0.8

        # Ensure background doesn't exceed spectrum
        background = np.minimum(background, spectrum * 0.9)

        return background
