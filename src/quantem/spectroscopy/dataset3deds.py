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
        signal_units: str = 'arb. units',
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
        self.dataset_type = 'eds'

    @staticmethod
    def _normalize_specs(specs, param_name='spec', allow_none=False):
        """Parse specs into a flat list of stripped strings."""
        if specs is None:
            if allow_none:
                return None
            raise TypeError(f'{param_name} must be a string or sequence of strings')
        if isinstance(specs, str):
            return [s.strip() for s in specs.split(',') if s.strip()]
        if isinstance(specs, (list, tuple, set)):
            return [s.strip() for item in specs for s in str(item).split(',') if s.strip()]
        raise TypeError(f'{param_name} must be a string or sequence of strings')

    @staticmethod
    def _normalize_token(text):
        """Return a lowercase alphanumeric-only token for fuzzy matching."""
        return re.sub(r'[^a-z0-9]', '', str(text).lower())

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
        m = re.match(r'^[A-Z][a-z]?', label)
        return m.group(0) if m else None

    @classmethod
    def _ensure_element_info(cls):
        """Load element X-ray line data if not already cached."""
        if cls.element_info is None:
            cls.load_element_info()
        return cls.element_info or {}

    @classmethod
    def _parse_element_selectors(cls, specs, *, allow_none=False, param_name='spec'):
        """Parse element/line specifiers into a dict of {element: set_of_suffixes | None}."""
        tokens = cls._normalize_specs(specs, param_name=param_name, allow_none=allow_none)
        if tokens is None:
            return None

        ordered = cls._ordered_element_keys(cls._ensure_element_info())
        out: dict[str, set[str] | None] = {}
        for raw in tokens:
            compact = re.sub(r'[\s_-]+', '', str(raw).strip())
            if not compact:
                continue
            element = next((k for k in ordered if compact.lower().startswith(k.lower())), None)
            if element is None:
                raise ValueError(f"Could not resolve element from specifier '{raw}'")
            suffix = compact[len(element):]
            out.setdefault(element, None if not suffix else set())
            if suffix and out[element] is not None:
                out[element].add(suffix)
        return out or None

    @staticmethod
    def _canonical_line_name(line_name: str) -> str:
        """Strip any suffix after '__' from a line name."""
        return str(line_name).split('__', 1)[0]

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
            raise ValueError(f"No X-ray lines matched specifier '{raw_spec}' for element '{element}'")
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
    def _select_labels(cls, selector: str, *, labels: list[str], labels_by_element: dict[str, list[str]]):
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
        return 'K' if line_name.startswith('K') else 'L' if line_name.startswith('L') else 'M' if line_name.startswith('M') else '?'

    @staticmethod
    def _peak_confidence(snr_value: float, line_weight: float, distance_value: float, tolerance: float) -> float:
        """Compute a confidence score for a peak-to-line match."""
        sigma = max(float(tolerance) / 3.0, 1e-9)
        return np.log1p(max(float(snr_value), 0.0)) * max(float(line_weight), 0.0) * np.exp(
            -0.5 * (float(distance_value) / sigma) ** 2
        )

    @staticmethod
    def _line_matches_selector(line_name: str, selector: str) -> bool:
        """Check whether a line name matches a shell or substring selector."""
        line = str(line_name).strip().lower()
        selector = str(selector).strip().lower()
        return line.startswith(selector) if selector in {'k', 'l', 'm'} else selector in line

    @classmethod
    def _line_allowed_for_element(cls, element_name: str, line_name: str, edge_filters=None) -> bool:
        """Return True if the line passes the edge filter for its element."""
        selectors = None if edge_filters is None else edge_filters.get(str(element_name))
        return selectors is None or any(cls._line_matches_selector(line_name, token) for token in selectors)

    def _get_spectrum_images(self, method='integration'):
        """Retrieve cached spectrum images for the given method."""
        return {
            'integration': getattr(self, '_spectrum_images', None),
            'fit': getattr(self, '_spectrum_images_pytorch', None),
        }.get(method)

    @staticmethod
    def _shell_preference_factor(shell_name: str) -> float:
        """Return a down-weighting factor for M-shell lines."""
        return 0.72 if shell_name == 'M' else 1.0

    @staticmethod
    def _merge_edge_filters(requested, saved):
        """Merge requested and saved edge filters, unioning selectors per element."""
        if requested and saved:
            merged = dict(saved)
            for element, selectors in requested.items():
                current = merged.get(element)
                merged[element] = None if current is None or selectors is None else set(current).union(selectors)
            return merged
        return requested or saved

    @staticmethod
    def _estimate_snr_thresholds(snr_values, peaks, snr_min=None, snr_threshold=None):
        """Auto-estimate snr_min and snr_threshold from peak SNR distribution."""
        snr_values = np.asarray(snr_values, dtype=float)
        snr_values = snr_values[np.isfinite(snr_values)]

        if snr_min is None:
            if snr_values.size:
                sorted_snrs = np.sort(snr_values)
                target_rank = min(sorted_snrs.size, int(np.clip(2 * int(peaks), 12, 64)))
                rank_cutoff = float(sorted_snrs[-target_rank])
                q30, q40, q50 = np.percentile(sorted_snrs, [30, 40, 50])
                snr_min = float(np.clip(min(q50, max(q30, 0.35 * rank_cutoff, 0.9 * q40)), 7.0, 14.0))
            else:
                snr_min = 8.0
        else:
            snr_min = float(snr_min)

        if snr_threshold is None:
            if snr_values.size:
                high = snr_values[snr_values >= snr_min]
                high = high if high.size else snr_values
                high = np.sort(high)[::-1]
                anchor = high[: min(high.size, int(np.clip(int(peaks), 10, 40)))]
                med, q75, q90 = np.percentile(anchor, [50, 75, 90])
                snr_threshold = float(np.clip(max(med, 0.7 * q75, 2.5 * snr_min), max(2.5 * snr_min, snr_min), q90))
            else:
                snr_threshold = max(4.0 * snr_min, 30.0)
        else:
            snr_threshold = float(snr_threshold)

        return snr_min, snr_threshold

    def x_ray_lookup(self, spec: str | list[str] | tuple[str, ...] | set[str]) -> tuple[np.ndarray, np.ndarray, list[str]]:
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
        specs = type(self)._normalize_specs(spec, param_name='spec')

        rows: list[tuple[str, float, float]] = []
        for raw in specs:
            compact = re.sub(r'[\s_-]+', '', str(raw).strip())
            if not compact:
                continue
            element = next((k for k in ordered if compact.lower().startswith(k.lower())), None)
            if element is None:
                raise ValueError(f"Could not resolve element from specifier '{raw}'")
            suffix = compact[len(element):]
            for line_name, line_info in type(self)._iter_selected_lines(element, suffix, raw_spec=str(raw)):
                if not isinstance(line_info, dict):
                    continue
                try:
                    energy = float(line_info.get('energy (keV)', line_info.get('energy')))
                except (TypeError, ValueError):
                    continue
                try:
                    weight = float(line_info.get('weight', 0.0))
                except (TypeError, ValueError):
                    weight = 0.0
                rows.append((f'{element}{type(self)._canonical_line_name(line_name)}', energy, weight))

        if not rows:
            raise ValueError(f'No X-ray lines matched specifier(s): {specs}')

        unique = sorted(
            {(lbl, round(float(e), 12), round(float(w), 12)) for lbl, e, w in rows},
            key=lambda t: (t[1], -t[2], t[0]),
        )
        return (
            np.asarray([e for _, e, _ in unique], dtype=float),
            np.asarray([w for _, _, w in unique], dtype=float),
            [lbl for lbl, _, _ in unique],
        )

    def generage_spectrum_images(self, elements=None, width=0.15, return_maps=False, show=True):
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
        show : bool, optional
            If ``True``, display the generated maps.

        Returns
        -------
        tuple[ndarray, list[str]] | None
            Only returned when *return_maps* is ``True``.
        """
        if elements is None:
            if self.model_elements is None:
                raise ValueError('elements must be specified')
            elements = list(self.model_elements)

        energies, _, labels = self.x_ray_lookup(elements)
        keep = (energies > self.energy_axis.min()) & (energies < self.energy_axis.max())
        energies = energies[keep]
        labels = [label for label, ok in zip(labels, keep) if ok]

        mask = (self.energy_axis[:, None] > energies[None, :] - width) & (self.energy_axis[:, None] < energies[None, :] + width)
        n, h, w = self.array.shape
        maps = (mask.astype(self.array.dtype).T @ self.array.reshape(n, -1)).reshape(mask.shape[1], h, w)

        self._spectrum_images = {**getattr(self, '_spectrum_images', {}), **dict(zip(labels, maps))}
        if show:
            self.show_spectrum_images(x_ray_lines=elements)
        if return_maps:
            return maps, labels

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
        specs = type(self)._normalize_specs(spec, param_name='spec')
        arr = np.asarray(self.array, dtype=float)
        energy_axis = np.asarray(self.energy_axis, dtype=float)
        energy_min, energy_max = float(energy_axis.min()), float(energy_axis.max())

        selector_masks, integrated_maps = {}, {}
        for selector in map(str, specs):
            line_energies, _, _ = self.x_ray_lookup(selector.strip())
            line_energies = line_energies[(line_energies >= energy_min) & (line_energies <= energy_max)]
            if not len(line_energies):
                raise ValueError(f"No X-ray lines for selector '{selector}' are within the dataset energy range")

            mask = np.any(
                (energy_axis[:, None] >= line_energies[None, :] - width)
                & (energy_axis[:, None] <= line_energies[None, :] + width),
                axis=1,
            )
            selector_masks[selector] = mask
            integrated_maps[selector] = arr[mask].sum(axis=0)

        if show:
            cmap = kwargs.pop('cmap', 'magma')
            if len(integrated_maps) == 1:
                selector = next(iter(integrated_maps))
                self.show_energy_window_map(
                    energy_window=[energy_min, energy_max],
                    roi=kwargs.pop('roi', None),
                    roi_cal=kwargs.pop('roi_cal', None),
                    mask=selector_masks[selector],
                    data_type=kwargs.pop('data_type', 'eds'),
                    cmap=cmap,
                    show=True,
                )
            else:
                show_2d(
                    list(integrated_maps.values()),
                    title=list(integrated_maps),
                    cmap=cmap,
                    scalebar={'sampling': self.sampling[1], 'units': self.units[1]},
                    **kwargs,
                )

        return integrated_maps if return_maps or len(integrated_maps) != 1 else next(iter(integrated_maps.values()))

    def integrate(self, spec, width=0.15, return_maps=False, show=True, **kwargs):
        """Convenience wrapper for Integrate."""
        return self.Integrate(spec=spec, width=width, return_maps=return_maps, show=show, **kwargs)

    def show_spectrum_images(self, x_ray_lines=None, return_fig=False, method='integration', **kwargs):
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
            raise ValueError('No spectrum images found. Run generage_spectrum_images(...) first.')

        line_map = {str(k): np.asarray(v) for k, v in spectrum_images.items()}
        labels = list(line_map)
        labels_by_element = type(self)._group_labels_by_element(labels)

        def sum_maps(lbls):
            return np.sum([line_map[lbl] for lbl in lbls], axis=0)

        specs = type(self)._normalize_specs(x_ray_lines, param_name='x_ray_lines', allow_none=True)
        if not specs:
            titles = sorted(labels_by_element)
            images = [sum_maps(labels_by_element[t]) for t in titles]
        else:
            selected = [type(self)._select_labels(str(raw), labels=labels, labels_by_element=labels_by_element) for raw in specs]
            if any(not s for s in selected):
                bad = next(raw for raw, s in zip(specs, selected) if not s)
                raise ValueError(f"No spectrum images matched selector '{bad}'")
            images = [line_map[s[0]] if len(s) == 1 else sum_maps(s) for s in selected]
            titles = [s[0] if len(s) == 1 else str(raw).strip() for raw, s in zip(specs, selected)]

        fig, ax = show_2d(
            images,
            title=titles,
            cmap=kwargs.pop('cmap', 'magma'),
            scalebar={'sampling': self.sampling[1], 'units': self.units[1]},
            returnfig=True,
            **kwargs,
        )
        if return_fig:
            return fig, ax

    def _build_pytorch_spectrum_images(self, abundance_maps: np.ndarray, element_names: list[str] | tuple[str, ...]) -> dict[str, np.ndarray]:
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

    def quantify_composition_cliff_lorimer(self, k_factors, method='integration', return_maps=False, verbose=True):
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
            raise ValueError('k_factors must be a non-empty dict')
        spectrum_images = self._get_spectrum_images(method)
        if not spectrum_images:
            raise ValueError('No spectrum images available for quantification')

        ordered_elements = type(self)._ordered_element_keys(type(self)._ensure_element_info())
        line_map = {str(k): np.asarray(v, dtype=float) for k, v in spectrum_images.items()}
        labels = list(line_map)
        labels_by_element = type(self)._group_labels_by_element(labels)

        def match(selector: str) -> list[str]:
            return type(self)._select_labels(selector, labels=labels, labels_by_element=labels_by_element)

        intensities, weighted_intensities = {}, {}
        selector_maps = {} if return_maps else None
        intensity_maps = {} if return_maps else None
        weighted_intensity_maps = {} if return_maps else None

        for selector, k_raw in k_factors.items():
            k_val = float(k_raw)
            sel_labels = match(str(selector).strip())
            if not sel_labels:
                raise ValueError(f'No spectrum images matched selector {selector!r}')

            matched_elements = {type(self)._resolve_element_from_label(lbl, ordered_elements) for lbl in sel_labels} - {None}
            if len(matched_elements) != 1:
                raise ValueError(f'Selector {selector!r} matched multiple elements: {sorted(matched_elements)}')
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
                weighted_intensity_maps[element] = weighted_intensity_maps.get(element, 0) + weighted_map

        if len(weighted_intensities) < 2:
            raise ValueError('At least two elements are required for Cliff-Lorimer quantification')

        weighted_sum = sum(weighted_intensities.values())
        atomic_percent = {el: 100.0 * val / weighted_sum if weighted_sum > 0 else 0.0 for el, val in weighted_intensities.items()}

        if type(self).atomic_weights is None:
            type(self).load_atomic_weights()
        atomic_weights = type(self).atomic_weights or {}
        missing = [el for el in atomic_percent if el not in atomic_weights]
        if missing:
            raise ValueError(f'Atomic weights not found for elements: {missing}')

        weight_sum = sum((atomic_percent[el] / 100.0) * float(atomic_weights[el]) for el in atomic_percent)
        weight_percent = {
            el: (atomic_percent[el] / 100.0) * float(atomic_weights[el]) / weight_sum * 100.0 if weight_sum > 0 else 0.0
            for el in atomic_percent
        }

        ordered = sorted(weighted_intensities, key=weighted_intensities.get, reverse=True)
        table_text = '\n'.join([
            'Element  Intensity      Weighted Intensity    Atomic %    Weight %',
            '-------  -------------  --------------------  ----------  ----------',
            *[
                f'{el:<7}  {intensities[el]:>13.3f}  {weighted_intensities[el]:>20.3f}  {atomic_percent[el]:>10.3f}  {weight_percent[el]:>10.3f}'
                for el in ordered
            ],
        ])
        result = {
            'intensities': intensities,
            'weighted_intensities': weighted_intensities,
            'atomic_percent': atomic_percent,
            'weight_percent': weight_percent,
            'summary_table': table_text,
        }
        if verbose:
            print(table_text)

        if return_maps:
            weighted_stack = np.stack(list(weighted_intensity_maps.values()), axis=0)
            weighted_sum_map = weighted_stack.sum(axis=0)
            atomic_percent_maps = {
                el: np.divide(wmap * 100.0, weighted_sum_map, out=np.zeros_like(weighted_sum_map, dtype=float), where=weighted_sum_map > 0)
                for el, wmap in weighted_intensity_maps.items()
            }
            mass_maps = {el: atomic_percent_maps[el] / 100.0 * float(atomic_weights[el]) for el in atomic_percent_maps}
            mass_sum_map = np.sum(np.stack(list(mass_maps.values()), axis=0), axis=0)
            weight_percent_maps = {
                el: np.divide(mmap * 100.0, mass_sum_map, out=np.zeros_like(mass_sum_map, dtype=float), where=mass_sum_map > 0)
                for el, mmap in mass_maps.items()
            }
            result.update({
                'selector_maps': selector_maps,
                'intensity_maps': intensity_maps,
                'weighted_intensity_maps': weighted_intensity_maps,
                'atomic_percent_maps': atomic_percent_maps,
                'weight_percent_maps': weight_percent_maps,
            })
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
        refline : str | None, optional
            Reserved for future use.
        ignore_elements : str | sequence[str] | None, optional
            Elements to exclude from autodetection.
        ignore_range : sequence[float] | None, optional
            Energy range ``[emin, emax]`` whose peaks are ignored.  Defaults to
            ``[0, 0.25]`` keV to skip the noise floor.
        threshold : float, optional
            Legacy parameter (currently unused).  SNR filtering is controlled
            by *snr_min* and *snr_threshold*.
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
        snr_min : float | None, optional
            Minimum signal-to-noise ratio for a peak to be displayed.  If
            ``None``, estimated automatically from the SNR distribution.
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
        return_details : bool, optional
            If ``True``, return a dict with detection details instead of the
            figure.

        Returns
        -------
        tuple[Figure, tuple[Axes, Axes]] | dict
            By default returns ``(fig, (ax_img, ax_spec))``.  When
            *return_details* is ``True``, returns a dict containing
            ``detected_elements``, ``element_confidence``, ``display_peaks``,
            ``peak_matches``, ``snr_min``, ``snr_threshold``, and the figure.
        """
        type(self)._ensure_element_info()
        all_info = type(self).element_info or {}
        grid_peaks = grid_peaks or {}
        ignore_range = [0, 0.25] if ignore_range is None else ignore_range
        ignored_elements = set(map(str, type(self)._normalize_specs(ignore_elements, allow_none=True) or []))
        min_line_weight = max(float(min_line_weight), 0.0)

        requested = type(self)._parse_element_selectors(elements, allow_none=True, param_name='elements')
        saved = {
            str(k): (set(map(str, v.keys())) if isinstance(v, dict) and v else None)
            for k, v in (getattr(self, 'model_elements', {}) or {}).items()
        } or None
        edge_filters = type(self)._merge_edge_filters(requested, saved)
        requested_elements = set(edge_filters) if edge_filters else None

        mode = (str(mode).strip().lower() if mode is not None else None) or ('elements_only' if requested_elements else 'autofill')
        search_elements = requested_elements if mode == 'elements_only' else None
        preferred_elements = set(map(str, requested_elements or [])) if mode == 'elements_preferred' else set()
        reference_elements = requested_elements

        fig, (ax_img, ax_spec) = self.show_mean_spectrum(
            roi=roi,
            roi_cal=roi_cal,
            energy_range=energy_range,
            mask=mask,
            data_type='eds',
            show=False,
        )
        spec = self.calculate_mean_spectrum(
            roi=roi,
            roi_cal=roi_cal,
            energy_range=energy_range,
            ignore_range=ignore_range,
            mask=mask,
        )
        E = float(self.origin[0]) + float(self.sampling[0]) * np.arange(self.shape[0])
        if energy_range is not None:
            keep = (energy_range[0] <= E) & (E <= energy_range[1])
            E = E[keep]
            spec = spec[keep]

        def in_ignore(energy):
            return len(ignore_range) == 2 and ignore_range[0] <= float(energy) <= ignore_range[1]

        peak_indices, props = find_peaks(spec, height=0, distance=5)
        peak_heights = props['peak_heights']
        background_std = np.nanstd(spec[spec <= np.nanpercentile(spec, 50)])
        if not np.isfinite(background_std) or background_std <= 0:
            background_std = np.nanstd(spec)
        if not np.isfinite(background_std) or background_std <= 0:
            background_std = 1.0

        snr_values = np.asarray([height / background_std for height in peak_heights], dtype=float)
        snr_min, snr_threshold = type(self)._estimate_snr_thresholds(snr_values, peaks, snr_min, snr_threshold)

        display_peaks = [
            (int(i), float(h), float(E[i]), float(h / background_std))
            for i, h in zip(peak_indices, peak_heights)
            if not in_ignore(E[i]) and h / background_std >= snr_min
        ]
        display_peaks.sort(key=lambda item: item[3], reverse=True)
        significant_peaks = list(display_peaks)
        display_peaks = display_peaks[:peaks]

        def candidate_matches(peak_energy, snr, allowed_elements=None):
            matches = []
            for element_name, lines in all_info.items():
                if allowed_elements is not None and element_name not in allowed_elements:
                    continue
                for line_name, line_info in lines.items():
                    if not type(self)._line_allowed_for_element(element_name, line_name, edge_filters):
                        continue
                    line_weight = float(line_info.get('weight', 0.5))
                    line_energy = float(line_info['energy (keV)'])
                    shell = type(self)._line_shell(line_name)
                    tol = tolerance * 0.5 if shell == 'M' and ('Ma' not in line_name and 'Mb' not in line_name) else tolerance
                    distance = abs(peak_energy - line_energy)
                    if line_weight < min_line_weight or distance > tol:
                        continue
                    score = type(self)._peak_confidence(snr, line_weight, distance, tolerance) * type(self)._shell_preference_factor(shell)
                    matches.append({
                        'element': str(element_name),
                        'line': str(line_name),
                        'weight': line_weight,
                        'distance': distance,
                        'score': float(score),
                        'shell': shell,
                    })
            matches.sort(key=lambda m: m['score'], reverse=True)
            return matches

        peak_matches = []
        for peak_idx, height, peak_energy, snr in display_peaks:
            matches = candidate_matches(peak_energy, snr, search_elements)
            if not matches:
                continue
            best = matches[0]
            peak_matches.append((
                peak_idx,
                height,
                peak_energy,
                snr,
                best['element'],
                f"{best['element']} {best['line']}",
                best['distance'],
                best['line'],
                best['weight'],
                best['score'],
            ))

        element_stats, line_evidence = {}, {}
        for _, _, peak_energy, snr, element, _, distance, line_name, line_weight, conf in peak_matches:
            if search_elements is not None and element not in search_elements:
                continue
            shell = type(self)._line_shell(line_name)
            stats = element_stats.setdefault(element, {
                'raw_conf': 0.0,
                'shells': set(),
                'lines': set(),
                'strong_matches': 0,
                'match_count': 0,
                'best_match_conf': 0.0,
                'best_match_snr': 0.0,
                'best_match_energy': 0.0,
                'best_match_distance': float('inf'),
                'best_match_weight': 0.0,
                'best_match_shell': '?',
            })
            label = f'{element} {line_name}'
            evidence = line_evidence.setdefault(label, {'match_count': 0, 'strong_matches': 0, 'best_conf': 0.0, 'best_snr': 0.0, 'energies': []})

            stats['raw_conf'] += float(conf)
            stats['shells'].add(shell)
            stats['lines'].add(line_name)
            stats['match_count'] += 1
            stats['strong_matches'] += int(snr > snr_threshold and distance < distance_threshold_for_sample)
            if conf > stats['best_match_conf']:
                stats.update({
                    'best_match_conf': float(conf),
                    'best_match_snr': float(snr),
                    'best_match_energy': float(peak_energy),
                    'best_match_distance': float(distance),
                    'best_match_weight': float(line_weight),
                    'best_match_shell': shell,
                })

            evidence['match_count'] += 1
            evidence['energies'].append(float(peak_energy))
            evidence['strong_matches'] += int(snr > snr_threshold and distance < distance_threshold_for_sample)
            if conf > evidence['best_conf']:
                evidence['best_conf'] = float(conf)
                evidence['best_snr'] = float(snr)

        element_confidence = {}
        # --- Intensity ratio check and multi-peak pattern boost ---
        for element, stats in element_stats.items():
            valid_shells = {shell for shell in stats['shells'] if shell in {'K', 'L', 'M'}}
            shell_bonus = float(np.sqrt(max(1, len(valid_shells))))
            line_bonus = 1.0 + 0.30 * float(np.log1p(max(0, len(stats['lines']) - 1)))
            strong_bonus = 1.0 + 0.40 * float(np.log1p(stats['strong_matches']))
            major_bonus = 1.20 if {'K', 'L'} & valid_shells else 1.0

            # Intensity ratio logic
            element_peak_intensities = {}
            for _, height, peak_energy, snr, el, _, distance, line_name, line_weight, conf in peak_matches:
                if el == element:
                    element_peak_intensities.setdefault(line_name, []).append(float(height))
            # Only consider if at least 2 lines detected
            if len(element_peak_intensities) >= 2:
                observed = []
                expected = []
                for line_name, intensities in element_peak_intensities.items():
                    observed.append(max(intensities))
                    weight = all_info.get(element, {}).get(line_name, {}).get('weight', None)
                    try:
                        expected.append(float(weight) if weight is not None else 0.0)
                    except Exception:
                        expected.append(0.0)
                obs_sum = sum(observed)
                exp_sum = sum(expected)
                if obs_sum > 0 and exp_sum > 0:
                    observed_norm = [x / obs_sum for x in observed]
                    expected_norm = [x / exp_sum for x in expected]
                    ratio_score = 1.0 - (sum(abs(o - e) for o, e in zip(observed_norm, expected_norm)) / 2.0)
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
            k_lines = {'Ka1', 'Kb1'}
            l_lines = {'La1', 'Lb1'}
            m_lines = {'Ma1', 'Mb1'}
            pattern_factor = 1.0
            if k_lines.issubset(matched_lines):
                pattern_factor = 3.0
            elif l_lines.issubset(matched_lines):
                pattern_factor = 2.5
            elif m_lines.issubset(matched_lines):
                pattern_factor = 2.0

            element_confidence[element] = stats['raw_conf'] * shell_bonus * line_bonus * strong_bonus * major_bonus * ratio_factor * pattern_factor

        detected_elements = set()
        if element_confidence:
            conf_values = np.asarray(list(element_confidence.values()), dtype=float)
            poisson_mdl_snr = 3.0
            cutoff = max(float(np.percentile(conf_values, 45)), 0.30 * float(conf_values.max()))
            for element, confidence in element_confidence.items():
                stats = element_stats[element]
                lines = set(stats['lines'])
                # Criterion 1: Both main lines matched (pattern match) → always autodetect
                strong_pattern = (
                    {'Ka1', 'Kb1'}.issubset(lines)
                    or {'La1', 'Lb1'}.issubset(lines)
                    or {'Ma1', 'Mb1'}.issubset(lines)
                ) and confidence > 0
                # Criterion 2: High confidence above cutoff and sufficient SNR
                high_confidence = confidence >= cutoff and stats['best_match_snr'] >= poisson_mdl_snr
                if strong_pattern or high_confidence:
                    detected_elements.add(element)

        dominant_elements = set()
        if element_confidence:
            conf_values = np.asarray(list(element_confidence.values()), dtype=float)
            conf_floor = max(float(np.median(conf_values)) if conf_values.size else 0.0, 1e-9)
            conf_p80 = float(np.percentile(conf_values, 80)) if conf_values.size > 1 else 0.0
            for element, confidence in element_confidence.items():
                stats = element_stats.get(element, {})
                repeat_support = int(stats.get('match_count', 0)) >= 2 or int(stats.get('strong_matches', 0)) >= 1
                if confidence >= conf_p80 and confidence >= 1.8 * conf_floor and repeat_support:
                    dominant_elements.add(element)

        anchor_elements = {
            element for element in detected_elements
            if element in element_stats and element_stats[element].get('best_match_energy', 0.0) >= 6.0 and element_stats[element].get('best_match_weight', 0.0) >= 0.8
        }
        max_detected_conf = max([element_confidence.get(el, 0.0) for el in detected_elements], default=0.0)

        def prior_boost(element):
            prior = float(element_confidence.get(element, 0.0)) / max(float(max_detected_conf), 1e-9)
            factor = 1.0 + 0.5 * prior
            if prior >= 0.90:
                factor *= 1.9
            elif prior >= 0.75:
                factor *= 1.5
            elif prior >= 0.55:
                factor *= 1.2
            return prior, factor

        def consistency_boost(element, line_name, peak_energy):
            if element not in dominant_elements:
                return 1.0
            evidence = line_evidence.get(f'{element} {line_name}')
            if not evidence or not any(abs(float(peak_energy) - float(prev)) >= 0.04 for prev in evidence.get('energies', [])):
                return 1.0
            best_conf = float(evidence.get('best_conf', 0.0))
            best_snr = float(evidence.get('best_snr', 0.0))
            strong = int(evidence.get('strong_matches', 0))
            line_weight = float((all_info.get(element, {}).get(line_name, {}) or {}).get('weight', 0.5))
            tier = 1.0 + 0.7 * max(0.0, line_weight - 0.35)
            if strong >= 1 and best_conf >= 1.4:
                return min(3.2, 2.4 * tier)
            if best_conf >= 1.1 and best_snr >= max(snr_min, 0.75 * snr_threshold):
                return min(2.6, 1.9 * tier)
            if best_conf >= 0.8:
                return min(2.0, 1.5 * tier)
            return min(1.5, 1.2 * tier)

        def dominant_boost(element):
            if element not in dominant_elements:
                return 1.0
            prior, _ = prior_boost(element)
            stats = element_stats.get(element, {})
            repeat_support = max(int(stats.get('strong_matches', 0)), max(0, int(stats.get('match_count', 0)) - 1))
            base = 2.30 if prior >= 0.90 else 1.85 if prior >= 0.75 else 1.45
            if repeat_support >= 2:
                base *= 1.10
            return min(base, 2.60)

        def reranked_matches(peak_energy, snr, allowed_elements=None, top_k=None):
            # Compute which elements have both main lines matched (pattern boost)
            element_to_lines = {}
            for _, _, _, _, el, _, _, ln, _, _ in peak_matches:
                element_to_lines.setdefault(el, set()).add(ln)
            scored = []
            for match in candidate_matches(peak_energy, snr, allowed_elements):
                element, line_name, shell = match['element'], match['line'], match['shell']
                prior, prior_factor = prior_boost(element)
                pref = 1.35 if element in preferred_elements else 1.0
                anchor = 1.15 if element in anchor_elements and shell in {'K', 'L'} else 1.0
                consistency = consistency_boost(element, line_name, peak_energy)
                dom = dominant_boost(element)
                # Pattern boost: if both main lines for K, L, or M are matched by detected peaks, boost candidate score
                lines_matched = element_to_lines.get(element, set())
                k_lines = {'Ka1', 'Kb1'}
                l_lines = {'La1', 'Lb1'}
                m_lines = {'Ma1', 'Mb1'}
                pattern_factor = 1.0
                if k_lines.issubset(lines_matched):
                    pattern_factor = 3.0
                elif l_lines.issubset(lines_matched):
                    pattern_factor = 2.5
                elif m_lines.issubset(lines_matched):
                    pattern_factor = 2.0
                if shell == 'M':
                    prior_factor = 1.0 + 0.3 * prior
                    consistency = 1.0
                    dom = min(dom, 1.30)
                score = match['score'] * prior_factor * pref * anchor * consistency * dom * pattern_factor
                scored.append({**match, 'score': float(score)})

            scored.sort(key=lambda m: m['score'], reverse=True)
            if mode == 'elements_preferred' and preferred_elements:
                preferred = [m for m in scored if m['element'] in preferred_elements]
                scored = preferred + [m for m in scored if m['element'] not in preferred_elements] if preferred else scored

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
            used_elements = {unique[0]['element']}
            for match in unique[1:]:
                if match['element'] in used_elements:
                    continue
                selected.append(match)
                used_elements.add(match['element'])
                if len(selected) >= int(top_k):
                    return selected
            for match in unique[1:]:
                if match not in selected:
                    selected.append(match)
                if len(selected) >= int(top_k):
                    break
            return selected

        rematch_allowed = {str(match[4]) for match in peak_matches if str(match[4]) not in ignored_elements}
        rematch_allowed.update(map(str, detected_elements))
        rematch_allowed.update(preferred_elements)

        refined_peak_matches = []
        raw_match_by_idx = {int(match[0]): match for match in peak_matches}
        for peak_idx, height, peak_energy, snr in display_peaks:
            best = reranked_matches(peak_energy, snr, rematch_allowed or None, top_k=1)
            best = best[0] if best else None
            if best is None:
                continue
            refined_peak_matches.append((
                peak_idx,
                height,
                peak_energy,
                snr,
                best['element'],
                f"{best['element']} {best['line']}",
                best['distance'],
                best['line'],
                best['weight'],
                best['score'],
            ))
        peak_matches = refined_peak_matches

        matched_elements = {str(match[4]) for match in peak_matches}
        detected_elements = {str(el) for el in detected_elements if str(el) in matched_elements and str(el) not in ignored_elements}
        if mode == 'elements_preferred':
            detected_elements.update(str(el) for el in preferred_elements if str(el) in matched_elements)
        refined_match_by_idx = {int(match[0]): match for match in peak_matches}

        final_matches_by_element: dict[str, set[str]] = {}
        for _, _, _, _, element, _, _, line_name, _, _ in peak_matches:
            if element not in ignored_elements:
                final_matches_by_element.setdefault(element, set()).add(str(line_name))

        candidate_elements = sorted(str(el) for el in final_matches_by_element if str(el) not in detected_elements)
        possible_elements = set(candidate_elements)
        possible_line_labels = {f'{element} {line}' for _, _, _, _, element, _, _, line, _, _ in peak_matches if element in possible_elements}

        def format_elements_with_lines(names):
            items = []
            for element in sorted(map(str, names)):
                lines = sorted(map(str, final_matches_by_element.get(element, set())))
                line_strs = [str(line) for line in lines]
                items.append(f"{element} [{', '.join(line_strs)}]" if lines else f'{element}')
            return ', '.join(items)

        def format_saved(edge_filters):
            if edge_filters is None:
                return 'None'
            out = []
            for element in sorted(map(str, edge_filters)):
                selectors = edge_filters.get(element)
                out.append(f'{element} [all]' if not selectors else f"{element} [{', '.join(sorted(map(str, selectors)))}]")
            return '\n'.join(out) if out else 'None'

        print(f"\nAutodetected: {format_elements_with_lines(detected_elements) if detected_elements else 'None'}")
        if dominant_elements:
            dominant_str = ', '.join(f"{el} (conf={element_confidence.get(str(el), 0.0):.2f})" for el in sorted(dominant_elements, key=lambda el: element_confidence.get(str(el), 0.0), reverse=True))
            print(f'Dominant (strong prior): {dominant_str}')
        print(f"Possible: {format_elements_with_lines(candidate_elements) if candidate_elements else 'None'}")

        elements_for_color = set(detected_elements) | {str(match[4]) for match in peak_matches}
        if search_elements is not None:
            elements_for_color.update(map(str, search_elements))
        palette = ['#1f77b4', '#d62728', '#2ca02c', '#9467bd', '#ff7f0e', '#8c564b', '#e377c2', '#17becf', '#bcbd22', '#7f7f7f', '#003f5c', '#7a5195', '#ef5675', '#ffa600', '#2f4b7c']
        element_color_map = {el: palette[i % len(palette)] for i, el in enumerate(sorted(elements_for_color))}

        detected_sample_peaks = {
            float(peak_energy)
            for _, _, peak_energy, _, element, _, _, _, _, _ in peak_matches
            if element in detected_elements
        }
        y_min = float(np.nanmin(spec)) if len(spec) else 0.0
        y_max = float(np.nanmax(spec)) if len(spec) else 1.0
        y_scale = max(max(1e-9, y_max - y_min), abs(y_max), 1.0)
        y_dot = -0.04 * y_scale

        def infer_requested_color(peak_energy):
            if reference_elements is None:
                return None
            best_element, best_distance = None, float('inf')
            for element in reference_elements:
                for line_name, line_info in (all_info.get(str(element), {}) or {}).items():
                    if not type(self)._line_allowed_for_element(str(element), line_name, edge_filters):
                        continue
                    try:
                        distance = abs(float(peak_energy) - float(line_info.get('energy (keV)')))
                    except (TypeError, ValueError):
                        continue
                    if distance < best_distance:
                        best_distance, best_element = distance, str(element)
            return best_element

        table_rows = []
        for peak_idx, height, peak_energy, snr in display_peaks:
            match = refined_match_by_idx.get(int(peak_idx))
            is_sample = float(peak_energy) in detected_sample_peaks
            is_possible = match is not None and str(match[4]) in possible_elements
            color = element_color_map.get(match[4], 'red') if match is not None else element_color_map.get(str(infer_requested_color(peak_energy)), 'red')

            if not in_ignore(peak_energy):
                # Only plot solid lines for matched peaks (autodetected or requested elements)
                if match is not None:
                    ax_spec.axvline(peak_energy, color=color, linestyle='-', alpha=0.7, linewidth=1.5)
                else:
                    ax_spec.plot([peak_energy], [y_dot], marker='|', markersize=4, color='gray', alpha=0.8, linestyle='None')

                if show_text and match is not None:
                    for grid_element, grid_energy in grid_peaks.items():
                        if abs(peak_energy - grid_energy) < 0.1:
                            ax_spec.text(peak_energy, height * 0.7, f'{grid_element}\n(grid)', ha='center', va='bottom', fontsize=8, color='gray', style='italic')
                            print(f'Peak at {peak_energy} keV may come from the grid.')
                            break

            def label_with_energy_and_ratio(label, detected_peak_intensity=None):
                # label is like 'Fe Ka', want to append (energy, ratio) from all_info and observed/expected
                if not label or label == '-' or label == 'Unmatched' or label == 'Unknown':
                    return label
                parts = label.split()
                if len(parts) < 2:
                    return label
                element, line = parts[0], parts[1].replace('*','')
                line_info = all_info.get(element, {}).get(line, {})
                ref_energy = None
                if isinstance(line_info, dict):
                    ref_energy = line_info.get('energy (keV)', line_info.get('energy'))
                try:
                    ref_energy = float(ref_energy)
                except (TypeError, ValueError):
                    ref_energy = None
                # Compute observed/expected ratio: use detected peak intensity / expected weight
                ratio_str = ''
                weight = line_info.get('weight', None)
                try:
                    weight = float(weight) if weight is not None else 0.0
                except Exception:
                    weight = 0.0
                if detected_peak_intensity is not None and weight:
                    try:
                        ratio = float(detected_peak_intensity) / float(weight)
                        ratio_str = f", {ratio:.2f}"
                    except Exception:
                        ratio_str = ''
                label_core = label.rstrip('*')
                star = '*' if label.endswith('*') else ''
                if ref_energy is not None:
                    return f"{label_core} ({ref_energy:.3f}{ratio_str}){star}"
                else:
                    return label

            if match is None:
                table_rows.append((peak_energy, height, snr, 'Unmatched' if search_elements is not None else 'Unknown', '-', '-'))
                continue

            allowed_for_table = set(map(str, search_elements)) if search_elements is not None else ({str(el) for el in all_info if str(el) not in ignored_elements} or None)
            ranked = reranked_matches(peak_energy, snr, allowed_for_table, top_k=3)
            labels = [(f"{m['element']} {m['line']}", float(m['score']), m['element'], m['line']) for m in ranked]
            best_label = f'{match[4]} {match[7]}'

            def fmt(label, score=None):
                label = f'{label}*' if str(label).split()[0] in detected_elements else label
                return label

            # Gather all intensities for this element for ratio calculation
            all_element_intensities = {}
            for l in all_info.get(match[4], {}):
                # Find the highest observed intensity for each line
                obs = 0.0
                for _, h, _, _, el, _, _, ln, _, _ in peak_matches:
                    if el == match[4] and ln == l:
                        obs = max(obs, float(h))
                weight = all_info.get(match[4], {}).get(l, {}).get('weight', None)
                try:
                    weight = float(weight) if weight is not None else 0.0
                except Exception:
                    weight = 0.0
                all_element_intensities[l] = (obs, weight)

            remaining = [(label, score, elem, line) for label, score, elem, line in labels if label.lower() != best_label.lower()]
            # For each label, show ratio for that line
            def get_peak_intensity(elem, line):
                obs = 0.0
                for _, h, _, _, el, _, _, ln, _, _ in peak_matches:
                    if el == elem and ln == line:
                        obs = max(obs, float(h))
                return obs

            table_rows.append((
                peak_energy,
                height,
                snr,
                label_with_energy_and_ratio(fmt(best_label), detected_peak_intensity=height),
                label_with_energy_and_ratio(fmt(remaining[0][0]), detected_peak_intensity=height) if len(remaining) > 0 else '-',
                label_with_energy_and_ratio(fmt(remaining[1][0]), detected_peak_intensity=height) if len(remaining) > 1 else '-',
            ))

        current_bottom, current_top = ax_spec.get_ylim()
        ax_spec.set_ylim(bottom=min(current_bottom, y_dot - 0.02 * y_scale, -0.02 * y_scale), top=current_top)

        label_candidates = []
        top_label_y = 0.92
        # Plot reference lines (dotted) ONLY for explicitly requested elements, not for autodetected/possible
        if requested_elements:
            energy_min, energy_max = float(np.min(E)), float(np.max(E))
            matched_by_element = {}
            for _, _, peak_energy, _, element, _, _, _, _, _ in peak_matches:
                matched_by_element.setdefault(str(element), []).append(float(peak_energy))

            for element in sorted(requested_elements):
                candidates = []
                for line_name, line_info in (all_info.get(str(element), {}) or {}).items():
                    if not type(self)._line_allowed_for_element(str(element), line_name, edge_filters):
                        continue
                    try:
                        line_energy = float(line_info.get('energy (keV)'))
                        line_weight = float(line_info.get('weight', 0.0))
                    except (TypeError, ValueError):
                        continue
                    if energy_min <= line_energy <= energy_max:
                        candidates.append((str(line_name), line_energy, line_weight))
                candidates = sorted([c for c in candidates if c[2] >= 0.05] or candidates, key=lambda item: item[2], reverse=True)[:6]
                for line_name, line_energy, _ in candidates:
                    if in_ignore(line_energy):
                        continue
                    # Skip if already matched by a detected peak
                    if any(abs(line_energy - matched_energy) <= max(0.05, 0.5 * tolerance) for matched_energy in matched_by_element.get(str(element), [])):
                        continue
                    color = element_color_map.get(str(element), 'black')
                    style = '--'
                    alpha = 0.45
                    ax_spec.axvline(line_energy, color=color, linestyle=style, alpha=alpha, linewidth=1.2)
                    label_candidates.append((float(line_energy), f'{element} {line_name}', color, style, float(top_label_y), 'axes_top', 8, 'normal', 0.8))

        if show_text and peak_matches:
            label_offset = max(0.03 * y_scale, 0.08)
            # Only label autodetected elements and explicitly requested elements
            label_allowed = set(detected_elements)
            if requested_elements:
                label_allowed.update(str(el) for el in requested_elements)
            for _, height, peak_energy, _, element, match_str, _, _, _, _ in peak_matches:
                is_detected = element in detected_elements and float(peak_energy) in detected_sample_peaks
                if element not in label_allowed or in_ignore(peak_energy):
                    continue
                if not is_detected and element not in (requested_elements or set()):
                    continue
                label = f"{element} {match_str.split()[-1]}" + ('*' if is_detected else '')
                style = '-' if is_detected else '--'
                y_value = float(height + label_offset) if is_detected else float(top_label_y)
                y_mode = 'data' if is_detected else 'axes_top'
                label_candidates.append((float(peak_energy), label, element_color_map.get(element, 'black'), style, y_value, y_mode, 10 if is_detected else 9, 'bold' if is_detected else 'normal', 1.0 if is_detected else 0.9))

        legend_handles, legend_labels = [], set()
        if show_text and label_candidates:
            label_candidates.sort(key=lambda item: item[0])
            groups, current = [], []
            overlap_threshold = max(0.16, 1.1 * float(tolerance))
            for label in label_candidates:
                if not current or abs(label[0] - current[-1][0]) <= overlap_threshold:
                    current.append(label)
                else:
                    groups.append(current)
                    current = [label]
            if current:
                groups.append(current)

            for group in groups:
                if len(group) == 1:
                    peak_energy, label_text, color, _, y_value, y_mode, font_size, font_weight, alpha_value = group[0]
                    common = dict(ha='center', fontsize=font_size, color=color, weight=font_weight, rotation=90, alpha=alpha_value)
                    if y_mode == 'axes_top':
                        ax_spec.text(peak_energy, y_value, label_text, va='top', transform=ax_spec.get_xaxis_transform(), clip_on=True, **common)
                    else:
                        ax_spec.text(peak_energy, y_value, label_text, va='bottom', **common)
                else:
                    for _, label_text, color, linestyle, *_ in group:
                        key = (label_text, str(color), linestyle)
                        if key in legend_labels:
                            continue
                        legend_labels.add(key)
                        legend_handles.append(Line2D([0], [0], color=color, linestyle=linestyle, linewidth=1.5, label=label_text))

        if legend_handles:
            overlap_legend = ax_spec.legend(handles=legend_handles, loc='upper right', fontsize=8, title='Overlapping Labels')
            ax_spec.add_artist(overlap_legend)

        fig.tight_layout()
        plt.show()

        sorted_table_rows = sorted(table_rows, key=lambda item: item[0])
        print(f"{'Energy (keV)':<12} {'Intensity':<12} {'SNR':<8} {'Best Match':<22} {'Alt 2':<22} {'Alt 3':<22}")
        print('-' * 105)
        for peak_energy, height, snr, best_match, alt_2, alt_3 in sorted_table_rows:
            print(f'{peak_energy:<12.3f} {height:<12.2f} {snr:<8.1f} {best_match:<22} {alt_2:<22} {alt_3:<22}')
        print('-' * 105)
        print(f'{len(display_peaks)} of {len(significant_peaks)} peaks above snr_min={snr_min:.1f}, snr_threshold={snr_threshold:.1f} displayed.\n')

        if return_details:
            return {
                'figure': fig,
                'axes': (ax_img, ax_spec),
                'detected_elements': sorted(detected_elements),
                'element_confidence': element_confidence,
                'display_peaks': display_peaks,
                'peak_matches': peak_matches,
                'snr_min': snr_min,
                'snr_threshold': snr_threshold,
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
        """Estimate a power-law Bremsstrahlung background from the spectrum."""
        import numpy as np
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
