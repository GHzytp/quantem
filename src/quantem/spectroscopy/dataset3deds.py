import builtins
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
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
    element_info_path = "xray_lines.json"
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
        self._virtual_images = {}
        self.dataset_type = "EDS"

    def peak_autoid(
        self,
        roi=None,
        energy_range=None,
        elements=None,
        ignore_elements=None,
        ignore_range=None,
        threshold=5.0,
        tolerance=0.15,
        mask=None,
        show_text=True,
        snr_min=None,
        snr_threshold=None,
        distance_threshold_for_sample=0.05,
        grid_peaks=None,
        peaks=15,
        return_details=False,
    ):
        """Auto-detect and label EDS peaks from the mean ROI spectrum.

        This method calls ``show_mean_spectrum`` to generate the baseline plot, then
        overlays peak markers and element labels from X-ray line matching.

        Parameters
        ----------
        return_details : bool, optional
            If True, return full internal results (matches/confidence/peaks).
            If False (default), return only figure and axes.
        """
        if type(self).element_info is None:
            type(self).load_element_info()

        if grid_peaks is None:
            grid_peaks = {}

        if isinstance(elements, str):
            elements = [elements]
        if elements is not None and not isinstance(elements, (list, tuple, builtins.set)):
            raise TypeError("elements must be None, a string, or a sequence of strings")

        def _line_matches_selector(line_name: str, selector: str) -> bool:
            line = str(line_name).strip().lower()
            token = str(selector).strip().lower()
            if token in {"k", "l", "m"}:
                return line.startswith(token)
            return token in line

        def _parse_requested_elements_with_edges(element_specs):
            if element_specs is None:
                return None

            parsed = {}
            for spec in element_specs:
                raw = str(spec).strip()
                if not raw:
                    continue

                tokens = raw.replace(",", " ").split()
                if len(tokens) == 0:
                    continue

                element_key = str(tokens[0])
                selectors = [str(token).strip() for token in tokens[1:] if str(token).strip()]

                if element_key not in parsed:
                    parsed[element_key] = None if len(selectors) == 0 else builtins.set(selectors)
                    continue

                existing = parsed[element_key]
                if existing is None:
                    continue
                if len(selectors) == 0:
                    parsed[element_key] = None
                else:
                    existing.update(selectors)

            return parsed if len(parsed) > 0 else None

        def _edge_filters_from_saved_model(model_elements):
            if not isinstance(model_elements, dict) or len(model_elements) == 0:
                return None

            parsed = {}
            for element_name, lines_info in model_elements.items():
                element_key = str(element_name)
                if isinstance(lines_info, dict) and len(lines_info) > 0:
                    parsed[element_key] = builtins.set(str(line_name) for line_name in lines_info.keys())
                else:
                    parsed[element_key] = None

            return parsed if len(parsed) > 0 else None

        requested_edge_filters = _parse_requested_elements_with_edges(elements)
        saved_model_edge_filters = _edge_filters_from_saved_model(getattr(self, "model_elements", None))
        using_saved_model_elements = False
        if requested_edge_filters is None and elements is None:
            if saved_model_edge_filters is not None:
                requested_edge_filters = saved_model_edge_filters
                using_saved_model_elements = True
        if requested_edge_filters is not None:
            elements = list(requested_edge_filters.keys())

        if isinstance(ignore_elements, str):
            ignore_elements = [ignore_elements]
        if ignore_elements is not None and not isinstance(
            ignore_elements, (list, tuple, builtins.set)
        ):
            raise TypeError("ignore_elements must be None, a string, or a sequence of strings")
        ignored_elements = (
            {str(element_name) for element_name in ignore_elements}
            if ignore_elements is not None
            else builtins.set()
        )

        fig, (ax_img, ax_spec) = self.show_mean_spectrum(
            roi=roi,
            energy_range=energy_range,
            elements=elements,
            ignore_range=ignore_range,
            threshold=threshold,
            tolerance=tolerance,
            mask=mask,
            show_lines=True,
            show_text=show_text,
            snr_min=snr_min,
            snr_threshold=snr_threshold,
            distance_threshold_for_sample=distance_threshold_for_sample,
            grid_peaks=grid_peaks,
            data_type="eds",
            peaks=peaks,
            show=False,
        )

        spec = self.calculate_mean_spectrum(roi, energy_range, ignore_range, mask)
        dE = float(self.sampling[0])
        E0 = float(self.origin[0]) if hasattr(self, "origin") else 0.0
        E = E0 + dE * np.arange(self.shape[0])

        if energy_range is not None:
            indices = np.where((E >= energy_range[0]) & (E <= energy_range[1]))[0]
            E = E[indices]

        if ignore_range is None:
            ignore_range = [0, 0.25]

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
            snr_median = float(np.nanmedian(initial_snrs))
            snr_75th = float(np.nanpercentile(initial_snrs, 75))
            num_high_snr_peaks = int(np.sum(np.array(initial_snrs) > 50))
        else:
            snr_median = 0.0
            snr_75th = 0.0
            num_high_snr_peaks = 0

        if snr_min is None:
            if len(initial_snrs) > 0:
                snr_values = np.asarray(initial_snrs, dtype=float)
                target_rank = max(1, min(int(peaks), int(snr_values.size)))
                kth_largest_snr = float(np.sort(snr_values)[-target_rank])
                distribution_cutoff = float(np.percentile(snr_values, 55))
                adaptive_cutoff = min(distribution_cutoff, 0.9 * kth_largest_snr)
                min_snr = float(np.clip(adaptive_cutoff, 8.0, 20.0))
            else:
                min_snr = 8.0
        else:
            min_snr = float(snr_min)

        if snr_threshold is None:
            if num_high_snr_peaks > 50:
                snr_threshold_for_sample = min(80.0, snr_75th * 1.2)
            elif num_high_snr_peaks > 20:
                snr_threshold_for_sample = min(60.0, snr_75th * 1.1)
            elif num_high_snr_peaks < 10:
                snr_threshold_for_sample = max(30.0, snr_75th * 0.8)
            else:
                snr_threshold_for_sample = 40.0
        else:
            snr_threshold_for_sample = float(snr_threshold)

        all_candidate_peaks = []
        significant_peaks = []
        for peak_idx, height in zip(peak_indices, peak_heights):
            peak_energy = E[peak_idx]

            if ignore_range is not None and len(ignore_range) == 2:
                min_ignore, max_ignore = ignore_range
                if min_ignore <= peak_energy <= max_ignore:
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

        def _peak_confidence(snr_value, line_weight, distance_value):
            quality = max(0.0, 1.0 - (distance_value / max(tolerance, 1e-9)))
            return np.log1p(max(float(snr_value), 0.0)) * (0.5 + float(line_weight)) * (0.5 + quality)

        def _line_allowed_for_element(element_name, line_name, edge_filters=None):
            if edge_filters is None:
                return True

            selectors = edge_filters.get(str(element_name))
            if selectors is None:
                return True

            return any(_line_matches_selector(line_name, token) for token in selectors)

        def _best_line_match(peak_energy, allowed_elements=None, edge_filters=None):
            best_distance = float("inf")
            best_element = None
            best_line_name = None
            best_line_weight = 0.0

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

                    is_m_line = "M" in line_name and not ("Ma" in line_name or "Mb" in line_name)
                    effective_tolerance = tolerance * 0.5 if is_m_line else tolerance

                    if line_weight > 0.3 and distance <= effective_tolerance and distance < best_distance:
                        best_distance = distance
                        best_element = element_name
                        best_line_name = line_name
                        best_line_weight = line_weight

            if best_element is None:
                return None

            return best_element, best_line_name, best_line_weight, best_distance

        if elements is not None:
            search_elements = builtins.set(elements)
        else:
            search_elements = None

        for peak_idx, height, peak_energy, snr in display_peaks:
            best_match_info = _best_line_match(peak_energy, search_elements, requested_edge_filters)
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

        detected_elements = builtins.set()
        detected_sample_peaks = {}
        element_confidence = {}
        element_stats = {}
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
            if element_name not in element_stats:
                element_stats[element_name] = {
                    "raw_conf": 0.0,
                    "shells": builtins.set(),
                    "lines": builtins.set(),
                    "strong_matches": 0,
                    "match_count": 0,
                    "best_match_conf": 0.0,
                    "best_match_energy": 0.0,
                    "best_match_distance": float("inf"),
                    "best_match_weight": 0.0,
                    "best_match_shell": "?",
                }

            element_stats[element_name]["raw_conf"] += float(match_confidence)
            element_stats[element_name]["shells"].add(shell)
            element_stats[element_name]["lines"].add(str(line_name))
            element_stats[element_name]["match_count"] += 1
            if snr > snr_threshold_for_sample and distance < distance_threshold_for_sample:
                element_stats[element_name]["strong_matches"] += 1

            if float(match_confidence) > float(element_stats[element_name]["best_match_conf"]):
                element_stats[element_name]["best_match_conf"] = float(match_confidence)
                element_stats[element_name]["best_match_energy"] = float(peak_energy)
                element_stats[element_name]["best_match_distance"] = float(distance)
                element_stats[element_name]["best_match_weight"] = float(line_weight)
                element_stats[element_name]["best_match_shell"] = shell

        for element_name, stats in element_stats.items():
            valid_shells = {shell for shell in stats["shells"] if shell in {"K", "L", "M"}}
            num_shells = len(valid_shells)
            num_lines = len(stats["lines"])
            has_major_shell = len(valid_shells.intersection({"K", "L"})) > 0

            shell_bonus = 1.0 + 0.45 * max(0, num_shells - 1)
            line_bonus = 1.0 + 0.15 * max(0, min(num_lines, 4) - 1)
            strong_bonus = 1.0 + 0.20 * stats["strong_matches"]
            major_bonus = 1.15 if has_major_shell else 1.0

            confidence = stats["raw_conf"] * shell_bonus * line_bonus * strong_bonus * major_bonus
            element_confidence[element_name] = float(confidence)

        if element_confidence:
            conf_values = np.array(list(element_confidence.values()), dtype=float)
            confidence_cutoff = max(np.percentile(conf_values, 45), 0.30 * float(conf_values.max()))

            for element_name, confidence in element_confidence.items():
                stats = element_stats[element_name]
                valid_shells = {shell for shell in stats["shells"] if shell in {"K", "L", "M"}}
                has_major_shell = len(valid_shells.intersection({"K", "L"})) > 0
                is_supported = confidence >= confidence_cutoff
                is_near_cutoff_but_consistent = (
                    confidence >= 0.75 * confidence_cutoff
                    and stats["match_count"] >= 2
                    and has_major_shell
                )
                is_high_energy_singleton_anchor = (
                    stats["match_count"] == 1
                    and stats["best_match_energy"] >= 6.0
                    and stats["best_match_weight"] >= 0.8
                    and stats["best_match_distance"] <= 0.35 * tolerance
                    and confidence >= 0.45 * confidence_cutoff
                )

                if is_supported or is_near_cutoff_but_consistent or is_high_energy_singleton_anchor:
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

        def _best_supported_line_match_with_prior(
            peak_energy, snr, allowed_elements, edge_filters=None
        ):
            if not all_info or not allowed_elements:
                return None

            best_tuple = None
            best_score = -float("inf")
            denom = max(float(max_detected_conf), 1e-9)

            for element_name, lines in all_info.items():
                if element_name not in allowed_elements:
                    continue

                prior = float(element_confidence.get(element_name, 0.0)) / denom
                prior_factor = 1.0 + 0.5 * prior

                for line_name, line_info in lines.items():
                    if not _line_allowed_for_element(element_name, line_name, edge_filters):
                        continue
                    line_energy = line_info["energy (keV)"]
                    line_weight = line_info.get("weight", 0.5)
                    distance = abs(peak_energy - line_energy)
                    shell = _line_shell(line_name)

                    is_m_line = "M" in line_name and not ("Ma" in line_name or "Mb" in line_name)
                    effective_tolerance = tolerance * 0.5 if is_m_line else tolerance

                    if line_weight <= 0.3 or distance > effective_tolerance:
                        continue

                    local_conf = _peak_confidence(snr, line_weight, distance)
                    anchor_boost = 1.0
                    if element_name in anchor_elements and shell == "M" and peak_energy <= 3.0:
                        anchor_boost = 2.2
                    elif element_name in anchor_elements and shell in {"K", "L"}:
                        anchor_boost = 1.15

                    score = local_conf * prior_factor * anchor_boost

                    if score > best_score:
                        best_score = score
                        best_tuple = (element_name, line_name, line_weight, distance)

            return best_tuple

        for peak_idx, height, peak_energy, snr in display_peaks:
            match = raw_match_by_idx.get(int(peak_idx))

            if detected_elements:
                alt_match_info = _best_supported_line_match_with_prior(
                    peak_energy, snr, detected_elements, requested_edge_filters
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
            str(element_name) for element_name in detected_elements if str(element_name) in matched_elements
        }
        if ignored_elements:
            detected_elements = {
                str(element_name)
                for element_name in detected_elements
                if str(element_name) not in ignored_elements
            }

        refined_match_by_idx = {int(match[0]): match for match in peak_matches}

        displayed_peak_count = len(display_peaks)
        total_over_snr_peak_count = len(significant_peaks)

        candidate_elements = sorted(
            str(element_name)
            for element_name in builtins.set(element_stats.keys())
            if str(element_name) not in detected_elements
        )

        def _format_saved_model_elements(edge_filters):
            if edge_filters is None:
                return "None"

            formatted = []
            for element_name in sorted(str(name) for name in edge_filters.keys()):
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
                stats = element_stats.get(str(element_name), {})
                lines = stats.get("lines", builtins.set())
                line_names = sorted(str(line_name) for line_name in lines)
                if len(line_names) > 0:
                    formatted.append(f"{element_name} [{', '.join(line_names)}]")
                else:
                    formatted.append(str(element_name))
            return ", ".join(formatted)

        model_elements_header = "Saved Model Elements (Plotted):\n" if using_saved_model_elements else "Saved Model Elements (Not Plotted When Elements Specified):\n"
        print(f"\n{model_elements_header} {_format_saved_model_elements(saved_model_edge_filters)}")

        if detected_elements:
            print(f"\nAutodetected: {_format_elements_with_lines(detected_elements)}")
        else:
            print("\nAutodetected: None")
        if candidate_elements:
            print(f"Possible: {_format_elements_with_lines(candidate_elements)}")
        else:
            print("Possible: None")

        # Stable per-element colors (same element color across K/L/M lines)
        elements_for_color = builtins.set(detected_elements)
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
            element: color_palette[i]
            for i, element in enumerate(sorted_elements_for_colors)
        }

        table_rows = []
        matched_row_count = 0
        for peak_idx, height, peak_energy, snr in display_peaks:
            match = refined_match_by_idx.get(int(peak_idx))
            if match is not None:
                table_rows.append((peak_energy, height, snr, str(match[5])))
                matched_row_count += 1
            else:
                row_label = "Unmatched" if search_elements is not None else "Unknown"
                table_rows.append((peak_energy, height, snr, row_label))

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
            if search_elements is None or not all_info:
                return None

            best_element = None
            best_distance = float("inf")
            for element_name in search_elements:
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
            is_sample = detected_sample_peaks.get(peak_energy, False)
            match = refined_match_by_idx.get(int(peak_idx))
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
        dotted_reference_rows = []
        if search_elements is not None:
            energy_min = float(np.min(E))
            energy_max = float(np.max(E))
            reference_label_counts = {}
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

            y_top = float(np.nanmax(spec)) if len(spec) > 0 else 1.0
            y_top = max(y_top, 1.0)

            for element_name in sorted(search_elements):
                element_key = str(element_name)
                lines_info = all_info.get(element_key, {}) if all_info is not None else {}
                if not isinstance(lines_info, dict) or len(lines_info) == 0:
                    continue

                candidate_lines = []
                for line_name, line_info in lines_info.items():
                    if not _line_allowed_for_element(element_key, line_name, requested_edge_filters):
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
                    filtered_lines = sorted(candidate_lines, key=lambda item: item[2], reverse=True)[:1]
                else:
                    filtered_lines = sorted(filtered_lines, key=lambda item: item[2], reverse=True)[:6]

                for line_name, line_energy, line_weight in filtered_lines:
                    matched_energies = existing_matches_by_element.get(element_key, [])
                    if any(abs(line_energy - matched_energy) <= max(0.05, 0.5 * tolerance) for matched_energy in matched_energies):
                        continue

                    line_color = element_color_map.get(element_key, "black")
                    ax_spec.axvline(
                        line_energy,
                        color=line_color,
                        linestyle="--",
                        alpha=0.3,
                        linewidth=1.2,
                    )
                    dotted_reference_rows.append((element_key, str(line_name), float(line_energy)))
                    label_index = reference_label_counts.get(element_key, 0)
                    reference_label_counts[element_key] = label_index + 1
                    y_label = y_top * (0.95 - 0.05 * (label_index % 3))
                    ax_spec.text(
                        line_energy,
                        y_label,
                        f"{element_key} {line_name}",
                        ha="center",
                        va="bottom",
                        fontsize=8,
                        color=line_color,
                        rotation=90,
                        alpha=0.8,
                    )

        if detected_elements:
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
                if element_name in detected_elements and detected_sample_peaks.get(peak_energy, False):
                    line_name = match_str.split()[-1] if match_str else ""
                    label_text = f"{element_name} {line_name}" if line_name else element_name
                    color = element_color_map.get(element_name, "black")
                    labels_to_plot.append((peak_energy, label_text, color, height))

            labels_to_plot.sort(key=lambda item: item[0])

            if show_text:
                # Merge labels that are too close in energy into a single line of text
                # to avoid unreadable overlap.
                overlap_threshold = max(0.10, 0.7 * float(tolerance))
                same_label_overlap_threshold = max(0.16, 1.1 * float(tolerance))
                same_color_overlap_threshold = max(0.22, 1.6 * float(tolerance))
                grouped_labels = []
                current_group = []

                def _same_color(c1, c2):
                    try:
                        return np.allclose(np.asarray(c1), np.asarray(c2))
                    except Exception:
                        return str(c1) == str(c2)

                for label in labels_to_plot:
                    if not current_group:
                        current_group.append(label)
                        continue

                    prev_energy = current_group[-1][0]
                    prev_text = current_group[-1][1]
                    prev_color = current_group[-1][2]
                    energy_delta = abs(label[0] - prev_energy)

                    should_group = energy_delta <= overlap_threshold
                    if not should_group and label[1] == prev_text:
                        should_group = energy_delta <= same_label_overlap_threshold
                    if not should_group and _same_color(label[2], prev_color):
                        should_group = energy_delta <= same_color_overlap_threshold

                    if should_group:
                        current_group.append(label)
                    else:
                        grouped_labels.append(current_group)
                        current_group = [label]

                if current_group:
                    grouped_labels.append(current_group)

                label_vertical_offset = max(0.03 * y_scale, 0.08)
                grouped_bucket_step = max(0.02 * y_scale, 0.05)

                for group in grouped_labels:
                    if len(group) == 1:
                        peak_energy, label_text, color, height = group[0]
                        y_pos = height + label_vertical_offset
                        ax_spec.text(
                            peak_energy,
                            y_pos,
                            label_text,
                            ha="center",
                            va="bottom",
                            fontsize=10,
                            color=color,
                            weight="bold",
                            rotation=90,
                        )
                    else:
                        x_pos = float(np.mean([item[0] for item in group]))
                        y_pos = max(item[3] for item in group) + label_vertical_offset
                        merged_text = ", ".join(item[1] for item in group)
                        first_color = group[0][2]
                        all_same_color = all(
                            _same_color(item[2], first_color) for item in group
                        )
                        if all_same_color:
                            ax_spec.text(
                                x_pos,
                                y_pos,
                                merged_text,
                                ha="center",
                                va="bottom",
                                fontsize=9,
                                color=group[0][2],
                                weight="bold",
                                rotation=90,
                            )
                        else:
                            # Keep grouped behavior, but color by respective line colors.
                            # Build one comma-list per color and stack them tightly.
                            color_buckets = []
                            for _, label_text, label_color, _ in group:
                                matched_bucket = None
                                for bucket in color_buckets:
                                    if _same_color(bucket["color"], label_color):
                                        matched_bucket = bucket
                                        break

                                if matched_bucket is None:
                                    color_buckets.append(
                                        {"color": label_color, "labels": [label_text]}
                                    )
                                else:
                                    matched_bucket["labels"].append(label_text)

                            for bucket_index, bucket in enumerate(color_buckets):
                                bucket_text = ", ".join(bucket["labels"])
                                y_offset = bucket_index * grouped_bucket_step
                                ax_spec.text(
                                    x_pos,
                                    y_pos + y_offset,
                                    bucket_text,
                                    ha="center",
                                    va="bottom",
                                    fontsize=9,
                                    color=bucket["color"],
                                    weight="bold",
                                    rotation=90,
                                )

        fig.tight_layout()
        plt.show()

        print(f"{'Energy (keV)':<12} {'Intensity':<12} {'SNR':<8} {'Best Match':<25}")
        print("-" * 60)
        for peak_energy, height, snr, best_match in sorted_table_rows:
            print(f"{peak_energy:<12.3f} {height:<12.1f} {snr:<8.1f} {best_match:<25}")
        if dotted_reference_rows:
            print("-" * 60)
            for element_key, line_name, line_energy in sorted(
                dotted_reference_rows, key=lambda item: item[2]
            ):
                print(
                    f"{line_energy:<12.3f} {'-':<12} {'-':<8} "
                    f"{(element_key + ' ' + line_name + ' (ref)'):<25}"
                )
        print("-" * 60)
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

        background = PolynomialBackground(energy_axis, degree=polynomial_background_degree)
        peaks = GaussianPeaks(energy_axis, peak_width=peak_width, elements_to_fit=elements_to_fit)
        model = EDSModel(peaks, background, energy_axis=energy_axis)
        model = model.to(device=energy_axis.device, dtype=energy_axis.dtype)
        if len(model.peak_model.element_names) == 0:
            raise ValueError("No elements found in the selected energy range/elements_to_fit.")

        with torch.no_grad():
            model.peak_model.concentrations.fill_(1.0)

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
                max_iter=1,
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
        return self.fit_spectrum_pytorch(
            energy_range=energy_range,
            elements_to_fit=elements_to_fit,
            peak_width=peak_width,
            num_iters=num_iters,
            lr=lr,
            polynomial_background_degree=polynomial_background_degree,
            optimizer=optimizer,
            loss="mse",
            fit_mean_only=True,
            show_plot=True,
            device=device,
        )

    def fit_spectrum_pytorch(
        self,
        energy_range=None,
        elements_to_fit=None,
        peak_width=0.1,
        num_iters=300,
        num_iters_global=200,
        lr=None,
        polynomial_background_degree=3,
        optimizer=None,
        loss=None,
        optimizer_global=None,
        optimizer_local=None,
        loss_global=None,
        loss_local=None,
        freeze_peak_width=True,
        spatial_lambda=0.0,
        min_total_counts=0.0,
        verbose=True,
        fit_mean_only=False,
        show_plot=True,
        lr_global=None,
        lr_local=None,
        device=None,
        constrain_background=True,
    ):
        """Fit EDS spectra with one entrypoint for mean-only or full-cube fitting.

        Parameters
        ----------
        energy_range : list[float] | tuple[float, float] | None
            Energy range [emin, emax] to include in the fit.
        elements_to_fit : list[str] | None
            Element symbols to fit. If None, all available elements in range are used.
        peak_width : float, optional
            Initial peak FWHM in keV.
        num_iters : int, optional
            Number of optimization iterations for per-pixel fitting.
        num_iters_global : int, optional
            Number of mean-spectrum iterations used to initialize local fitting (3D mode).
        lr : float, optional
            Backward-compatible shared learning rate fallback. Used for global/local
            fitting only when lr_global/lr_local are not provided.
        lr_global : float, optional
            Learning rate for the global mean-spectrum stage. In mean-only mode, this
            is the learning rate used for that fit.
        lr_local : float, optional
            Learning rate for the position-by-position stage (3D mode).
        polynomial_background_degree : int, optional
            Degree of per-pixel polynomial background.
        optimizer : str | None, optional
            Backward-compatible optimizer selector. In mean-only mode it controls
            the global stage. In 3D mode it controls the local stage.
        optimizer_global : str | None, optional
            Global/mean-stage optimizer, "adam" or "lbfgs". In 3D mode, defaults
            to "lbfgs" unless explicitly set.
        optimizer_local : str | None, optional
            Local position-by-position optimizer, "adam" or "lbfgs". In 3D mode,
            defaults to optimizer if provided, otherwise "lbfgs".
        loss : str | None, optional
            Backward-compatible shared data term, "poisson" or "mse". If provided,
            it applies to both stages unless stage-specific losses are set.
        loss_global : str | None, optional
            Global/mean-stage data term, "poisson" or "mse".
        loss_local : str | None, optional
            Local position-by-position data term, "poisson" or "mse".
        freeze_peak_width : bool, optional
            If True, lock peak widths after global fit (3D mode).
        spatial_lambda : float, optional
            Weight for spatial smoothness on abundance maps (3D mode).
        min_total_counts : float, optional
            Ignore pixels with summed counts below this threshold in data loss (3D mode).
        verbose : bool, optional
            Print progress updates.
        fit_mean_only : bool, optional
            If True, fit only the summed spectrum over (x, y).
        show_plot : bool, optional
            Plot fit diagnostics in mean-only mode.
        device : str | torch.device | None, optional
            Compute device to run fitting on. If None, uses CUDA when available,
            otherwise CPU.
        constrain_background : bool, optional
            If True (3D mode), regularize local backgrounds using the global fit as
            a prior and soft physical constraints with built-in weights.

        Returns
        -------
        dict
            Mean-only mode keys include concentrations, fit, and diagnostics.
            3D mode keys include abundance maps and fit diagnostics.
        """

        def _normalize_choice(name, param_name, allowed_values):
            if name is None:
                return None
            name_norm = name.lower()
            if name_norm not in allowed_values:
                allowed_display = "', '".join(sorted(allowed_values))
                raise ValueError(f"{param_name} must be '{allowed_display}'")
            return name_norm

        def _resolve_stage_settings(
            fit_mean_only_mode,
            optimizer_default_global,
            loss_default_global,
            loss_default_local,
        ):
            if fit_mean_only_mode:
                return (
                    optimizer_global_name or optimizer_name or optimizer_default_global,
                    None,
                    loss_global_name or loss_name or loss_default_global,
                    None,
                )
            # Preserve historical behavior: global stage defaults to LBFGS.
            return (
                optimizer_global_name or optimizer_default_global,
                optimizer_local_name or optimizer_name or "lbfgs",
                loss_global_name or loss_name or loss_default_global,
                loss_local_name or loss_name or loss_default_local,
            )

        optimizer_name = _normalize_choice(optimizer, "optimizer", {"adam", "lbfgs"})
        optimizer_global_name = _normalize_choice(
            optimizer_global, "optimizer_global", {"adam", "lbfgs"}
        )
        optimizer_local_name = _normalize_choice(
            optimizer_local, "optimizer_local", {"adam", "lbfgs"}
        )

        loss_name = _normalize_choice(loss, "loss", {"poisson", "mse"})
        loss_global_name = _normalize_choice(loss_global, "loss_global", {"poisson", "mse"})
        loss_local_name = _normalize_choice(loss_local, "loss_local", {"poisson", "mse"})

        (
            effective_optimizer_global,
            effective_optimizer_local,
            effective_loss_global,
            effective_loss_local,
        ) = _resolve_stage_settings(
            fit_mean_only_mode=fit_mean_only,
            optimizer_default_global="lbfgs",
            loss_default_global="mse" if fit_mean_only else "poisson",
            loss_default_local="poisson",
        )

        if spatial_lambda < 0:
            raise ValueError("spatial_lambda must be >= 0")

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA device requested but torch.cuda.is_available() is False.")

        effective_lr_global = lr if lr_global is None else lr_global
        effective_lr_local = lr if lr_local is None else lr_local

        energy_axis_np = np.arange(self.shape[0]) * self.sampling[0] + self.origin[0]
        energy_axis = torch.tensor(energy_axis_np, dtype=torch.float32, device=device)
        spectra = torch.tensor(self.array, dtype=torch.float32, device=device)  # (E, Y, X)

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
                concs = (
                    nn.functional.softplus(model.peak_model.concentrations).detach().cpu().numpy()
                )
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

        # Stage 1: global mean-spectrum fit to initialize per-pixel parameters.
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

        with torch.no_grad():
            # If the global stage fit a normalized target, convert amplitude-like
            # parameters back to raw-count scale for local initialization.
            global_conc = (
                nn.functional.softplus(global_model.peak_model.concentrations).detach()
                * global_scale
            )
            global_bg_coeffs = global_model.background_model.coeffs.detach() * global_scale
            if global_bg_coeffs.numel() > 0:
                global_bg_coeffs = global_bg_coeffs.clone()
                global_bg_coeffs[0] = global_bg_coeffs[0] + global_offset
            global_peak_width_params = global_model.peak_model.peak_width_by_peak.detach().clone()

        # Stage 2: vectorized per-pixel fit with shared peak shapes.
        n_elements = len(global_model.peak_model.element_names)
        peak_energies = global_model.peak_model.peak_energies
        peak_weights = global_model.peak_model.peak_weights
        peak_element_indices = global_model.peak_model.peak_element_indices
        energy_step = float(global_model.peak_model.energy_step)

        background_basis = polynomial_energy_basis(
            energy_axis, degree=polynomial_background_degree
        )

        mean_total = torch.clamp(mean_spectrum.sum(), min=1e-8)
        pixel_scales = (total_counts / mean_total).unsqueeze(1)
        # Avoid near-zero concentration initialization that can cause vanishing
        # softplus gradients in local optimization (especially on normalized data).
        conc_init = torch.clamp(
            global_conc.unsqueeze(0) * pixel_scales,
            min=1e-3,
        )
        # Small random perturbation helps break symmetry across pixels.
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
            local_opt = torch.optim.Adam(trainable_params, lr=local_lr)
        else:
            local_opt = torch.optim.LBFGS(
                trainable_params,
                lr=local_lr,
                max_iter=1,
                line_search_fn="strong_wolfe",
                history_size=10,
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
            peaks_pred = conc @ basis.t()  # (P, E)
            bg_raw = bg_coeffs @ background_basis  # (P, E)
            # Keep local background parameterization consistent with global initialization.
            bg_pred = bg_raw
            predicted = torch.clamp(peaks_pred + bg_pred, min=1e-8, max=1e8)
            return predicted, conc, bg_pred

        def _prepare_local_loss_inputs(pred_local):
            pred_eval = pred_local[valid_pixel_mask]
            target_eval = spectra_flat[valid_pixel_mask]
            local_scale = torch.clamp(global_scale, min=1e-8)
            pred_eval = pred_eval / local_scale
            target_eval = target_eval / local_scale
            return pred_eval, target_eval

        def _background_regularization(bg_local):
            if not constrain_background:
                return bg_local.new_tensor(0.0)

            prior_lambda = 0.1
            nonneg_lambda = 0.5
            monotonic_lambda = 0.05
            smoothness_lambda = 0.01

            reg_loss = bg_local.new_tensor(0.0)
            local_scale = torch.clamp(global_scale, min=1e-8)

            if prior_lambda > 0:
                coeff_init_eval = bg_coeffs_init[valid_pixel_mask]
                coeff_eval = bg_coeffs[valid_pixel_mask]
                coeff_scale = torch.clamp(coeff_init_eval.abs().mean(), min=1e-8)
                reg_prior = ((coeff_eval - coeff_init_eval) / coeff_scale).pow(2).mean()
                reg_loss = reg_loss + prior_lambda * reg_prior

            bg_eval = bg_local[valid_pixel_mask] / local_scale

            if nonneg_lambda > 0:
                reg_nonneg = torch.relu(-bg_eval).pow(2).mean()
                reg_loss = reg_loss + nonneg_lambda * reg_nonneg

            if monotonic_lambda > 0 and bg_eval.shape[1] > 1:
                slope = bg_eval[:, 1:] - bg_eval[:, :-1]
                reg_monotonic = torch.relu(slope).pow(2).mean()
                reg_loss = reg_loss + monotonic_lambda * reg_monotonic

            if smoothness_lambda > 0 and bg_eval.shape[1] > 2:
                curvature = bg_eval[:, 2:] - 2.0 * bg_eval[:, 1:-1] + bg_eval[:, :-2]
                reg_smooth = curvature.pow(2).mean()
                reg_loss = reg_loss + smoothness_lambda * reg_smooth

            return reg_loss

        def _local_loss(pred_local, conc_local, bg_local):
            pred_eval, target_eval = _prepare_local_loss_inputs(pred_local)

            loss_data = eds_data_loss(
                pred_eval,
                target_eval,
                loss=effective_loss_local,
            )
            loss_total = loss_data + _background_regularization(bg_local)

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
                    pred_local, conc_local, bg_local = _forward_model()
                    loss_total = _local_loss(pred_local, conc_local, bg_local)
                    loss_total.backward()
                    return loss_total

                loss_value = local_opt.step(_local_closure)
                if not torch.is_tensor(loss_value):
                    with torch.no_grad():
                        pred_local, conc_local, bg_local = _forward_model()
                        loss_value = _local_loss(pred_local, conc_local, bg_local)
            else:
                local_opt.zero_grad()
                pred_local, conc_local, bg_local = _forward_model()
                loss_value = _local_loss(pred_local, conc_local, bg_local)
                loss_value.backward()
                local_opt.step()

            loss_history.append(float(loss_value.detach().cpu().item()))
            if verbose and ((i + 1) % max(1, num_iters // 10) == 0 or i == 0):
                print(f"iter {i + 1:4d}/{num_iters}: loss={loss_history[-1]:.6g}")

        with torch.no_grad():
            pred_final, conc_final, bg_final = _forward_model()

            # Keep global/local/data comparison on equal footing by averaging
            # over the same valid-pixel mask used in the global stage.
            mean_input_spectrum = spectra_flat[valid_pixel_mask].mean(dim=0).cpu().numpy()
            mean_fitted_spectrum = pred_final[valid_pixel_mask].mean(dim=0).cpu().numpy()
            mean_background_spectrum = bg_final[valid_pixel_mask].mean(dim=0).cpu().numpy()

            # Also provide all-pixel aggregates for diagnostics.
            mean_input_spectrum_all = spectra_flat.mean(dim=0).cpu().numpy()
            mean_fitted_spectrum_all = pred_final.mean(dim=0).cpu().numpy()
            mean_background_spectrum_all = bg_final.mean(dim=0).cpu().numpy()

            abundance_maps = conc_final.view(n_y, n_x, n_elements).permute(2, 0, 1).cpu().numpy()
            peak_widths = nn.functional.softplus(peak_width_params).detach().cpu().numpy()
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

            map_titles = [f"{name}" for name in global_model.peak_model.element_names]
            show_2d(list(abundance_maps), title=map_titles)

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
