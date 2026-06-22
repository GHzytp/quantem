import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from scipy.signal import find_peaks, peak_prominences

from quantem.spectroscopy.spectroscopy_models import (
    GaussianPeaks,
    PolynomialBackground,
    XEDSModel,
    abundance_smoothness_l2,
    build_element_basis,
    inverse_softplus,
    polynomial_energy_basis,
    xeds_data_loss,
)


def peak_autoid(
    self,
    roi=None,
    roi_cal=None,
    energy_range=None,
    elements=None,
    ignore_elements=None,
    ignore_range=None,
    tolerance=0.15,
    threshold=None,
    noise_percentile=75,
    min_line_weight=0.0,
    mask=None,
    show_text=True,
    peaks=15,
    line=None,
    return_details=False,
):
    """Identify likely elements by matching XEDS spectrum peaks to known lines.

    This routine keeps the matching logic intentionally direct:
    calculate a mean spectrum, find local maxima above an optional intensity
    threshold, match each peak to database lines within an energy tolerance,
    and rank elements by the quality of those matches.

    Parameters
    ----------
    roi, roi_cal : sequence or None, optional
        Spatial region used to calculate the mean spectrum. See
        ``show_mean_spectrum`` for ROI formats.
    energy_range : sequence[float] or None, optional
        Energy range ``[emin, emax]`` in keV to analyze.
    elements : str or sequence[str] or None, optional
        Element or element-line selectors to search, such as ``"Fe"``,
        ``"Fe K"``, or ``["Cu", "Zn"]``. If omitted, all database elements
        are considered.
    ignore_elements : str or sequence[str] or None, optional
        Elements to exclude from matching.
    ignore_range : sequence[float] or None, optional
        Energy interval ``[emin, emax]`` in keV where detected peaks are
        ignored.
    tolerance : float, optional
        Maximum allowed energy difference in keV between a detected peak
        and a database line. This controls line matching, not peak finding.
    threshold : float, "mean", or None, optional
        Minimum mean-spectrum intensity required for a peak to be
        considered. Use ``"mean"`` to require peaks above the average
        spectrum intensity. If ``None``, no intensity threshold is applied.
    noise_percentile : float or None, optional
        Percentile intensity used as the SNR denominator. The default
        ``75`` uses the 75th percentile of the mean-spectrum intensity. If
        ``None``, the mean finite intensity is used.
    min_line_weight : float, optional
        Minimum database line weight required for a line to be considered.
    mask : ndarray or None, optional
        Boolean mask forwarded to ``calculate_mean_spectrum``.
    show_text : bool, optional
        If ``True``, label matched plotted peaks.
    peaks : int or None, optional
        Maximum number of peaks to plot and print in the table. Matching is
        still performed on all peaks that pass ``threshold``.
    line : float or sequence[float] or None, optional
        Reference energy line(s) in keV to draw as dashed black vertical
        markers.
    return_details : bool, optional
        If ``True``, return a dictionary with figure, axes, peaks, matches,
        alternatives, and element scores.

    Returns
    -------
    tuple or dict
        By default returns ``(fig, (ax_img, ax_spec))``. If
        ``return_details`` is ``True``, returns a details dictionary.
    """
    type(self)._ensure_element_info()
    all_info = type(self).element_info or {}
    ignored_elements = set(
        map(str, type(self)._normalize_specs(ignore_elements, allow_none=True) or [])
    )
    min_line_weight = max(float(min_line_weight), 0.0)

    edge_filters = type(self)._parse_element_selectors(
        elements, allow_none=True, param_name="elements"
    )
    search_elements = set(edge_filters) if edge_filters else None

    fig, (ax_img, ax_spec) = self.show_mean_spectrum(
        roi=roi,
        roi_cal=roi_cal,
        energy_range=energy_range,
        mask=mask,
        data_type="xeds",
        show=False,
    )
    spec = np.asarray(
        self.calculate_mean_spectrum(
            roi=roi,
            roi_cal=roi_cal,
            energy_range=energy_range,
            mask=mask,
        ),
        dtype=float,
    )
    energy_axis = np.asarray(self.energy_axis, dtype=float)

    if mask is not None:
        mask_arr = np.asarray(mask, dtype=bool)
        if mask_arr.shape != energy_axis.shape:
            raise ValueError(
                f"Mask shape {mask_arr.shape} does not match energy axis shape "
                f"{energy_axis.shape}."
            )
        energy_axis = energy_axis[mask_arr]

    if energy_range is not None:
        keep = (float(energy_range[0]) <= energy_axis) & (energy_axis <= float(energy_range[1]))
        energy_axis = energy_axis[keep]

    if spec.shape != energy_axis.shape:
        raise ValueError(
            "Energy axis length does not match mean spectrum length after filtering. "
            f"Got len(E)={len(energy_axis)} and len(spec)={len(spec)}."
        )

    def in_ignore_range(value):
        return (
            ignore_range is not None
            and len(ignore_range) == 2
            and float(ignore_range[0]) <= float(value) <= float(ignore_range[1])
        )

    def noise_level(values, percentile=75):
        values = np.asarray(values, dtype=float)
        values = values[np.isfinite(values)]
        if values.size == 0:
            return 1.0
        if percentile is None:
            noise = float(np.mean(values))
        else:
            percentile = float(percentile)
            if not 0 <= percentile <= 100:
                raise ValueError("noise_percentile must be between 0 and 100, or None")
            noise = float(np.percentile(values, percentile))
        return noise if np.isfinite(noise) and noise > 0 else 1.0

    def resolve_threshold(value):
        if value is None:
            return None
        if isinstance(value, str):
            if value.lower() != "mean":
                raise ValueError("threshold must be a number, 'mean', or None")
            finite = spec[np.isfinite(spec)]
            return float(np.mean(finite)) if finite.size else None
        threshold_value = float(value)
        if not np.isfinite(threshold_value):
            raise ValueError("threshold must be finite")
        return threshold_value

    noise = noise_level(spec, noise_percentile)
    threshold_value = resolve_threshold(threshold)
    peak_indices, _ = find_peaks(spec, height=threshold_value)
    prominences = (
        peak_prominences(spec, peak_indices)[0]
        if len(peak_indices)
        else np.asarray([], dtype=float)
    )
    peak_rows = []
    for idx, prominence in zip(peak_indices, prominences):
        energy = float(energy_axis[int(idx)])
        if in_ignore_range(energy):
            continue
        height = float(spec[int(idx)])
        snr = height / noise
        peak_rows.append((int(idx), height, energy, float(snr), float(prominence)))

    peak_rows.sort(key=lambda row: row[4], reverse=True)
    all_peaks = [(idx, height, energy, snr) for idx, height, energy, snr, _ in peak_rows]
    display_peaks = all_peaks if peaks is None else all_peaks[: max(int(peaks), 0)]

    def candidate_matches(peak_energy, snr, allowed_elements=None):
        candidates = []
        for element_name, lines in all_info.items():
            element_name = str(element_name)
            if element_name in ignored_elements:
                continue
            if allowed_elements is not None and element_name not in allowed_elements:
                continue
            for line_name, line_info in (lines or {}).items():
                if not type(self)._line_allowed_for_element(
                    element_name, str(line_name), edge_filters
                ):
                    continue
                try:
                    line_energy = float(
                        line_info["energy (keV)"]
                        if "energy (keV)" in line_info
                        else line_info["energy"]
                    )
                    line_weight = float(line_info.get("weight", 0.5))
                except (TypeError, ValueError, KeyError):
                    continue
                distance = abs(float(peak_energy) - line_energy)
                if line_weight < min_line_weight or distance > float(tolerance):
                    continue
                score = type(self)._peak_confidence(snr, line_weight, distance, float(tolerance))
                candidates.append(
                    {
                        "element": element_name,
                        "line": str(line_name),
                        "energy": line_energy,
                        "weight": line_weight,
                        "distance": distance,
                        "score": float(score),
                    }
                )
        candidates.sort(key=lambda item: item["score"], reverse=True)
        return candidates

    peak_matches = []
    alternatives_by_peak = {}
    for peak_idx, height, peak_energy, snr in all_peaks:
        matches = candidate_matches(peak_energy, snr, search_elements)
        alternatives_by_peak[int(peak_idx)] = matches[:3]
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

    element_confidence: dict[str, float] = {}
    for _, _, _, _, element, _, _, _, _, score in peak_matches:
        element_confidence[element] = element_confidence.get(element, 0.0) + float(score)

    detected_elements = set(element_confidence)
    match_by_idx = {int(match[0]): match for match in peak_matches}

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
    ]
    element_color_map = {
        element: palette[i % len(palette)]
        for i, element in enumerate(sorted(detected_elements or (search_elements or [])))
    }

    y_min = float(np.nanmin(spec)) if len(spec) else 0.0
    y_max = float(np.nanmax(spec)) if len(spec) else 1.0
    y_span = max(y_max - y_min, abs(y_max), 1.0)
    label_y = 0.96

    table_rows = []
    for peak_idx, height, peak_energy, snr in display_peaks:
        match = match_by_idx.get(int(peak_idx))
        if match is None:
            ax_spec.plot(
                [peak_energy],
                [y_min - 0.04 * y_span],
                marker="|",
                markersize=5,
                color="gray",
                linestyle="None",
            )
            table_rows.append((peak_energy, height, snr, "Unmatched", "-", "-"))
            continue

        element = str(match[4])
        line_name = str(match[7])
        color = element_color_map.get(element, "black")
        ax_spec.axvline(peak_energy, color=color, linestyle="-", alpha=0.55, linewidth=1.5)
        if show_text:
            ax_spec.text(
                peak_energy,
                label_y,
                f"{element} {line_name}",
                transform=ax_spec.get_xaxis_transform(),
                ha="center",
                va="top",
                rotation=90,
                fontsize=9,
                color=color,
                clip_on=True,
            )

        labels = [
            f"{m['element']} {m['line']} ({m['energy']:.3f})"
            for m in alternatives_by_peak[int(peak_idx)]
        ]
        labels = labels + ["-"] * (3 - len(labels))
        table_rows.append((peak_energy, height, snr, labels[0], labels[1], labels[2]))

    if line is not None:
        x_min, x_max = ax_spec.get_xlim()
        ref_energies = [line] if isinstance(line, (int, float)) else list(line)
        for ref_energy in ref_energies:
            try:
                ref_energy = float(ref_energy)
            except (TypeError, ValueError):
                continue
            if x_min <= ref_energy <= x_max:
                ax_spec.axvline(ref_energy, color="black", linestyle="--", linewidth=1.2, zorder=3)
        ax_spec.set_xlim(x_min, x_max)

    current_bottom, current_top = ax_spec.get_ylim()
    ax_spec.set_ylim(bottom=min(current_bottom, y_min - 0.10 * y_span), top=current_top)
    fig.tight_layout()
    plt.show()

    print(
        f"{'Energy (keV)':<12} {'Intensity':<12} {'SNR':<8} "
        f"{'Best Match':<24} {'Alt 2':<24} {'Alt 3':<24}"
    )
    print("-" * 112)
    for peak_energy, height, snr, best_match, alt_2, alt_3 in sorted(table_rows):
        print(
            f"{peak_energy:<12.3f} {height:<12.2f} {snr:<8.1f} "
            f"{best_match:<24} {alt_2:<24} {alt_3:<24}"
        )
    print("-" * 112)
    print(f"Matched {len(peak_matches)} peaks; displayed {len(display_peaks)} prominent peaks.\n")

    if return_details:
        return {
            "figure": fig,
            "axes": (ax_img, ax_spec),
            "detected_elements": sorted(detected_elements),
            "element_confidence": element_confidence,
            "display_peaks": display_peaks,
            "peak_matches": peak_matches,
            "peak_alternatives": alternatives_by_peak,
            "threshold": threshold_value,
            "noise": noise,
            "noise_percentile": noise_percentile,
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
    """Fit a single mean spectrum using the PyTorch XEDS model."""
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
    model = XEDSModel(peaks, background)
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
                loss = xeds_data_loss(predicted, target, loss=loss_name)
                loss.backward()
                return loss

            loss = optimizer_obj.step(closure)
            if not torch.is_tensor(loss):
                with torch.no_grad():
                    loss = xeds_data_loss(model(), target, loss=loss_name)
        else:
            optimizer_obj.zero_grad()
            predicted = model()
            loss = xeds_data_loss(predicted, target, loss=loss_name)
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
    """Fit the spatially-summed mean XEDS spectrum and display results.

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
        shell_element_indices = model.peak_model.shell_group_element_indices.detach().cpu().numpy()
        concs = np.zeros(len(model.peak_model.element_names), dtype=np.float32)
        np.add.at(concs, shell_element_indices, shell_concs)
        final_fwhm = (
            torch.nn.functional.softplus(model.peak_model.peak_width_by_peak)
            .detach()
            .cpu()
            .numpy()
        )
        background_fit = (
            (model.background_model().detach() * spectrum_scale + spectrum_offset).cpu().numpy()
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
    """Fit XEDS spectra using a PyTorch model.

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
            nn.functional.softplus(global_model.peak_model.concentrations).detach() * global_scale
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

    background_basis = polynomial_energy_basis(energy_axis, degree=polynomial_background_degree)

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

        loss_data = xeds_data_loss(
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
