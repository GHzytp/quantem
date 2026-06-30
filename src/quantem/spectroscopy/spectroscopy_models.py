from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from quantem.spectroscopy.utils import load_xray_lines_database


def inverse_softplus(x: torch.Tensor, min_value: float = 1e-8) -> torch.Tensor:
    """Numerically stable inverse of softplus for positive initialization values."""
    x = torch.clamp(x, min=min_value)
    # For large x, log(expm1(x)) can overflow in float32. Use a stable branch.
    return torch.where(
        x > 20.0,
        x + torch.log1p(-torch.exp(-x)),
        torch.log(torch.expm1(x)),
    )


def xeds_data_loss(
    predicted: torch.Tensor, target: torch.Tensor, loss: str = "poisson", min_value: float = 1e-8
) -> torch.Tensor:
    """Compute XEDS fit loss with clamped positive predictions."""
    pred_safe = torch.nan_to_num(predicted, nan=min_value, posinf=1e8, neginf=min_value)
    pred_safe = torch.clamp(pred_safe, min=min_value, max=1e8)
    if loss == "poisson":
        target_safe = torch.nan_to_num(target, nan=0.0, posinf=1e8, neginf=0.0)
        target_safe = torch.clamp(target_safe, min=0.0, max=1e8)
        if hasattr(torch, "xlogy"):
            log_term = torch.xlogy(target_safe, pred_safe)
        elif hasattr(torch.special, "xlogy"):
            log_term = torch.special.xlogy(target_safe, pred_safe)
        else:
            log_term = target_safe * torch.log(pred_safe)
            log_term = torch.nan_to_num(log_term, nan=0.0, posinf=1e8, neginf=-1e8)
        loss_terms = pred_safe - log_term
        return torch.mean(torch.nan_to_num(loss_terms, nan=1e8, posinf=1e8, neginf=-1e8))
    if loss == "mse":
        target_safe = torch.nan_to_num(target, nan=0.0, posinf=1e8, neginf=-1e8)
        return nn.functional.mse_loss(pred_safe, target_safe)
    raise ValueError("loss must be 'poisson' or 'mse'")


def polynomial_energy_basis(energy_axis: torch.Tensor, degree: int) -> torch.Tensor:
    """Return polynomial basis in normalized energy coordinates."""
    energy_norm = (energy_axis - energy_axis.min()) / (
        energy_axis.max() - energy_axis.min() + 1e-12
    )
    return torch.stack([energy_norm**i for i in range(degree + 1)], dim=0)


def build_element_basis(
    energy_axis: torch.Tensor,
    peak_energies: torch.Tensor,
    peak_weights: torch.Tensor,
    peak_element_indices: torch.Tensor,
    peak_width_by_peak: torch.Tensor,
    n_elements: int,
    energy_step: float,
) -> torch.Tensor:
    """Build matrix mapping per-element concentrations to spectral intensity."""
    fwhm = nn.functional.softplus(peak_width_by_peak)
    sigma = (fwhm / 2.355).unsqueeze(1)
    centers = peak_energies.unsqueeze(1)
    energies = energy_axis.unsqueeze(0)
    all_peaks = torch.exp(-0.5 * ((energies - centers) / sigma) ** 2)
    sqrt_2pi = torch.sqrt(torch.tensor(2 * np.pi, dtype=all_peaks.dtype, device=all_peaks.device))
    all_peaks = all_peaks * energy_step / (sqrt_2pi * sigma)
    weighted_peaks = all_peaks * peak_weights.unsqueeze(1)

    basis = torch.zeros(
        (n_elements, energy_axis.shape[0]),
        dtype=weighted_peaks.dtype,
        device=weighted_peaks.device,
    )
    basis.index_add_(0, peak_element_indices.to(weighted_peaks.device), weighted_peaks)
    return basis.t()


def abundance_smoothness_l2(abundance_maps: torch.Tensor) -> torch.Tensor:
    """Spatial L2 smoothness for abundance maps shaped (n_elements, y, x)."""
    if abundance_maps.ndim != 3:
        raise ValueError("abundance_maps must have shape (n_elements, y, x)")

    loss = abundance_maps.new_tensor(0.0)
    if abundance_maps.shape[2] > 1:
        dx = abundance_maps[:, :, 1:] - abundance_maps[:, :, :-1]
        loss = loss + dx.pow(2).mean()
    if abundance_maps.shape[1] > 1:
        dy = abundance_maps[:, 1:, :] - abundance_maps[:, :-1, :]
        loss = loss + dy.pow(2).mean()
    return loss


class XEDSModel(nn.Module):
    """XEDS spectrum model = peaks + optional background."""

    def __init__(self, peak_model, background_model=None):
        super().__init__()
        self.peak_model = peak_model
        self.background_model = background_model

    def forward(self):
        spectrum = self.peak_model()
        if self.background_model is not None:
            spectrum = spectrum + self.background_model()
        return spectrum


class GaussianPeaks(nn.Module):
    """Generate Gaussian peak spectra from X-ray line data."""

    def __init__(
        self,
        energy_axis,
        peak_width,
        elements_to_fit=None,
    ):
        super().__init__()

        current_dir = Path(__file__).parent
        data = load_xray_lines_database(current_dir / "x_ray_lines.csv")

        energy_axis_tensor = (
            energy_axis.float()
            if torch.is_tensor(energy_axis)
            else torch.tensor(energy_axis, dtype=torch.float32)
        )
        self.register_buffer("energy_axis", energy_axis_tensor)
        self.energy_min = self.energy_axis.min().item()
        self.energy_max = self.energy_axis.max().item()
        self.energy_step = (self.energy_axis[1] - self.energy_axis[0]).item()

        all_element_data = {}
        for elem, lines in data.items():
            if len(lines) > 0:
                energies = []
                weights = []
                line_names = []

                for line_name, line_data in lines.items():
                    energy = line_data["energy (keV)"]
                    if self.energy_min - 0.5 <= energy <= self.energy_max + 0.5:
                        energies.append(energy)
                        weights.append(line_data["weight"])
                        line_names.append(line_name)

                if len(energies) > 0:
                    all_element_data[elem] = {
                        "energies": energies,
                        "weights": weights,
                        "line_names": line_names,
                    }

        if elements_to_fit is not None:
            self.element_data = {}
            for elem in elements_to_fit:
                if elem in all_element_data:
                    self.element_data[elem] = all_element_data[elem]
        else:
            self.element_data = all_element_data

        self.element_names = list(self.element_data.keys())
        n_elements = len(self.element_names)

        all_peak_energies = []
        all_peak_weights = []
        all_peak_element_indices = []
        all_peak_shell_group_indices = []
        shell_group_lookup = {}
        shell_group_names = []
        shell_group_element_indices = []
        shell_group_shell_labels = []

        for elem_idx, elem in enumerate(self.element_names):
            energies = self.element_data[elem]["energies"]
            weights = np.asarray(self.element_data[elem]["weights"], dtype=np.float32)
            line_names = self.element_data[elem]["line_names"]

            shell_labels = [self._line_shell_label(line_name) for line_name in line_names]
            normalized_weights = np.zeros_like(weights)
            for shell_label in set(shell_labels):
                shell_indices = [i for i, label in enumerate(shell_labels) if label == shell_label]
                shell_weights = np.clip(weights[shell_indices], a_min=0.0, a_max=None)
                shell_sum = np.sum(shell_weights)
                if shell_sum <= 0.0:
                    shell_weights = np.ones(len(shell_indices), dtype=np.float32) / len(
                        shell_indices
                    )
                else:
                    shell_weights = shell_weights / shell_sum
                normalized_weights[shell_indices] = shell_weights
            weights_to_use = normalized_weights

            all_peak_energies.extend(energies)
            all_peak_weights.extend(weights_to_use)
            all_peak_element_indices.extend([elem_idx] * len(energies))
            for shell_label in shell_labels:
                key = (elem_idx, shell_label)
                if key not in shell_group_lookup:
                    shell_group_lookup[key] = len(shell_group_names)
                    shell_group_names.append(f"{elem} {shell_label}")
                    shell_group_element_indices.append(elem_idx)
                    shell_group_shell_labels.append(shell_label)
                all_peak_shell_group_indices.append(shell_group_lookup[key])

        self.register_buffer(
            "peak_energies",
            torch.tensor(
                all_peak_energies,
                dtype=self.energy_axis.dtype,
                device=self.energy_axis.device,
            ),
        )
        self.register_buffer(
            "peak_weights",
            torch.tensor(
                all_peak_weights,
                dtype=self.energy_axis.dtype,
                device=self.energy_axis.device,
            ),
        )
        self.register_buffer(
            "peak_element_indices",
            torch.tensor(
                all_peak_element_indices,
                dtype=torch.long,
                device=self.energy_axis.device,
            ),
        )
        self.register_buffer(
            "peak_shell_group_indices",
            torch.tensor(
                all_peak_shell_group_indices,
                dtype=torch.long,
                device=self.energy_axis.device,
            ),
        )
        self.register_buffer(
            "shell_group_element_indices",
            torch.tensor(
                shell_group_element_indices,
                dtype=torch.long,
                device=self.energy_axis.device,
            ),
        )
        self.shell_group_names = shell_group_names
        self.shell_group_shell_labels = shell_group_shell_labels

        self.n_peaks = len(all_peak_energies)
        init_fwhm = torch.tensor(
            peak_width,
            dtype=self.energy_axis.dtype,
            device=self.energy_axis.device,
        )
        self.peak_width_by_peak = nn.Parameter(
            inverse_softplus(init_fwhm)
            * torch.ones(
                self.n_peaks,
                dtype=self.energy_axis.dtype,
                device=self.energy_axis.device,
            )
        )

        n_shell_groups = len(shell_group_names)
        print(
            f"Fitting {n_elements} elements with {self.n_peaks} total peaks "
            f"across {n_shell_groups} edge groups"
        )

        concentration_size = len(shell_group_names)
        if concentration_size > 0:
            init_concentration = torch.ones(
                concentration_size,
                dtype=self.energy_axis.dtype,
                device=self.energy_axis.device,
            )
            concentration_init_logits = inverse_softplus(init_concentration)
        else:
            concentration_init_logits = torch.ones(
                concentration_size,
                dtype=self.energy_axis.dtype,
                device=self.energy_axis.device,
            )

        self.concentrations = nn.Parameter(concentration_init_logits)

    @staticmethod
    def _line_shell_label(line_name: str) -> str:
        text = str(line_name).strip()
        for char in text:
            if char.isalpha():
                return char.upper()
        return "Other"

    def forward(self):
        centers = self.peak_energies.unsqueeze(1)
        energies = self.energy_axis.unsqueeze(0)

        fwhm = nn.functional.softplus(self.peak_width_by_peak)
        sigma = (fwhm / 2.355).unsqueeze(1)

        all_peaks = torch.exp(-0.5 * ((energies - centers) / sigma) ** 2)

        sqrt_2pi = torch.sqrt(
            torch.tensor(
                2 * np.pi,
                dtype=all_peaks.dtype,
                device=all_peaks.device,
            )
        )
        all_peaks = all_peaks * self.energy_step / (sqrt_2pi * sigma)

        concentration_lookup = self.peak_shell_group_indices
        peak_concentrations = nn.functional.softplus(self.concentrations[concentration_lookup])
        weighted_peaks = all_peaks * (peak_concentrations * self.peak_weights).unsqueeze(1)

        spectrum = weighted_peaks.sum(dim=0)

        return spectrum


class PolynomialBackground(nn.Module):
    """Polynomial background model"""

    def __init__(self, energy_axis, degree=3):
        super().__init__()
        energy_axis_tensor = (
            energy_axis.float()
            if torch.is_tensor(energy_axis)
            else torch.tensor(energy_axis, dtype=torch.float32)
        )
        self.register_buffer("energy_axis", energy_axis_tensor)
        self.degree = degree

        energy_norm = (self.energy_axis - self.energy_axis.min()) / (
            self.energy_axis.max() - self.energy_axis.min()
        )
        self.register_buffer("energy_norm", energy_norm)

        self.coeffs = nn.Parameter(
            torch.randn(
                degree + 1,
                dtype=self.energy_axis.dtype,
                device=self.energy_axis.device,
            )
            * 0.1
        )

    def forward(self):
        background = torch.zeros_like(self.energy_axis)
        for i, coeff in enumerate(self.coeffs):
            background += coeff * (self.energy_norm**i)
        return background
