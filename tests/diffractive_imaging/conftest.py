"""Shared synthetic 4D-STEM data for the direct-ptychography test modules.

Mirrors the simulation idiom in ``test_ptychography.py`` (white-noise phase object,
soft-aperture defocused probe, ``|FFT(obj_patch * probe)|**2``) but on a smaller grid and
with a probe small enough that parallax shifts stay well inside the scan.

Imported by ``test_direct_ptychography.py`` and ``test_shadow_montage.py`` via
``from .conftest import ...``, matching the pattern used by the tomography suite.
"""

import numpy as np
import pytest

from quantem.core.datastructures import Dataset4d
from quantem.core.utils.utils import electron_wavelength_angstrom

N = 32  # detector / object gridpoints
Q_MAX = 0.5  # inverse Angstroms
Q_PROBE = Q_MAX / 4  # inverse Angstroms -> BF disk radius of N/8 = 4 px
PROBE_ENERGY = 300e3  # eV
SCAN_STEP_SIZE = 1  # pixels

SAMPLING = 1 / Q_MAX / 2  # Angstroms
RECIPROCAL_SAMPLING = 2 * Q_MAX / N  # inverse Angstroms
SEMIANGLE_CUTOFF = Q_PROBE * electron_wavelength_angstrom(PROBE_ENERGY) * 1e3  # mrad
SCAN_SAMPLING = SAMPLING * SCAN_STEP_SIZE  # Angstroms

#: fixing the fitted origin keeps the center-of-mass step exact and deterministic
ORIGIN = (N // 2, N // 2)

DECONVOLUTION_KERNELS = ("ssb", "obf", "mf", "prlx", "icom")


def integer_shift_defocus(pixel_shift_per_k_step: int, upsampling_factor: int = 1) -> float:
    """``C10`` (in Angstrom) placing every BF pixel's parallax shift on an exact pixel.

    For pure defocus the aberration gradient is ``dchi/dk = 2*pi*wavelength*C10*k``, so the
    lateral shift is ``wavelength*C10*k`` Angstrom. Because ``k = m * reciprocal_sampling``
    exactly (``spatial_frequencies`` uses ``fftfreq(n, 1/(dk*n))``), choosing

        C10 = p * scan_sampling / (U * wavelength * dk)

    gives a shift of exactly ``p * m`` upsampled pixels for integer detector index ``m``.
    """
    wavelength = electron_wavelength_angstrom(PROBE_ENERGY)
    return (
        pixel_shift_per_k_step
        * SCAN_SAMPLING
        / (upsampling_factor * wavelength * RECIPROCAL_SAMPLING)
    )


def make_complex_obj(seed: int = 42) -> np.ndarray:
    """White-noise pure-phase object."""
    rng = np.random.default_rng(seed)
    arr = rng.random((N, N))
    arr -= arr.mean()
    return np.exp(1.0j * arr.astype(np.float32))


def band_limited_phase(seed: int = 42, cutoff: float = 2 * Q_PROBE) -> np.ndarray:
    """Ground-truth phase, low-passed to what the bright-field disk can actually transfer.

    Correlating a reconstruction against the raw white-noise phase caps out near 0.4 simply
    because most of its power sits above the aperture cutoff; band-limiting first makes
    "did this recover the object" a meaningful question.
    """
    phase = np.angle(make_complex_obj(seed))
    qx = qy = np.fft.fftfreq(N, SCAN_SAMPLING)
    q = np.hypot(qx[:, None], qy[None, :])
    limited = np.fft.ifft2(np.fft.fft2(phase) * (q <= cutoff)).real
    return limited - limited.mean()


def make_probe_array(defocus: float) -> np.ndarray:
    """Soft-aperture probe with ``C10 = defocus`` Angstrom."""
    qx = qy = np.fft.fftfreq(N, SAMPLING)
    q = np.sqrt(qx[:, None] ** 2 + qy[None, :] ** 2)

    aperture_fourier = np.sqrt(
        np.clip((Q_PROBE - q) / RECIPROCAL_SAMPLING + 0.5, 0, 1),
    )
    chi = q**2 * electron_wavelength_angstrom(PROBE_ENERGY) * np.pi * defocus
    probe_fourier = aperture_fourier * np.exp(-1j * chi)
    probe_fourier /= np.sqrt(np.sum(np.abs(probe_fourier) ** 2))
    return np.fft.ifft2(probe_fourier) * N


def scan_positions_px() -> np.ndarray:
    """``(N_pos, 2)`` raster positions in object pixels, "ij" (row, col) ordering."""
    n = N // SCAN_STEP_SIZE
    ii, jj = np.meshgrid(
        np.arange(n) * SCAN_STEP_SIZE,
        np.arange(n) * SCAN_STEP_SIZE,
        indexing="ij",
    )
    return np.stack((ii.ravel(), jj.ravel()), axis=-1).astype(np.float64)


def simulate_intensities(complex_obj: np.ndarray, probe: np.ndarray) -> np.ndarray:
    """``(N_pos, N, N)`` diffraction intensities, corner-centered in reciprocal space."""
    positions_px = scan_positions_px()
    x0 = np.round(positions_px[:, 0]).astype(int)
    y0 = np.round(positions_px[:, 1]).astype(int)

    x_ind = np.fft.fftfreq(N, d=1 / N).astype(int)
    y_ind = np.fft.fftfreq(N, d=1 / N).astype(int)

    row = (x0[:, None, None] + x_ind[None, :, None]) % N
    col = (y0[:, None, None] + y_ind[None, None, :]) % N

    exit_waves = complex_obj[row, col] * probe
    return np.abs(np.fft.fft2(exit_waves)) ** 2


def make_dataset4d(defocus: float | None = None, seed: int = 42) -> Dataset4d:
    """``(n_scan, n_scan, N, N)`` 4D-STEM dataset, reciprocal axes fftshifted."""
    if defocus is None:
        defocus = integer_shift_defocus(1)

    complex_obj = make_complex_obj(seed)
    probe = make_probe_array(defocus)
    intensities = simulate_intensities(complex_obj, probe)

    n = N // SCAN_STEP_SIZE
    array = np.fft.fftshift(intensities * 100, axes=(-2, -1)).reshape((n, n, N, N))

    return Dataset4d.from_array(
        array.astype(np.float32),
        name="synthetic 4D-STEM",
        sampling=(SCAN_SAMPLING, SCAN_SAMPLING, RECIPROCAL_SAMPLING, RECIPROCAL_SAMPLING),
        units=("A", "A", "A^-1", "A^-1"),
    )


def direct_ptycho_kwargs(defocus: float) -> dict:
    """Constructor kwargs shared by both direct-ptychography classes."""
    return dict(
        energy=PROBE_ENERGY,
        semiangle_cutoff=SEMIANGLE_CUTOFF,
        rotation_angle=0.0,
        aberration_coefs={"C10": defocus},
        force_fitted_origin=ORIGIN,
        verbose=False,
    )


def correlation(image: np.ndarray, reference: np.ndarray) -> float:
    """Pearson correlation between a reconstruction and a reference, means removed."""
    a = np.asarray(image, dtype=np.float64).ravel()
    b = np.asarray(reference, dtype=np.float64).ravel()
    a = a - a.mean()
    b = b - b.mean()
    return float(np.corrcoef(a, b)[0, 1])


@pytest.fixture(scope="module")
def dataset4d():
    """Read-only synthetic dataset at the integer-shift defocus."""
    return make_dataset4d()
