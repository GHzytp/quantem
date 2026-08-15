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
    """``(N_pos, N, N)`` diffraction intensities, corner-centered in reciprocal space.

    ``probe`` is either a single ``(N, N)`` array or a ``(N_pos, N, N)`` stack, which
    broadcasts against the extracted object patches and gives every scan position its own
    probe -- how a tilted sample is simulated.
    """
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


def defocus_per_position(mean_defocus: float, defocus_gradient: tuple[float, float]) -> np.ndarray:
    """``(N_pos,)`` local defocus for a tilted sample, mean-zero about the scan centroid.

    Matches ``ShadowMontagePtychography._return_delta_c10``: the offset is measured from the
    centroid of the scan positions in Angstrom, in the unrotated scan frame.
    """
    positions_ang = scan_positions_px() * SCAN_SAMPLING
    offsets = positions_ang - positions_ang.mean(axis=0)
    return mean_defocus + offsets @ np.asarray(defocus_gradient, dtype=np.float64)


def make_tilted_dataset4d(
    mean_defocus: float,
    defocus_gradient: tuple[float, float],
    seed: int = 42,
) -> Dataset4d:
    """4D-STEM dataset whose defocus varies linearly across the field of view.

    The gradients used in the tests look enormous as tilts -- at this fixture's 32 Angstrom
    field of view a gradient of 12.7 A/A is needed to swing the parallax shift by a single
    scan pixel. That is an artifact of the tiny synthetic scan, not of the method: it is only
    a +/-12% swing on the ~1625 A baseline defocus.
    """
    complex_obj = make_complex_obj(seed)
    defocus = defocus_per_position(mean_defocus, defocus_gradient)
    probes = np.stack([make_probe_array(float(d)) for d in defocus])
    intensities = simulate_intensities(complex_obj, probes)

    n = N // SCAN_STEP_SIZE
    array = np.fft.fftshift(intensities * 100, axes=(-2, -1)).reshape((n, n, N, N))

    return Dataset4d.from_array(
        array.astype(np.float32),
        name="synthetic tilted 4D-STEM",
        sampling=(SCAN_SAMPLING, SCAN_SAMPLING, RECIPROCAL_SAMPLING, RECIPROCAL_SAMPLING),
        units=("A", "A", "A^-1", "A^-1"),
    )


def _bilinear_sample_periodic(image: np.ndarray, coords: np.ndarray) -> np.ndarray:
    """Sample ``image`` at fractional ``(..., 2)`` coordinates, wrapping at the edges."""
    n_rows, n_cols = image.shape
    base = np.floor(coords)
    frac = coords - base
    base = base.astype(np.int64)

    out = np.zeros(coords.shape[:-1], dtype=np.float64)
    for d_row, d_col in ((0, 0), (1, 0), (0, 1), (1, 1)):
        w_row = frac[..., 0] if d_row else 1 - frac[..., 0]
        w_col = frac[..., 1] if d_col else 1 - frac[..., 1]
        out += (
            w_row * w_col * image[(base[..., 0] + d_row) % n_rows, (base[..., 1] + d_col) % n_cols]
        )
    return out


def make_model_vbf_stack(
    mean_defocus: float,
    defocus_gradient: tuple[float, float] = (0.0, 0.0),
    scan_gpts: tuple[int, int] = (96, 96),
    seed: int = 42,
):
    """Virtual bright-field stack built directly from the parallax model, at any scan size.

    Returns ``(vbf_dataset, bf_mask_dataset, object_phase)``.

    Every image is ``v_m(r) = object[r + shift_m(C10(r))]``, the relation the montage inverts
    -- the ``+`` sign was measured against the 4D pipeline, where a bright-field image
    correlates with ``obj[r + shift]`` at 0.30 versus 0.09 for ``obj[r - shift]``.

    The full 4D simulator cannot be used for the position-dependent tests: its 32x32 scan
    only affords ~16-position patches, and at that size the per-patch defocus estimator has a
    ~430 Angstrom bias -- larger than the signal, as the zero-gradient control shows. This
    builds the same relation at whatever scan size the estimator needs, cheaply.

    Bright-field pixels are ordered by ``np.nonzero`` over the corner-centered mask, which is
    the order ``torch.nonzero`` gives in ``_return_bf_context``. Pass ``crop_bf_mask=False``
    so the class uses this mask verbatim and the ordering is preserved.
    """
    from quantem.core.datastructures import Dataset2d, Dataset3d

    wavelength = electron_wavelength_angstrom(PROBE_ENERGY)

    # detector grid just large enough to hold the bright-field disk
    radius_px = int(round(Q_PROBE / RECIPROCAL_SAMPLING))
    n_det = 2 * radius_px + 3
    centered = np.hypot(*np.meshgrid(*(np.arange(n_det) - n_det // 2,) * 2, indexing="ij"))
    bf_mask = np.fft.ifftshift(centered <= radius_px)

    # k = m * reciprocal_sampling exactly, independent of the grid size
    freqs = np.fft.fftfreq(n_det, 1 / (RECIPROCAL_SAMPLING * n_det))
    inds_i, inds_j = np.nonzero(bf_mask)
    k_vec = np.stack((freqs[inds_i], freqs[inds_j]), axis=-1)  # (num_bf, 2)

    rng = np.random.default_rng(seed)
    phase = rng.random(scan_gpts)
    qx = np.fft.fftfreq(scan_gpts[0], SCAN_SAMPLING)
    qy = np.fft.fftfreq(scan_gpts[1], SCAN_SAMPLING)
    q = np.hypot(qx[:, None], qy[None, :])
    obj = np.fft.ifft2(np.fft.fft2(phase) * (q <= 2 * Q_PROBE)).real
    obj -= obj.mean()

    ii, jj = np.meshgrid(np.arange(scan_gpts[0]), np.arange(scan_gpts[1]), indexing="ij")
    positions_px = np.stack((ii.ravel(), jj.ravel()), axis=-1).astype(np.float64)

    offsets_ang = (positions_px - positions_px.mean(0)) * SCAN_SAMPLING
    c10 = mean_defocus + offsets_ang @ np.asarray(defocus_gradient, dtype=np.float64)

    # shift_px[m, n] = wavelength * C10[n] * k[m] / scan_sampling
    shifts = (
        wavelength * c10[None, :, None] * k_vec[:, None, :] / SCAN_SAMPLING
    )  # (num_bf, N_pos, 2)
    stack = _bilinear_sample_periodic(obj, positions_px[None] + shifts)

    vbf_dataset = Dataset3d.from_array(
        stack.reshape(len(k_vec), *scan_gpts).astype(np.float32),
        name="model virtual BF stack",
        sampling=(1.0, SCAN_SAMPLING, SCAN_SAMPLING),
        units=("index", "A", "A"),
    )
    bf_mask_dataset = Dataset2d.from_array(
        bf_mask,
        name="BF mask",
        sampling=(RECIPROCAL_SAMPLING, RECIPROCAL_SAMPLING),
        units=("A^-1", "A^-1"),
    )
    return vbf_dataset, bf_mask_dataset, obj


def model_vbf_kwargs(defocus: float) -> dict:
    """Constructor kwargs for a stack from :func:`make_model_vbf_stack`."""
    return dict(
        energy=PROBE_ENERGY,
        semiangle_cutoff=SEMIANGLE_CUTOFF,
        rotation_angle=0.0,
        aberration_coefs={"C10": defocus},
        crop_bf_mask=False,
        verbose=False,
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
