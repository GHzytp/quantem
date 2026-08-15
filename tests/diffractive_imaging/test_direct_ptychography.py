"""Tests for the Fourier-space direct-ptychography reconstruction.

Covers all five deconvolution kernels, the aberration/rotation conventions they share, the
four hyperparameter-fitting routines, and save/load. Synthetic data comes from
``conftest.py``: a white-noise pure-phase object imaged with a defocused soft aperture.
"""

import warnings

import numpy as np
import pytest
import torch

from quantem.core.datastructures import Dataset2d, Dataset3d
from quantem.core.io.serialize import load
from quantem.core.utils.utils import electron_wavelength_angstrom, to_numpy
from quantem.diffractive_imaging import DirectPtychography, OptimizationParameter
from quantem.diffractive_imaging.complex_probe import spatial_frequencies

from .conftest import (
    DECONVOLUTION_KERNELS,
    PROBE_ENERGY,
    Q_PROBE,
    SCAN_SAMPLING,
    SEMIANGLE_CUTOFF,
    N,
    band_limited_phase,
    correlation,
    direct_ptycho_kwargs,
    integer_shift_defocus,
    make_dataset4d,
    scan_positions_px,
)

TRUE_C10 = integer_shift_defocus(1)


def _build(dataset4d, defocus=TRUE_C10, **overrides):
    kwargs = dict(direct_ptycho_kwargs(defocus), edge_blend_pixels=0)
    kwargs.update(overrides)
    return DirectPtychography.from_dataset4d(dataset4d, **kwargs)


@pytest.fixture(scope="module")
def recon(dataset4d):
    """A reconstruction seeded with the true defocus. Reconstruct before asserting."""
    return _build(dataset4d)


class TestConstruction:
    def test_geometry_matches_the_dataset(self, recon):
        assert recon.bf_mask.shape == recon.gpts
        assert recon.vbf_stack.shape == (recon.num_bf, N, N)
        assert recon.num_bf == int(recon.bf_mask.sum())
        assert recon.scan_gpts == (N, N)
        assert recon.scan_sampling[0] == pytest.approx(SCAN_SAMPLING)

    def test_bf_mask_is_about_the_aperture_size(self, recon):
        """The mask is thresholded from the mean pattern, so it should track the BF disk."""
        expected_area = np.pi * (Q_PROBE / recon.reciprocal_sampling[0]) ** 2
        assert recon.num_bf == pytest.approx(expected_area, rel=0.35)

    def test_preprocess_zeroes_the_dc_bin(self, recon):
        assert torch.allclose(
            recon._vbf_fourier[..., 0, 0], torch.zeros_like(recon._vbf_fourier[..., 0, 0])
        )

    def test_from_virtual_bfs_reproduces_from_dataset4d(self, dataset4d, recon):
        """Re-wrapping the stored stack must give a bit-identical reconstruction."""
        vbf_dataset = Dataset3d.from_array(
            to_numpy(recon.vbf_stack),
            name="vBF stack",
            units=("index", "A", "A"),
            sampling=(1, SCAN_SAMPLING, SCAN_SAMPLING),
        )
        bf_mask_dataset = Dataset2d.from_array(
            to_numpy(recon.bf_mask),
            name="BF mask",
            units=("A^-1", "A^-1"),
            sampling=tuple(recon.reciprocal_sampling),
        )
        rebuilt = DirectPtychography.from_virtual_bfs(
            vbf_dataset,
            bf_mask_dataset,
            energy=PROBE_ENERGY,
            rotation_angle=0.0,
            semiangle_cutoff=SEMIANGLE_CUTOFF,
            aberration_coefs={"C10": TRUE_C10},
            crop_bf_mask=False,  # the stored mask is already cropped
            verbose=False,
        )

        recon.reconstruct(deconvolution_kernel="ssb", verbose=False)
        rebuilt.reconstruct(deconvolution_kernel="ssb", verbose=False)
        assert np.allclose(rebuilt.obj, recon.obj, rtol=1e-5, atol=1e-8)

    def test_cropping_the_bf_mask_preserves_the_reconstruction(self, dataset4d):
        """Cropping shrinks the detector grid but must not permute the vBF stack order."""
        cropped = _build(dataset4d, crop_bf_mask=True)
        uncropped = _build(dataset4d, crop_bf_mask=False)

        assert cropped.gpts[0] < uncropped.gpts[0]
        assert cropped.num_bf == uncropped.num_bf

        cropped.reconstruct(deconvolution_kernel="ssb", verbose=False)
        uncropped.reconstruct(deconvolution_kernel="ssb", verbose=False)
        assert correlation(cropped.obj, uncropped.obj) > 0.99

    def test_direct_instantiation_is_blocked(self):
        with pytest.raises(RuntimeError, match="from_virtual_bfs"):
            DirectPtychography(
                vbf_dataset=None,
                bf_mask_dataset=None,
                energy=PROBE_ENERGY,
                rotation_angle=0.0,
                aberration_coefs={},
                semiangle_cutoff=SEMIANGLE_CUTOFF,
                soft_edges=True,
                crop_bf_mask=False,
                bf_mask_padding_px=1,
                rng=None,
                device="cpu",
                verbose=False,
            )


class TestDeconvolutionKernels:
    @pytest.mark.parametrize("kernel", DECONVOLUTION_KERNELS)
    def test_recovers_the_band_limited_object(self, recon, kernel):
        """Every kernel must recover the object over the band the aperture transfers."""
        recon.reconstruct(deconvolution_kernel=kernel, verbose=False)

        assert recon.obj.shape == (N, N)
        assert np.isfinite(recon.obj).all()
        assert abs(correlation(recon.obj, band_limited_phase())) > 0.7

    @pytest.mark.parametrize("kernel", DECONVOLUTION_KERNELS)
    def test_aliases_resolve(self, recon, kernel):
        aliases = {
            "ssb": "single-sideband",
            "obf": "optimum-bright-field",
            "mf": "matched-filter",
            "prlx": "tilt-corrected-bright-field",
            "icom": "center-of-mass",
        }
        by_short = recon.reconstruct(deconvolution_kernel=kernel, verbose=False).obj.copy()
        by_alias = recon.reconstruct(deconvolution_kernel=aliases[kernel], verbose=False).obj

        assert np.array_equal(by_short, by_alias)

    @pytest.mark.parametrize("upsampling_factor", [1, 2, 3])
    def test_upsampling_preserves_the_field_of_view(self, recon, upsampling_factor):
        recon.reconstruct(
            deconvolution_kernel="ssb", upsampling_factor=upsampling_factor, verbose=False
        )

        assert recon.obj.shape == (N * upsampling_factor, N * upsampling_factor)
        assert recon.obj.shape[0] * recon._obj_sampling[0] == pytest.approx(N * SCAN_SAMPLING)

    def test_corrected_stack_sums_to_corrected_bf(self, recon):
        recon.reconstruct(deconvolution_kernel="ssb", verbose=False)

        assert recon.corrected_stack.shape == (recon.num_bf, N, N)
        assert torch.allclose(recon.corrected_stack.sum(0), recon.corrected_bf)

    def test_unknown_kernel_raises(self, recon):
        with pytest.raises(ValueError, match="Unknown deconvolution kernel"):
            recon.reconstruct(deconvolution_kernel="wiener", verbose=False)


class TestLateralShifts:
    """`_return_lateral_shifts` underpins both the parallax kernel and the montage class."""

    def test_pure_defocus_matches_the_analytic_shift(self, recon):
        """For pure C10 the lateral shift is exactly `wavelength * C10 * k` Angstrom."""
        shifts = recon._return_lateral_shifts(0.0, {"C10": TRUE_C10}, recon.bf_mask)

        kxa, kya = spatial_frequencies(recon.gpts, recon.sampling, device=recon.device)
        expected = (
            torch.stack((kxa[recon.bf_mask], kya[recon.bf_mask]), -1)
            * electron_wavelength_angstrom(PROBE_ENERGY)
            * TRUE_C10
        )

        assert torch.allclose(shifts, expected, rtol=1e-5, atol=1e-6)

    def test_no_aberrations_means_no_shift(self, recon):
        shifts = recon._return_lateral_shifts(0.0, {}, recon.bf_mask)

        assert torch.count_nonzero(shifts) == 0

    def test_rotation_rotates_the_shifts(self, recon):
        """`_passively_rotate_grid` sends (kx, ky) -> (-ky, kx) at 90 degrees."""
        coefs = {"C10": TRUE_C10}
        unrotated = recon._return_lateral_shifts(0.0, coefs, recon.bf_mask)
        rotated = recon._return_lateral_shifts(90.0, coefs, recon.bf_mask)

        expected = torch.stack((-unrotated[:, 1], unrotated[:, 0]), dim=-1)
        assert torch.allclose(rotated, expected, atol=1e-5)

    def test_astigmatism_breaks_the_radial_symmetry(self, recon):
        radial = recon._return_lateral_shifts(0.0, {"C10": TRUE_C10}, recon.bf_mask)
        astigmatic = recon._return_lateral_shifts(
            0.0, {"C10": TRUE_C10, "C12": 0.3 * TRUE_C10}, recon.bf_mask
        )

        assert not torch.allclose(radial, astigmatic)


class TestBrightFieldSubsets:
    def test_checkerboard_halves_are_additive(self, recon):
        """Reconstructions are linear in the BF sum once the per-subset weight is undone."""
        recon.reconstruct(deconvolution_kernel="prlx", parallax_flip_phase=False, verbose=False)
        full = recon.corrected_bf.clone()

        halves = []
        for mask in recon._make_checkerboard_bf_masks(recon.gpts, recon.bf_mask):
            recon.reconstruct(
                bf_mask=mask,
                deconvolution_kernel="prlx",
                parallax_flip_phase=False,
                verbose=False,
            )
            halves.append(recon.corrected_bf.clone() * recon.corrected_stack.shape[0])

        # each half is normalized by its own BF weight; undo that before comparing
        combined = to_numpy(halves[0] + halves[1]) / recon.num_bf
        assert correlation(combined, to_numpy(full)) > 0.99

    def test_halfsets_helper_returns_two_images(self, recon):
        first, second = recon._reconstruct_with_halfsets(deconvolution_kernel="ssb")

        assert first.shape == second.shape == (N, N)
        assert correlation(to_numpy(first), to_numpy(second)) > 0.5

    def test_subset_uses_fewer_bright_field_pixels(self, recon):
        mask = recon._make_checkerboard_bf_masks(recon.gpts, recon.bf_mask)[0]
        recon.reconstruct(bf_mask=mask, deconvolution_kernel="ssb", verbose=False)

        assert recon.corrected_stack.shape[0] == int(mask.sum())
        assert int(mask.sum()) < recon.num_bf


class TestFilters:
    def test_lowpass_suppresses_high_frequencies(self, recon):
        recon.reconstruct(deconvolution_kernel="ssb", verbose=False)
        unfiltered = np.abs(np.fft.fft2(recon.obj))
        recon.reconstruct(deconvolution_kernel="ssb", q_lowpass=0.05, verbose=False)
        filtered = np.abs(np.fft.fft2(recon.obj))

        qx = qy = np.fft.fftfreq(N, SCAN_SAMPLING)
        high = np.hypot(qx[:, None], qy[None, :]) > 0.15

        assert filtered[high].sum() < 0.05 * unfiltered[high].sum()

    def test_highpass_suppresses_low_frequencies(self, recon):
        recon.reconstruct(deconvolution_kernel="ssb", verbose=False)
        unfiltered = np.abs(np.fft.fft2(recon.obj))
        recon.reconstruct(deconvolution_kernel="ssb", q_highpass=0.15, verbose=False)
        filtered = np.abs(np.fft.fft2(recon.obj))

        qx = qy = np.fft.fftfreq(N, SCAN_SAMPLING)
        low = (np.hypot(qx[:, None], qy[None, :]) < 0.05) & (
            np.hypot(qx[:, None], qy[None, :]) > 0
        )

        assert filtered[low].sum() < 0.2 * unfiltered[low].sum()

    def test_parallax_phase_flip_vanishes_at_zero_defocus(self, recon):
        """`sign(sin(chi))` is identically zero when chi is, so the image is too.

        Not a defect: an unaberrated probe transfers no phase contrast in this formulation.
        Worth pinning because the all-zero output is otherwise startling.
        """
        recon.reconstruct(
            deconvolution_kernel="prlx",
            override_aberration_coefs={"C10": 0.0},
            parallax_flip_phase=True,
            verbose=False,
        )
        assert np.ptp(recon.obj) == 0.0

        recon.reconstruct(
            deconvolution_kernel="prlx",
            override_aberration_coefs={"C10": 0.0},
            parallax_flip_phase=False,
            verbose=False,
        )
        assert np.ptp(recon.obj) > 0.0


class TestVarianceLoss:
    def test_is_positive_after_reconstructing(self, recon):
        recon.reconstruct(deconvolution_kernel="prlx", verbose=False)

        assert float(recon.variance_loss()) > 0

    def test_is_minimized_at_the_true_defocus(self, recon):
        losses = {}
        for scale in (0.5, 0.8, 1.0, 1.2, 1.5):
            recon.reconstruct(
                deconvolution_kernel="prlx",
                override_aberration_coefs={"C10": scale * TRUE_C10},
                parallax_flip_phase=False,
                verbose=False,
            )
            losses[scale] = float(recon.variance_loss())

        assert min(losses, key=losses.get) == pytest.approx(1.0)


class TestHyperparameterFitting:
    """All four routines must recover a seeded defocus from a blind start."""

    def test_grid_search(self, dataset4d):
        recon = _build(dataset4d, aberration_coefs={})
        recon.grid_search_hyperparameters(
            aberration_coefs={
                "C10": OptimizationParameter(low=0.4 * TRUE_C10, high=1.6 * TRUE_C10, n_points=7)
            },
            deconvolution_kernel="prlx",
            verbose=False,
        )
        fitted = recon.hyperparameter_state.optimized_aberrations["C10"]

        step = (1.6 - 0.4) * TRUE_C10 / 6
        assert abs(fitted - TRUE_C10) <= step

    def test_least_squares(self, dataset4d):
        recon = _build(dataset4d, aberration_coefs={})
        recon.fit_hyperparameters_least_squares(
            cartesian_basis="defocus", fit_method="global", verbose=False
        )
        fitted = recon.hyperparameter_state.optimized_aberrations["C10"]

        assert fitted == pytest.approx(TRUE_C10, rel=0.15)

    def test_cross_correlation(self, dataset4d):
        recon = _build(dataset4d, aberration_coefs={})
        recon.fit_hyperparameters_cross_correlation(bin_factors=(2, 1), verbose=False)
        state = recon.hyperparameter_state

        assert state.optimized_aberrations["C10"] == pytest.approx(TRUE_C10, rel=0.3)
        assert state.optimized_rotation_angle == pytest.approx(0.0, abs=5.0)

    @pytest.mark.slow
    def test_optuna(self, dataset4d):
        recon = _build(dataset4d, aberration_coefs={})
        recon.optimize_hyperparameters(
            aberration_coefs={
                "C10": OptimizationParameter(low=0.4 * TRUE_C10, high=1.6 * TRUE_C10)
            },
            n_trials=25,
            deconvolution_kernel="prlx",
            verbose=False,
        )
        fitted = recon.hyperparameter_state.optimized_aberrations["C10"]

        assert fitted == pytest.approx(TRUE_C10, rel=0.15)
        assert recon.hyperparameter_state.study is not None

    def test_fitting_leaves_the_initial_state_intact(self, dataset4d):
        """`use_initial_state=True` must ignore whatever a fit wrote back."""
        recon = _build(dataset4d)
        recon.hyperparameter_state.optimized_aberrations = {"C10": 5.0 * TRUE_C10}

        recon.reconstruct(deconvolution_kernel="ssb", use_initial_state=True, verbose=False)
        from_initial = recon.obj.copy()
        recon.reconstruct(
            deconvolution_kernel="ssb",
            override_aberration_coefs={"C10": TRUE_C10},
            verbose=False,
        )

        assert np.allclose(from_initial, recon.obj)


class TestHyperparameterState:
    def test_optimized_overrides_initial(self, recon):
        state = recon.hyperparameter_state
        state.clear_optimized()

        assert state.current_rotation_angle() == 0.0
        state.optimized_rotation_angle = 12.0
        assert state.current_rotation_angle() == 12.0
        assert state.current_rotation_angle(override_fixed=3.0) == 3.0

        state.clear_optimized()
        assert state.current_aberrations()["C10"] == pytest.approx(TRUE_C10)

    def test_defocus_alias_is_negated(self):
        from quantem.diffractive_imaging.direct_ptychography_base import HyperparameterState

        state = HyperparameterState(initial_aberrations={"defocus": 100.0})

        assert state.initial_aberrations == {"C10": -100.0}

    def test_rejects_unknown_aberrations(self):
        from quantem.diffractive_imaging.direct_ptychography_base import HyperparameterState

        with pytest.raises(ValueError):
            HyperparameterState(initial_aberrations={"C99": 1.0})


class TestSerialization:
    def test_round_trip_preserves_the_reconstruction(self, dataset4d, tmp_path):
        recon = _build(dataset4d)
        recon.reconstruct(deconvolution_kernel="ssb", verbose=False)
        before = recon.obj.copy()

        path = str(tmp_path / "direct.zip")
        recon.save(path, mode="o")
        restored = load(path)

        assert isinstance(restored, DirectPtychography)
        assert np.array_equal(restored.obj, before)
        assert restored.num_bf == recon.num_bf
        assert restored.gpts == recon.gpts

        restored.reconstruct(deconvolution_kernel="ssb", verbose=False)
        assert np.allclose(restored.obj, before)


class TestRotationSensitivity:
    def test_wrong_rotation_degrades_the_reconstruction(self, dataset4d):
        """The data is simulated unrotated, so 0 degrees must beat a large rotation."""
        recon = _build(dataset4d)
        reference = band_limited_phase()

        recon.reconstruct(deconvolution_kernel="prlx", override_rotation_angle=0.0, verbose=False)
        aligned = abs(correlation(recon.obj, reference))
        recon.reconstruct(deconvolution_kernel="prlx", override_rotation_angle=60.0, verbose=False)
        misaligned = abs(correlation(recon.obj, reference))

        assert aligned > misaligned

    def test_detector_rotation_is_estimated_when_not_supplied(self, dataset4d):
        recon = DirectPtychography.from_dataset4d(
            dataset4d,
            energy=PROBE_ENERGY,
            semiangle_cutoff=SEMIANGLE_CUTOFF,
            rotation_angle=None,
            aberration_coefs={"C10": TRUE_C10},
            edge_blend_pixels=0,
            verbose=False,
        )

        # simulated without rotation; the curl-minimizing estimate should land near 0 or 180
        estimated = abs(recon.rotation_angle) % 180
        assert min(estimated, 180 - estimated) < 20


class TestReconstructAllPermutations:
    def test_returns_one_image_per_kernel(self, recon):
        images = recon._reconstruct_all_permutations(verbose=False)

        assert len(images) == len(DECONVOLUTION_KERNELS)
        assert all(image.shape == (N, N) for image in images)
        assert all(np.isfinite(image).all() for image in images)


class TestNormalizationOrder:
    def test_linear_background_normalization_runs(self, dataset4d):
        recon = _build(dataset4d, normalization_order=1)
        recon.reconstruct(deconvolution_kernel="ssb", verbose=False)

        assert np.isfinite(recon.obj).all()
        assert abs(correlation(recon.obj, band_limited_phase())) > 0.5

    def test_rejects_unknown_order(self, dataset4d):
        with pytest.raises(ValueError, match="normalization_order"):
            _build(dataset4d, normalization_order=2)

    def test_edge_blending_tapers_the_stack(self, dataset4d):
        """A nonzero blend pulls the scan-edge vBF values toward unity."""
        blended = _build(dataset4d, edge_blend_pixels=4)
        sharp = _build(dataset4d, edge_blend_pixels=0)

        edge_blended = blended.vbf_stack[:, 0, :]
        edge_sharp = sharp.vbf_stack[:, 0, :]

        assert (edge_blended - 1).abs().mean() < (edge_sharp - 1).abs().mean()


class TestDatasetVariants:
    def test_reconstruction_tracks_the_simulated_defocus(self):
        """Data simulated at a different defocus must prefer that defocus."""
        other_C10 = 2 * TRUE_C10
        dataset = make_dataset4d(defocus=other_C10)
        recon = DirectPtychography.from_dataset4d(
            dataset, edge_blend_pixels=0, **direct_ptycho_kwargs(other_C10)
        )

        losses = {}
        for scale in (0.5, 1.0, 1.5):
            recon.reconstruct(
                deconvolution_kernel="prlx",
                override_aberration_coefs={"C10": scale * other_C10},
                parallax_flip_phase=False,
                verbose=False,
            )
            losses[scale] = float(recon.variance_loss())

        assert min(losses, key=losses.get) == pytest.approx(1.0)


class TestFromDataset3d:
    """Ungridded scans, by resampling the bright-field stack onto a grid first.

    Every kernel here is a scan-space Fourier multiplier and so needs a regular grid. Once
    regridded they all run unchanged, which is what makes this exact for positions that were
    already on a lattice.
    """

    @staticmethod
    def _dataset3d(dataset4d):
        return Dataset3d.from_array(
            np.asarray(dataset4d.array).reshape(-1, N, N),
            name="ungridded patterns",
            sampling=(1.0, dataset4d.sampling[-2], dataset4d.sampling[-1]),
            units=("index", "A^-1", "A^-1"),
        )

    @staticmethod
    def _build(dataset3d, positions, **overrides):
        kwargs = dict(
            energy=PROBE_ENERGY,
            semiangle_cutoff=SEMIANGLE_CUTOFF,
            rotation_angle=0.0,
            scan_sampling=(SCAN_SAMPLING, SCAN_SAMPLING),
            aberration_coefs={"C10": TRUE_C10},
            force_fitted_origin=(N // 2, N // 2),
            verbose=False,
        )
        kwargs.update(overrides)
        return DirectPtychography.from_dataset3d(dataset3d, positions, **kwargs)

    @pytest.mark.parametrize("kernel", DECONVOLUTION_KERNELS)
    def test_lattice_positions_reproduce_from_dataset4d(self, dataset4d, kernel):
        """On a lattice the splat is an identity map, so this must be exact."""
        gridded = _build(dataset4d)
        ungridded = self._build(self._dataset3d(dataset4d), scan_positions_px() * SCAN_SAMPLING)

        assert ungridded.scan_gpts == gridded.scan_gpts
        assert np.allclose(
            ungridded.reconstruct(deconvolution_kernel=kernel, verbose=False).obj,
            gridded.reconstruct(deconvolution_kernel=kernel, verbose=False).obj,
            atol=1e-6,
        )

    def test_position_axis_order_is_row_col(self, dataset4d):
        """Swapping the columns must transpose the regridded stack, not scramble it.

        Checked with no aberrations, so the parallax shifts vanish. With a shift present the
        relation does not hold: the shifts come from the detector k-grid, which transposing
        the *positions* leaves alone.
        """
        dataset3d = self._dataset3d(dataset4d)
        positions = scan_positions_px() * SCAN_SAMPLING
        flat = dict(aberration_coefs={}, scan_gpts=(N, N))

        row_col = self._build(dataset3d, positions, **flat)
        col_row = self._build(dataset3d, positions[:, ::-1].copy(), **flat)

        assert np.allclose(
            to_numpy(col_row.vbf_stack),
            to_numpy(row_col.vbf_stack).transpose(0, 2, 1),
            atol=1e-5,
        )

    def test_jittered_positions_recover_the_object(self, dataset4d):
        rng = np.random.default_rng(0)
        positions = scan_positions_px() * SCAN_SAMPLING
        positions = positions + rng.uniform(-0.3, 0.3, positions.shape) * SCAN_SAMPLING

        reconstruction = self._build(self._dataset3d(dataset4d), positions, scan_gpts=(N, N))
        obj = reconstruction.reconstruct(deconvolution_kernel="prlx", verbose=False).obj

        assert correlation(obj, band_limited_phase()) > 0.5

    def test_warns_when_the_grid_has_holes(self, dataset4d):
        """A pixel no probe reached stays zero, and the FFT reads that as signal."""
        positions = scan_positions_px() * SCAN_SAMPLING

        with pytest.warns(UserWarning, match="received no probe position"):
            self._build(self._dataset3d(dataset4d), positions, scan_gpts=(3 * N, 3 * N))

    def test_no_warning_on_a_fully_covered_grid(self, dataset4d):
        positions = scan_positions_px() * SCAN_SAMPLING

        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            self._build(self._dataset3d(dataset4d), positions)

    def test_requested_grid_keeps_the_field_of_view(self, dataset4d):
        """A finer grid must upsample the same area, not crop it."""
        positions = scan_positions_px() * SCAN_SAMPLING
        coarse = self._build(self._dataset3d(dataset4d), positions)
        with pytest.warns(UserWarning, match="received no probe position"):
            finer = self._build(self._dataset3d(dataset4d), positions, scan_gpts=(2 * N, 2 * N))

        assert finer.scan_gpts == (2 * N, 2 * N)
        assert finer.fov == pytest.approx(coarse.fov)

    def test_auto_scan_sampling_warns_and_infers(self, dataset4d):
        positions = scan_positions_px() * SCAN_SAMPLING

        with pytest.warns(UserWarning, match="Inferred scan_sampling"):
            reconstruction = self._build(
                self._dataset3d(dataset4d), positions, scan_sampling="auto"
            )

        assert reconstruction.scan_sampling[0] == pytest.approx(SCAN_SAMPLING)

    def test_accepts_a_dataset2d_of_positions(self, dataset4d):
        positions = Dataset2d.from_array(
            scan_positions_px() * SCAN_SAMPLING,
            name="positions",
            sampling=(1.0, 1.0),
            units=("A", "A"),
        )
        reconstruction = self._build(self._dataset3d(dataset4d), positions)

        assert reconstruction.scan_gpts == (N, N)

    def test_rejects_positions_in_the_wrong_units(self, dataset4d):
        positions = Dataset2d.from_array(
            scan_positions_px() * SCAN_SAMPLING,
            name="positions",
            sampling=(1.0, 1.0),
            units=("nm", "nm"),
        )
        with pytest.raises(ValueError, match="must be given in 'A'"):
            self._build(self._dataset3d(dataset4d), positions)

    def test_rejects_mismatched_position_count(self, dataset4d):
        with pytest.raises(ValueError, match="rows but `dataset` has"):
            self._build(self._dataset3d(dataset4d), scan_positions_px()[:10] * SCAN_SAMPLING)

    def test_rejects_linear_normalization(self, dataset4d):
        with pytest.raises(ValueError, match="needs a scan grid"):
            self._build(
                self._dataset3d(dataset4d),
                scan_positions_px() * SCAN_SAMPLING,
                normalization_order=1,
            )

    def test_survives_a_serialization_round_trip(self, dataset4d, tmp_path):
        reconstruction = self._build(
            self._dataset3d(dataset4d), scan_positions_px() * SCAN_SAMPLING
        )
        reconstruction.reconstruct(deconvolution_kernel="prlx", verbose=False)
        path = tmp_path / "ungridded.zip"
        reconstruction.save(path, mode="o")

        assert np.allclose(load(path).obj, reconstruction.obj)

    @staticmethod
    def _disk_masked(dataset4d, radius=14.0):
        """A non-rectangular scan subset -- the case that exposes hole handling."""
        rows_cols = scan_positions_px()
        center = (N - 1) / 2
        keep = ((rows_cols[:, 0] - center) ** 2 + (rows_cols[:, 1] - center) ** 2) < radius**2
        dataset3d = Dataset3d.from_array(
            np.asarray(dataset4d.array).reshape(-1, N, N)[keep],
            name="masked patterns",
            sampling=(1.0, dataset4d.sampling[-2], dataset4d.sampling[-1]),
            units=("index", "A^-1", "A^-1"),
        )
        return dataset3d, rows_cols[keep] * SCAN_SAMPLING, keep

    def test_mean_hole_fill_beats_zero_on_a_masked_scan(self, dataset4d):
        """Regression guard: zero-filled holes wreck the reconstruction, mean-filled do not.

        `_preprocess` zeroes the DC bin, subtracting the mean over the whole grid including
        holes, so zero-filled holes sit at `-mean` -- a hard-edged step the deconvolution
        smears everywhere. Measured 0.25 vs 0.69 correlation with ground truth.
        """
        dataset3d, positions, keep = self._disk_masked(dataset4d)
        rows_cols = scan_positions_px()[keep].astype(int)
        low, high = rows_cols.min(0), rows_cols.max(0) + 1
        truth = band_limited_phase()[low[0] : high[0], low[1] : high[1]]

        def score(hole_fill):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                recon = self._build(dataset3d, positions, hole_fill=hole_fill)
            obj = recon.reconstruct(deconvolution_kernel="prlx", verbose=False).obj
            return correlation(obj[: truth.shape[0], : truth.shape[1]], truth)

        assert score("mean") > 0.6
        assert score("mean") > score("zero") + 0.2

    def test_mean_fill_matches_the_montage_on_a_masked_scan(self, dataset4d):
        """With holes filled, the two formulations agree on data neither was built for."""
        from quantem.diffractive_imaging import ShadowMontagePtychography

        dataset3d, positions, keep = self._disk_masked(dataset4d)
        rows_cols = scan_positions_px()[keep].astype(int)
        low, high = rows_cols.min(0), rows_cols.max(0) + 1
        truth = band_limited_phase()[low[0] : high[0], low[1] : high[1]]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            fourier = self._build(dataset3d, positions)
        montage = ShadowMontagePtychography.from_dataset3d(
            dataset3d,
            positions,
            energy=PROBE_ENERGY,
            semiangle_cutoff=SEMIANGLE_CUTOFF,
            rotation_angle=0.0,
            scan_sampling=(SCAN_SAMPLING, SCAN_SAMPLING),
            aberration_coefs={"C10": TRUE_C10},
            force_fitted_origin=(N // 2, N // 2),
            verbose=False,
        )

        fourier_obj = fourier.reconstruct(deconvolution_kernel="prlx", verbose=False).obj
        montage.reconstruct(deconvolution_kernel="prlx", weight_normalize=False, verbose=False)
        origin = to_numpy(montage._canvas_origin_px)
        row0, col0 = int(round(-origin[0])), int(round(-origin[1]))
        montage_obj = montage.obj[row0 : row0 + truth.shape[0], col0 : col0 + truth.shape[1]]

        fourier_corr = correlation(fourier_obj[: truth.shape[0], : truth.shape[1]], truth)
        montage_corr = correlation(montage_obj, truth)

        assert fourier_corr > 0.6
        assert abs(fourier_corr - montage_corr) < 0.1

    def test_rejects_an_unknown_hole_fill(self, dataset4d):
        dataset3d, positions, _ = self._disk_masked(dataset4d)
        with pytest.raises(ValueError, match="`hole_fill` must be"):
            self._build(dataset3d, positions, hole_fill="interpolate")
