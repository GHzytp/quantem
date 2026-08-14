"""Tests for the real-space (shadow montage) direct-ptychography reconstruction.

The headline check is equivalence with the Fourier-space parallax kernel of
``DirectPtychography``: the two are the same linear operator written in different domains,
so on a raster scan with periodic wraparound they must agree.
"""

import inspect

import numpy as np
import pytest
import torch

from quantem.core.datastructures import Dataset2d, Dataset3d
from quantem.core.io.serialize import load
from quantem.core.utils.utils import to_numpy
from quantem.diffractive_imaging import (
    DirectPtychography,
    OptimizationParameter,
    ShadowMontagePtychography,
)
from quantem.diffractive_imaging.direct_ptycho_utils import (
    allocate_splat_buffers,
    scatter_add_splat,
)

from .conftest import (
    ORIGIN,
    PROBE_ENERGY,
    RECIPROCAL_SAMPLING,
    SCAN_SAMPLING,
    N,
    integer_shift_defocus,
    scan_positions_px,
)
from .conftest import (
    direct_ptycho_kwargs as _common_kwargs,
)


def _build_pair(dataset4d, defocus):
    """A `DirectPtychography` and a `ShadowMontagePtychography` over the same data."""
    fourier = DirectPtychography.from_dataset4d(
        dataset4d, edge_blend_pixels=0, **_common_kwargs(defocus)
    )
    montage = ShadowMontagePtychography.from_dataset4d(
        dataset4d, edge_blend_pixels=0, boundary="wrap", **_common_kwargs(defocus)
    )
    return fourier, montage


def _relative_error(a, b):
    return float(np.abs(a - b).max() / np.abs(b).max())


class TestSplatKernel:
    """Unit tests for `scatter_add_splat`, independent of any reconstruction."""

    def test_integer_coordinates_deposit_one_pixel(self):
        values = torch.tensor([[2.0]])
        coords = torch.tensor([[[3.0, 4.0]]])
        sum_w, sum_wv, sum_wv2 = scatter_add_splat(values, coords, (8, 8))

        assert sum_w.sum().item() == pytest.approx(1.0)
        assert sum_w.reshape(8, 8)[3, 4].item() == pytest.approx(1.0)
        assert sum_wv.reshape(8, 8)[3, 4].item() == pytest.approx(2.0)
        assert sum_wv2.reshape(8, 8)[3, 4].item() == pytest.approx(4.0)

    def test_half_pixel_splits_four_ways(self):
        values = torch.tensor([[1.0]])
        coords = torch.tensor([[[3.5, 4.5]]])
        sum_w, _, _ = scatter_add_splat(values, coords, (8, 8))

        nonzero = sum_w[sum_w > 0]
        assert nonzero.numel() == 4
        assert torch.allclose(nonzero, torch.full((4,), 0.25, dtype=nonzero.dtype))

    def test_wrap_is_periodic(self):
        values = torch.tensor([[1.0]])
        coords = torch.tensor([[[-0.5, 8.25]]])
        sum_w, _, _ = scatter_add_splat(values, coords, (8, 8), boundary="wrap")

        assert sum_w.sum().item() == pytest.approx(1.0)
        touched = {tuple(ij) for ij in torch.nonzero(sum_w.reshape(8, 8)).tolist()}
        assert touched == {(7, 0), (7, 1), (0, 0), (0, 1)}

    def test_pad_drops_out_of_bounds_without_corrupting_the_edge(self):
        values = torch.tensor([[100.0, 2.0]])
        coords = torch.tensor([[[-5.0, 4.0], [3.0, 4.0]]])
        sum_w, sum_wv, _ = scatter_add_splat(values, coords, (8, 8), boundary="pad")

        # only the in-bounds point contributes, and the clamped row picks up nothing
        assert sum_w.sum().item() == pytest.approx(1.0)
        assert sum_wv.sum().item() == pytest.approx(2.0)
        assert sum_w.reshape(8, 8)[0].max().item() == pytest.approx(0.0)

    def test_weights_are_a_partition_of_unity(self):
        generator = torch.Generator().manual_seed(0)
        values = torch.randn(4, 100, generator=generator)
        coords = torch.rand(4, 100, 2, generator=generator) * 8
        sum_w, _, _ = scatter_add_splat(values, coords, (8, 8), boundary="wrap")

        assert sum_w.sum().item() == pytest.approx(400.0)

    def test_matches_brute_force(self):
        generator = torch.Generator().manual_seed(1)
        values = torch.randn(3, 50, generator=generator)
        coords = torch.rand(3, 50, 2, generator=generator) * 10 - 1

        sum_w, sum_wv, sum_wv2 = scatter_add_splat(values, coords, (10, 10), boundary="pad")

        # the splat accumulates in float64, so the reference must too
        values_np = to_numpy(values).astype(np.float64)
        coords_np = to_numpy(coords).astype(np.float64)
        ref_w, ref_wv, ref_wv2 = (np.zeros(100) for _ in range(3))
        for b in range(3):
            for t in range(50):
                r0 = int(np.floor(coords_np[b, t, 0]))
                c0 = int(np.floor(coords_np[b, t, 1]))
                fr = coords_np[b, t, 0] - r0
                fc = coords_np[b, t, 1] - c0
                for d_row, d_col in ((0, 0), (1, 0), (0, 1), (1, 1)):
                    row, col = r0 + d_row, c0 + d_col
                    if not (0 <= row < 10 and 0 <= col < 10):
                        continue
                    weight = (fr if d_row else 1 - fr) * (fc if d_col else 1 - fc)
                    value = values_np[b, t]
                    ref_w[row * 10 + col] += weight
                    ref_wv[row * 10 + col] += weight * value
                    ref_wv2[row * 10 + col] += weight * value**2

        assert np.allclose(to_numpy(sum_w), ref_w)
        assert np.allclose(to_numpy(sum_wv), ref_wv)
        assert np.allclose(to_numpy(sum_wv2), ref_wv2)

    def test_nearest_rounds(self):
        values = torch.tensor([[1.0]])
        coords = torch.tensor([[[3.4, 4.6]]])
        sum_w, _, _ = scatter_add_splat(values, coords, (8, 8), interpolation="nearest")

        assert torch.nonzero(sum_w.reshape(8, 8)).tolist() == [[3, 5]]

    def test_accumulates_into_provided_buffers(self):
        buffers = allocate_splat_buffers((8, 8), "cpu")
        values = torch.tensor([[1.0]])
        coords = torch.tensor([[[3.0, 4.0]]])

        for _ in range(3):
            scatter_add_splat(values, coords, (8, 8), out=buffers)

        assert buffers[0].reshape(8, 8)[3, 4].item() == pytest.approx(3.0)

    def test_rejects_unknown_modes(self):
        values = torch.tensor([[1.0]])
        coords = torch.tensor([[[3.0, 4.0]]])
        with pytest.raises(ValueError, match="boundary"):
            scatter_add_splat(values, coords, (8, 8), boundary="reflect")
        with pytest.raises(ValueError, match="interpolation"):
            scatter_add_splat(values, coords, (8, 8), interpolation="cubic")


class TestIntegerShiftConstruction:
    """The defocus used below must put every BF pixel on an exact canvas pixel."""

    @pytest.mark.parametrize("upsampling_factor", [1, 2])
    @pytest.mark.parametrize("pixel_shift", [1, 2])
    def test_shifts_land_on_pixel_centers(self, dataset4d, upsampling_factor, pixel_shift):
        defocus = integer_shift_defocus(pixel_shift, upsampling_factor)
        montage = ShadowMontagePtychography.from_dataset4d(
            dataset4d, boundary="wrap", **_common_kwargs(defocus)
        )
        shifts, _ = montage._return_shifts_px(
            0.0, {"C10": defocus}, montage.bf_mask, upsampling_factor
        )
        residual = (shifts - shifts.round()).abs().max().item()

        assert residual < 1e-4, f"shifts are not integral: max residual {residual:.2e} px"
        assert shifts.abs().max().item() > 1.0, "test would be vacuous with zero shifts"


class TestFourierEquivalence:
    """`ShadowMontagePtychography` must reproduce `DirectPtychography`'s parallax kernel."""

    @pytest.mark.parametrize("upsampling_factor", [1, 2])
    @pytest.mark.parametrize("pixel_shift", [1, 2])
    def test_matches_parallax_kernel(self, dataset4d, upsampling_factor, pixel_shift):
        defocus = integer_shift_defocus(pixel_shift, upsampling_factor)
        fourier, montage = _build_pair(dataset4d, defocus)

        recon_kwargs = dict(
            deconvolution_kernel="prlx",
            parallax_flip_phase=False,
            upsampling_factor=upsampling_factor,
            verbose=False,
        )
        fourier.reconstruct(**recon_kwargs)
        montage.reconstruct(
            boundary="wrap", interpolation="bilinear", weight_normalize=False, **recon_kwargs
        )

        assert montage.obj.shape == fourier.obj.shape
        assert _relative_error(montage.obj, fourier.obj) < 1e-4

    def test_matches_with_phase_flip(self, dataset4d):
        """The phase-flip filter is BF-independent, so post-hoc application is exact."""
        defocus = integer_shift_defocus(1)
        fourier, montage = _build_pair(dataset4d, defocus)

        fourier.reconstruct(deconvolution_kernel="prlx", parallax_flip_phase=True, verbose=False)
        montage.reconstruct(
            deconvolution_kernel="prlx",
            parallax_flip_phase=True,
            weight_normalize=False,
            verbose=False,
        )

        assert _relative_error(montage.obj, fourier.obj) < 1e-4

    def test_matches_with_butterworth_filters(self, dataset4d):
        defocus = integer_shift_defocus(1)
        fourier, montage = _build_pair(dataset4d, defocus)

        recon_kwargs = dict(
            deconvolution_kernel="prlx",
            parallax_flip_phase=False,
            q_lowpass=0.2,
            q_highpass=0.02,
            verbose=False,
        )
        fourier.reconstruct(**recon_kwargs)
        montage.reconstruct(weight_normalize=False, **recon_kwargs)

        assert _relative_error(montage.obj, fourier.obj) < 1e-4

    def test_weight_normalization_differs_only_by_a_constant(self, dataset4d):
        """On a full grid at U=1 the accumulated weight is exactly num_bf everywhere."""
        defocus = integer_shift_defocus(1)
        _, montage = _build_pair(dataset4d, defocus)

        montage.reconstruct(
            deconvolution_kernel="prlx",
            parallax_flip_phase=False,
            weight_normalize=False,
            verbose=False,
        )
        unnormalized = montage.obj
        bf_weights = float(montage._bf_weights)

        montage.reconstruct(
            deconvolution_kernel="prlx",
            parallax_flip_phase=False,
            weight_normalize=True,
            verbose=False,
        )
        normalized = montage.obj

        assert np.allclose(to_numpy(montage.weights), montage.num_bf, rtol=1e-6)
        rescaled = normalized * montage.num_bf / bf_weights
        assert _relative_error(rescaled, unnormalized) < 1e-4

    def test_variance_loss_tracks_the_fourier_one(self, dataset4d):
        defocus = integer_shift_defocus(1)
        fourier, montage = _build_pair(dataset4d, defocus)

        recon_kwargs = dict(deconvolution_kernel="prlx", parallax_flip_phase=False, verbose=False)
        fourier.reconstruct(**recon_kwargs)
        montage.reconstruct(weight_normalize=False, **recon_kwargs)

        bf_weights = float(montage._bf_weights)
        expected = float(fourier.variance_loss()) * bf_weights**2
        assert float(montage.variance_loss()) == pytest.approx(expected, rel=1e-3)

    def test_bf_mask_subsets_are_additive(self, dataset4d):
        """Checkerboard half-sets, rescaled by their BF weights, sum to the whole."""
        defocus = integer_shift_defocus(1)
        _, montage = _build_pair(dataset4d, defocus)

        recon_kwargs = dict(
            deconvolution_kernel="prlx",
            parallax_flip_phase=False,
            weight_normalize=False,
            verbose=False,
        )
        montage.reconstruct(**recon_kwargs)
        full = montage.obj * float(montage._bf_weights)

        halves = []
        for mask in montage._make_checkerboard_bf_masks(montage.gpts, montage.bf_mask):
            montage.reconstruct(bf_mask=mask, **recon_kwargs)
            halves.append(montage.obj * float(montage._bf_weights))

        assert _relative_error(halves[0] + halves[1], full) < 1e-4

    def test_rejects_non_parallax_kernels(self, dataset4d):
        _, montage = _build_pair(dataset4d, integer_shift_defocus(1))
        with pytest.raises(ValueError, match="only implements the parallax kernel"):
            montage.reconstruct(deconvolution_kernel="ssb", verbose=False)


class TestRotationConvention:
    """`spatial_frequencies` rotates the k-grid; shifts must follow it, unflipped."""

    def test_ninety_degrees_rotates_the_shifts(self, dataset4d):
        defocus = integer_shift_defocus(1)
        _, montage = _build_pair(dataset4d, defocus)
        coefs = {"C10": defocus}

        unrotated, _ = montage._return_shifts_px(0.0, coefs, montage.bf_mask, 1)
        rotated, _ = montage._return_shifts_px(90.0, coefs, montage.bf_mask, 1)

        # _passively_rotate_grid sends (kx, ky) -> (kx cos a - ky sin a, kx sin a + ky cos a),
        # so at 90 deg the shifts, which are parallel to k for pure defocus, map (r, c) -> (-c, r)
        expected = torch.stack((-unrotated[:, 1], unrotated[:, 0]), dim=-1)
        assert torch.allclose(rotated, expected, atol=1e-4)

    def test_rotation_changes_the_reconstruction(self, dataset4d):
        defocus = integer_shift_defocus(1)
        _, montage = _build_pair(dataset4d, defocus)

        montage.reconstruct(override_rotation_angle=0.0, verbose=False)
        unrotated = montage.obj.copy()
        montage.reconstruct(override_rotation_angle=30.0, verbose=False)

        assert not np.allclose(unrotated, montage.obj)


class TestReconstructDefaults:
    """The defaults are behaviour; pin them."""

    def test_interpolation_defaults_to_nearest(self, dataset4d):
        _, montage = _build_pair(dataset4d, integer_shift_defocus(1))

        default = montage.reconstruct(upsampling_factor=2, verbose=False).obj.copy()
        nearest = montage.reconstruct(
            upsampling_factor=2, interpolation="nearest", verbose=False
        ).obj

        assert np.array_equal(default, nearest)

    def test_nearest_is_a_roll_of_the_bright_field_images(self, dataset4d):
        """On a raster scan, snapping moves every position of a BF image by one integer.

        `positions_px * U` is an exact integer, so `round(n + s) == n + round(s)`.
        """
        _, montage = _build_pair(dataset4d, integer_shift_defocus(1))
        upsampling_factor = 4
        shifts, _ = montage._return_shifts_px(
            0.0, montage.aberration_coefs, montage.bf_mask, upsampling_factor
        )
        positions = montage.positions_px * upsampling_factor
        coords = positions[None] + shifts[:, None]

        offsets = coords.round() - positions[None]
        # every position of a given BF image moves by the same integer
        assert torch.equal(offsets, shifts.round()[:, None].expand_as(offsets))

    def test_gridded_constructors_flag_the_scan_as_gridded(self, dataset4d):
        _, montage = _build_pair(dataset4d, integer_shift_defocus(1))
        assert montage.gridded_scan is True

    def test_weight_normalize_defaults_off_for_a_raster_scan(self, dataset4d):
        """Uniform density needs no correction, and normalizing amplifies edge noise."""
        _, montage = _build_pair(dataset4d, integer_shift_defocus(1))

        default = montage.reconstruct(boundary="pad", verbose=False).obj.copy()
        unnormalized = montage.reconstruct(
            boundary="pad", weight_normalize=False, verbose=False
        ).obj

        assert np.array_equal(default, unnormalized)

    def test_weight_normalize_defaults_on_for_an_ungridded_scan(self, dataset4d):
        dataset3d, positions = TestNonGridScan._dataset3d_and_positions(dataset4d)
        recon = ShadowMontagePtychography.from_dataset3d(
            dataset3d,
            positions,
            scan_sampling=(SCAN_SAMPLING, SCAN_SAMPLING),
            **_common_kwargs(integer_shift_defocus(1)),
        )

        assert recon.gridded_scan is False
        default = recon.reconstruct(verbose=False).obj.copy()
        normalized = recon.reconstruct(weight_normalize=True, verbose=False).obj
        assert np.array_equal(default, normalized)


class TestPadBoundary:
    """`boundary="pad"` grows the canvas instead of wrapping."""

    def test_interior_matches_wrap(self, dataset4d):
        defocus = integer_shift_defocus(1)
        _, montage = _build_pair(dataset4d, defocus)

        recon_kwargs = dict(
            deconvolution_kernel="prlx",
            parallax_flip_phase=False,
            weight_normalize=False,
            verbose=False,
        )
        montage.reconstruct(boundary="wrap", **recon_kwargs)
        wrapped = montage.obj

        shifts, _ = montage._return_shifts_px(0.0, {"C10": defocus}, montage.bf_mask, 1)
        margin = int(np.ceil(float(shifts.abs().max()))) + 1

        montage.reconstruct(boundary="pad", **recon_kwargs)
        padded = montage.obj
        row0, col0 = (-to_numpy(montage._canvas_origin_px)).astype(int)

        # far enough from every edge, the wrap modulo is a no-op and the two agree exactly
        n_rows, n_cols = wrapped.shape
        interior_wrap = wrapped[margin : n_rows - margin, margin : n_cols - margin]
        interior_pad = padded[
            row0 + margin : row0 + n_rows - margin, col0 + margin : col0 + n_cols - margin
        ]

        assert interior_pad.shape == interior_wrap.shape
        assert _relative_error(interior_pad, interior_wrap) < 1e-5

    def test_canvas_covers_the_shifted_positions(self, dataset4d):
        defocus = integer_shift_defocus(2)
        _, montage = _build_pair(dataset4d, defocus)
        montage.reconstruct(boundary="pad", verbose=False)

        # nothing was dropped: every (BF pixel, position) pair landed on the canvas
        assert float(montage.weights.sum()) == pytest.approx(
            montage.num_bf * montage.num_positions, rel=1e-6
        )
        assert montage.obj.shape[0] > N and montage.obj.shape[1] > N

    def test_pad_px_freezes_the_canvas(self, dataset4d):
        _, montage = _build_pair(dataset4d, integer_shift_defocus(1))

        shapes = set()
        for pixel_shift in (1, 2):
            montage.reconstruct(
                override_aberration_coefs={"C10": integer_shift_defocus(pixel_shift)},
                boundary="pad",
                pad_px=12,
                verbose=False,
            )
            shapes.add(montage.obj.shape)

        assert len(shapes) == 1, f"canvas resized across trials: {shapes}"


class TestNonGridScan:
    """`from_dataset3d` with raster positions must reproduce the gridded path."""

    @staticmethod
    def _dataset3d_and_positions(dataset4d):
        n_scan = dataset4d.shape[0]
        dataset3d = Dataset3d.from_array(
            dataset4d.array.reshape(-1, N, N),
            name="synthetic 3D stack",
            units=("index", "A^-1", "A^-1"),
            sampling=(1, RECIPROCAL_SAMPLING, RECIPROCAL_SAMPLING),
        )
        positions = scan_positions_px()[: n_scan * n_scan] * SCAN_SAMPLING
        return dataset3d, positions

    def test_matches_from_dataset4d(self, dataset4d):
        defocus = integer_shift_defocus(1)
        dataset3d, positions = self._dataset3d_and_positions(dataset4d)

        gridded = ShadowMontagePtychography.from_dataset4d(
            dataset4d, edge_blend_pixels=0, boundary="wrap", **_common_kwargs(defocus)
        )
        ungridded = ShadowMontagePtychography.from_dataset3d(
            dataset3d,
            positions,
            scan_sampling=(SCAN_SAMPLING, SCAN_SAMPLING),
            boundary="wrap",
            **_common_kwargs(defocus),
        )

        recon_kwargs = dict(
            deconvolution_kernel="prlx",
            parallax_flip_phase=False,
            weight_normalize=False,
            verbose=False,
        )
        gridded.reconstruct(**recon_kwargs)
        ungridded.reconstruct(**recon_kwargs)

        assert ungridded.obj.shape == gridded.obj.shape
        assert _relative_error(ungridded.obj, gridded.obj) < 1e-5

    def test_position_axis_order_is_row_col(self, dataset4d):
        """Swapping the position columns must transpose the reconstruction."""
        defocus = integer_shift_defocus(1)
        dataset3d, positions = self._dataset3d_and_positions(dataset4d)

        recon_kwargs = dict(
            deconvolution_kernel="prlx",
            parallax_flip_phase=False,
            weight_normalize=False,
            verbose=False,
        )
        common = dict(
            scan_sampling=(SCAN_SAMPLING, SCAN_SAMPLING),
            boundary="wrap",
            **_common_kwargs(defocus),
        )

        straight = ShadowMontagePtychography.from_dataset3d(dataset3d, positions, **common)
        swapped = ShadowMontagePtychography.from_dataset3d(
            dataset3d, positions[:, ::-1].copy(), **common
        )
        straight.reconstruct(**recon_kwargs)
        swapped.reconstruct(**recon_kwargs)

        assert not np.allclose(straight.obj, swapped.obj)

    def test_scattered_positions_reconstruct(self, dataset4d):
        """A jittered, shuffled scan still produces a supported montage."""
        defocus = integer_shift_defocus(1)
        dataset3d, positions = self._dataset3d_and_positions(dataset4d)

        rng = np.random.default_rng(0)
        order = rng.permutation(len(positions))
        jittered = positions[order] + rng.normal(
            scale=0.25 * SCAN_SAMPLING, size=(len(positions), 2)
        )
        shuffled = Dataset3d.from_array(
            dataset3d.array[order],
            name="shuffled",
            units=dataset3d.units,
            sampling=dataset3d.sampling,
        )

        recon = ShadowMontagePtychography.from_dataset3d(
            shuffled,
            jittered,
            scan_sampling=(SCAN_SAMPLING, SCAN_SAMPLING),
            boundary="pad",
            **_common_kwargs(defocus),
        )
        recon.reconstruct(parallax_flip_phase=False, verbose=False)

        assert np.isfinite(recon.obj).all()
        assert float(recon.weights.max()) > 0

    def test_auto_scan_sampling_warns_and_infers(self, dataset4d):
        dataset3d, positions = self._dataset3d_and_positions(dataset4d)

        with pytest.warns(UserWarning, match="Inferred scan_sampling"):
            recon = ShadowMontagePtychography.from_dataset3d(
                dataset3d,
                positions,
                scan_sampling="auto",
                **_common_kwargs(integer_shift_defocus(1)),
            )

        assert recon.scan_sampling[0] == pytest.approx(SCAN_SAMPLING, rel=1e-6)

    def test_accepts_a_dataset2d_of_positions(self, dataset4d):
        dataset3d, positions = self._dataset3d_and_positions(dataset4d)
        positions_dataset = Dataset2d.from_array(positions, name="positions", units=("A", "A"))

        recon = ShadowMontagePtychography.from_dataset3d(
            dataset3d,
            positions_dataset,
            scan_sampling=(SCAN_SAMPLING, SCAN_SAMPLING),
            **_common_kwargs(integer_shift_defocus(1)),
        )
        assert recon.num_positions == len(positions)

    def test_rejects_positions_in_the_wrong_units(self, dataset4d):
        dataset3d, positions = self._dataset3d_and_positions(dataset4d)
        positions_dataset = Dataset2d.from_array(
            positions, name="positions", units=("pixels", "pixels")
        )

        with pytest.raises(ValueError, match="must be given in 'A'"):
            ShadowMontagePtychography.from_dataset3d(
                dataset3d,
                positions_dataset,
                scan_sampling=(SCAN_SAMPLING, SCAN_SAMPLING),
                **_common_kwargs(integer_shift_defocus(1)),
            )

    def test_rejects_mismatched_position_count(self, dataset4d):
        dataset3d, positions = self._dataset3d_and_positions(dataset4d)

        with pytest.raises(ValueError, match="rows but `dataset` has"):
            ShadowMontagePtychography.from_dataset3d(
                dataset3d,
                positions[:-1],
                scan_sampling=(SCAN_SAMPLING, SCAN_SAMPLING),
                **_common_kwargs(integer_shift_defocus(1)),
            )

    def test_requires_an_explicit_rotation_angle(self, dataset4d):
        dataset3d, positions = self._dataset3d_and_positions(dataset4d)
        kwargs = _common_kwargs(integer_shift_defocus(1))
        kwargs["rotation_angle"] = None

        with pytest.raises(ValueError, match="must be given for non-raster scans"):
            ShadowMontagePtychography.from_dataset3d(
                dataset3d,
                positions,
                scan_sampling=(SCAN_SAMPLING, SCAN_SAMPLING),
                **kwargs,
            )


class TestHyperparameterSearch:
    def test_grid_search_recovers_the_seeded_defocus(self, dataset4d):
        true_defocus = integer_shift_defocus(1)
        _, montage = _build_pair(dataset4d, true_defocus)

        montage.grid_search_hyperparameters(
            aberration_coefs={
                "C10": OptimizationParameter(
                    low=0.4 * true_defocus, high=1.6 * true_defocus, n_points=7
                )
            },
            parallax_flip_phase=False,
            verbose=False,
        )
        best = montage.hyperparameter_state.optimized_aberrations["C10"]

        step = (1.6 - 0.4) * true_defocus / 6
        assert abs(best - true_defocus) <= step

    def test_variance_loss_is_minimized_at_the_true_defocus(self, dataset4d):
        true_defocus = integer_shift_defocus(1)
        _, montage = _build_pair(dataset4d, true_defocus)

        losses = {}
        for scale in (0.5, 0.8, 1.0, 1.2, 1.5):
            montage.reconstruct(
                override_aberration_coefs={"C10": scale * true_defocus},
                parallax_flip_phase=False,
                verbose=False,
            )
            losses[scale] = float(montage.variance_loss())

        assert all(value > 0 for value in losses.values())
        assert min(losses, key=losses.get) == pytest.approx(1.0)


class TestSerialization:
    """Both classes must survive a save/load round-trip and stay usable afterwards."""

    @pytest.mark.parametrize("cls_name", ["fourier", "montage"])
    def test_round_trip(self, dataset4d, tmp_path, cls_name):
        defocus = integer_shift_defocus(1)
        fourier, montage = _build_pair(dataset4d, defocus)
        recon = fourier if cls_name == "fourier" else montage

        recon.hyperparameter_state.optimized_aberrations = {"C10": 123.0}
        recon.hyperparameter_state.optimized_rotation_angle = 7.5
        recon.reconstruct(deconvolution_kernel="prlx", verbose=False)
        before = recon.obj.copy()

        path = str(tmp_path / f"{cls_name}.zip")
        recon.save(path, mode="o")
        restored = load(path)

        assert type(restored) is type(recon)
        assert np.array_equal(restored.obj, before)
        assert restored.hyperparameter_state.optimized_aberrations == {"C10": 123.0}
        assert restored.hyperparameter_state.optimized_rotation_angle == 7.5
        assert restored.gpts == recon.gpts
        assert float(restored.variance_loss()) == pytest.approx(float(recon.variance_loss()))

        # and it must still be able to reconstruct
        restored.reconstruct(deconvolution_kernel="prlx", verbose=False)
        assert np.allclose(restored.obj, before)

    def test_torch_size_attributes_round_trip_as_tuples(self, dataset4d):
        """`torch.Size` is a tuple subclass; AutoSerialize used to choke on the subclass name."""
        _, montage = _build_pair(dataset4d, integer_shift_defocus(1))

        assert type(montage.gpts) is tuple
        assert type(montage.scan_gpts) is tuple


class TestVisualization:
    """The object sampling, and hence `visualize`'s scalebar, must follow the upsampling."""

    @pytest.mark.parametrize("cls_name", ["fourier", "montage"])
    @pytest.mark.parametrize("upsampling_factor", [1, 2, 3])
    def test_scalebar_follows_the_upsampling(self, dataset4d, cls_name, upsampling_factor):
        fourier, montage = _build_pair(dataset4d, integer_shift_defocus(1, upsampling_factor))
        recon = fourier if cls_name == "fourier" else montage
        recon.reconstruct(
            deconvolution_kernel="prlx", upsampling_factor=upsampling_factor, verbose=False
        )

        expected = SCAN_SAMPLING / upsampling_factor
        assert recon._obj_sampling[0] == pytest.approx(expected)
        # the reported sampling must span the same field of view as the image itself
        assert recon.obj.shape[0] * recon._obj_sampling[0] == pytest.approx(N * SCAN_SAMPLING)

    def test_sampling_resets_between_reconstructions(self, dataset4d):
        _, montage = _build_pair(dataset4d, integer_shift_defocus(1))

        montage.reconstruct(upsampling_factor=3, verbose=False)
        assert montage._obj_sampling[0] == pytest.approx(SCAN_SAMPLING / 3)
        montage.reconstruct(upsampling_factor=1, verbose=False)
        assert montage._obj_sampling[0] == pytest.approx(SCAN_SAMPLING)

    def test_sampling_is_defined_before_reconstructing(self, dataset4d):
        fourier, montage = _build_pair(dataset4d, integer_shift_defocus(1))

        for recon in (fourier, montage):
            assert recon._obj_sampling[0] == pytest.approx(SCAN_SAMPLING)

    def test_fov_matches_the_scan_extent(self, dataset4d):
        fourier, montage = _build_pair(dataset4d, integer_shift_defocus(1))

        for recon in (fourier, montage):
            assert recon.fov == pytest.approx((N * SCAN_SAMPLING, N * SCAN_SAMPLING))

    @pytest.mark.parametrize("upsampling_factor", [1, 2])
    def test_padded_canvas_keeps_the_scan_sampling(self, dataset4d, upsampling_factor):
        """A padded canvas spans more than the scan, so its sampling is not fov/shape.

        Deriving from the *scan* field of view would under-report the pixel size by the
        padding fraction; `_obj_fov` reports the canvas extent instead.
        """
        _, montage = _build_pair(dataset4d, integer_shift_defocus(1, upsampling_factor))
        montage.reconstruct(boundary="pad", upsampling_factor=upsampling_factor, verbose=False)

        assert montage.obj.shape[0] > N * upsampling_factor  # canvas really did grow
        assert montage._obj_sampling[0] == pytest.approx(SCAN_SAMPLING / upsampling_factor)
        assert montage._obj_fov[0] > montage.fov[0]
        # and the reported extent still matches the image it describes
        assert montage.obj.shape[0] * montage._obj_sampling[0] == pytest.approx(
            montage._obj_fov[0]
        )

    def test_wrapped_canvas_spans_exactly_the_scan(self, dataset4d):
        _, montage = _build_pair(dataset4d, integer_shift_defocus(1))
        montage.reconstruct(boundary="wrap", upsampling_factor=2, verbose=False)

        assert montage._obj_fov == pytest.approx(montage.fov)

    def test_visualize_before_reconstruct_raises(self, dataset4d):
        _, montage = _build_pair(dataset4d, integer_shift_defocus(1))
        with pytest.raises(RuntimeError, match="Run reconstruct"):
            montage.visualize()


class TestSemiangleCutoff:
    """`semiangle_cutoff` sets the probe aperture and is never optional."""

    @pytest.mark.parametrize("cls", [DirectPtychography, ShadowMontagePtychography])
    def test_from_virtual_bfs_requires_it(self, cls):
        signature = inspect.signature(cls.from_virtual_bfs)
        parameter = signature.parameters["semiangle_cutoff"]

        assert parameter.default is inspect.Parameter.empty

    @pytest.mark.parametrize("cls", [DirectPtychography, ShadowMontagePtychography])
    def test_none_raises_a_clear_error(self, dataset4d, cls):
        with pytest.raises(ValueError, match="`semiangle_cutoff` is required"):
            cls.from_dataset4d(
                dataset4d,
                energy=PROBE_ENERGY,
                semiangle_cutoff=None,
                rotation_angle=0.0,
                force_fitted_origin=ORIGIN,
                verbose=False,
            )

    @pytest.mark.parametrize("cls", [DirectPtychography, ShadowMontagePtychography])
    def test_non_positive_raises(self, dataset4d, cls):
        with pytest.raises(ValueError):
            cls.from_dataset4d(
                dataset4d,
                energy=PROBE_ENERGY,
                semiangle_cutoff=-1.0,
                rotation_angle=0.0,
                force_fitted_origin=ORIGIN,
                verbose=False,
            )
