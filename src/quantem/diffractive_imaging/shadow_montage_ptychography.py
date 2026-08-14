import gc
import math
import warnings
from typing import TYPE_CHECKING, Literal, Tuple

import numpy as np
from numpy.typing import NDArray
from tqdm.auto import tqdm

from quantem.core import config
from quantem.core.datastructures import Dataset2d, Dataset3d, Dataset4d
from quantem.core.utils.utils import electron_wavelength_angstrom
from quantem.core.utils.validators import validate_tensor
from quantem.diffractive_imaging.complex_probe import (
    aberration_surface,
    aberration_surface_cartesian_gradients,
    evaluate_probe,
    polar_coordinates,
    spatial_frequencies,
)
from quantem.diffractive_imaging.ptycho_utils import SimpleBatcher

if TYPE_CHECKING:
    import torch
else:
    if config.get("has_torch"):
        import torch

from quantem.diffractive_imaging.direct_ptycho_utils import (
    _crop_corner_centered_mask,
    _rotation_degrees_to_radians,
    allocate_splat_buffers,
    bf_mask_from_mean_pattern,
    build_vbf_stack_from_dataset4d,
    fit_and_shift_diffraction_origin,
    normalize_vbf_stack,
    scatter_add_splat,
)
from quantem.diffractive_imaging.direct_ptychography_base import DirectPtychographyBase

# target number of (BF pixel, scan position) points per splat batch
_DEFAULT_POINTS_PER_BATCH = 4_194_304


class ShadowMontagePtychography(DirectPtychographyBase):
    """
    Real-space ("shadow montage") direct ptychography.

    Each virtual bright-field image is translated by its own aberration-dependent lateral
    shift and accumulated onto a shared canvas -- the real-space dual of the ``"parallax"``
    (tilt-corrected bright field) kernel of
    :class:`~quantem.diffractive_imaging.direct_ptychography.DirectPtychography`.

    The two formulations are equivalent: the parallax Fourier multiplier
    ``exp(-1j * grad_chi . q)`` is exactly a translation by ``grad_chi / (2 * pi)`` Angstrom,
    and Fourier-space tiling by ``U`` is exactly real-space zero-insertion at every ``U``-th
    pixel. Working in real space instead buys two things:

    - no scan-space FFT is needed, so the scan positions need not lie on a grid --
      see :meth:`from_dataset3d`;
    - the phase-flip and Butterworth filters, which do not depend on the bright-field index,
      collapse into a single post-hoc filter on the finished image.

    Only the parallax kernel is available here. SSB, OBF and matched-filter deconvolutions
    have bright-field-dependent Fourier multipliers that are not translations, so they
    cannot be expressed as a real-space montage; use ``DirectPtychography`` for those.

    Instantiate with :meth:`from_dataset4d`, :meth:`from_virtual_bfs` or
    :meth:`from_dataset3d`.
    """

    _token = object()

    def __init__(
        self,
        vbf_stack: torch.Tensor | NDArray,
        positions_px: torch.Tensor | NDArray,
        bf_mask_dataset: Dataset2d,
        energy: float,
        rotation_angle: float,
        aberration_coefs: dict,
        semiangle_cutoff: float,
        scan_sampling: Tuple[float, float],
        scan_units: Tuple[str, str],
        scan_gpts: Tuple[int, int],
        boundary: Literal["wrap", "pad"],
        subtract_frame_mean: bool,
        soft_edges: bool,
        crop_bf_mask: bool,
        bf_mask_padding_px: int,
        rng: np.random.Generator | int | None,
        device: str | int,
        verbose: int | bool,
        _token: object | None = None,
    ):
        """ """
        if _token is not self._token:
            raise RuntimeError(
                "Use ShadowMontagePtychography.from_dataset4d(), .from_virtual_bfs() or "
                ".from_dataset3d() to instantiate this class."
            )

        self.device = device
        self.verbose = verbose
        self.vbf_stack = vbf_stack
        self.positions_px = positions_px
        self.bf_mask = bf_mask_dataset.array  # ty:ignore[invalid-assignment]
        if crop_bf_mask:
            self.bf_mask = _crop_corner_centered_mask(self.bf_mask, bf_mask_padding_px)

        self.wavelength = electron_wavelength_angstrom(energy)
        self.scan_units = scan_units
        self.detector_units = bf_mask_dataset.units

        self.scan_gpts = tuple(int(n) for n in scan_gpts)
        self.scan_sampling = scan_sampling
        self.reciprocal_sampling = bf_mask_dataset.sampling
        self.angular_sampling = tuple(d * 1e3 * self.wavelength for d in self.reciprocal_sampling)

        self.num_bf = int(self.vbf_stack.shape[0])
        self.num_positions = int(self.vbf_stack.shape[1])
        self.gpts = tuple(int(n) for n in self.bf_mask.shape[:2])
        self.sampling = tuple(1 / s / n for n, s in zip(self.reciprocal_sampling, self.gpts))

        self.semiangle_cutoff = semiangle_cutoff
        self.soft_edges = soft_edges
        self.boundary = boundary
        self.subtract_frame_mean = subtract_frame_mean
        self.rng = rng

        if self.positions_px.shape[0] != self.num_positions:
            raise ValueError(
                f"`positions_px` has {self.positions_px.shape[0]} rows but `vbf_stack` has "
                f"{self.num_positions} scan positions."
            )

        self.hyperparameter_state = self._make_hyperparameter_state(
            aberration_coefs, rotation_angle
        )

        self._preprocess()

    @staticmethod
    def _make_hyperparameter_state(aberration_coefs, rotation_angle):
        from quantem.diffractive_imaging.direct_ptychography_base import HyperparameterState

        return HyperparameterState(
            initial_aberrations=aberration_coefs, initial_rotation_angle=rotation_angle
        )

    # ------------------------------------------------------------------
    # constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_virtual_bfs(
        cls,
        vbf_dataset: Dataset3d,
        bf_mask_dataset: Dataset2d,
        energy: float,
        rotation_angle: float,
        semiangle_cutoff: float,
        aberration_coefs: dict = {},
        boundary: Literal["wrap", "pad"] = "wrap",
        subtract_frame_mean: bool = False,
        soft_edges: bool = True,
        crop_bf_mask: bool = True,
        bf_mask_padding_px: int = 1,
        rng: np.random.Generator | int | None = None,
        device: str | int = "cpu",
        verbose: int | bool = True,
    ):
        """
        Build from a gridded virtual bright-field stack.

        Accepts exactly the ``(N_bf, Rx, Ry)`` ``Dataset3d`` that
        :meth:`DirectPtychography.from_virtual_bfs` takes, so the same stack can be fed to
        both classes; the trailing scan axes are flattened internally.
        """
        scan_gpts = tuple(int(n) for n in vbf_dataset.shape[-2:])
        vbf_stack = np.asarray(vbf_dataset.array).reshape(vbf_dataset.shape[0], -1)

        return cls(
            vbf_stack=vbf_stack,
            positions_px=cls._raster_positions_px(scan_gpts),
            bf_mask_dataset=bf_mask_dataset,
            energy=energy,
            rotation_angle=rotation_angle,
            aberration_coefs=aberration_coefs,
            semiangle_cutoff=semiangle_cutoff,
            scan_sampling=tuple(vbf_dataset.sampling[-2:]),
            scan_units=tuple(vbf_dataset.units[-2:]),
            scan_gpts=scan_gpts,
            boundary=boundary,
            subtract_frame_mean=subtract_frame_mean,
            soft_edges=soft_edges,
            crop_bf_mask=crop_bf_mask,
            bf_mask_padding_px=bf_mask_padding_px,
            rng=rng,
            device=device,
            verbose=verbose,
            _token=cls._token,
        )

    @classmethod
    def from_dataset4d(
        cls,
        dataset: Dataset4d,
        energy: float,
        semiangle_cutoff: float,
        aberration_coefs: dict = {},
        rotation_angle: float | None = None,
        max_batch_size: int | None = None,
        fit_method: str = "plane",
        mode: str = "bilinear",
        force_measured_origin: Tuple[float, float] | torch.Tensor | NDArray | None = None,
        force_fitted_origin: Tuple[float, float] | torch.Tensor | NDArray | None = None,
        intensity_threshold: float = 0.5,
        boundary: Literal["wrap", "pad"] = "wrap",
        subtract_frame_mean: bool = False,
        soft_edges: bool = True,
        crop_bf_mask: bool = True,
        bf_mask_padding_px: int = 1,
        rng: np.random.Generator | int | None = None,
        device: str | int = "cpu",
        verbose: int | bool = True,
        normalization_order: int = 0,
        edge_blend_pixels: int = 0,
    ):
        """
        Build from a raster-scanned 4D-STEM dataset.

        Runs the same origin-correction, bright-field masking and normalization pipeline as
        :meth:`DirectPtychography.from_dataset4d` (they share
        :func:`~quantem.diffractive_imaging.direct_ptycho_utils.build_vbf_stack_from_dataset4d`),
        then flattens the scan axes onto an integer position grid.
        """
        vbf_dataset, bf_mask_dataset, rotation_angle = build_vbf_stack_from_dataset4d(
            dataset,
            device=device,
            max_batch_size=max_batch_size,
            fit_method=fit_method,
            mode=mode,
            force_measured_origin=force_measured_origin,
            force_fitted_origin=force_fitted_origin,
            rotation_angle=rotation_angle,
            intensity_threshold=intensity_threshold,
            normalization_order=normalization_order,
            edge_blend_pixels=edge_blend_pixels,
        )

        return cls.from_virtual_bfs(
            vbf_dataset=vbf_dataset,
            bf_mask_dataset=bf_mask_dataset,
            energy=energy,
            rotation_angle=rotation_angle,
            semiangle_cutoff=semiangle_cutoff,
            aberration_coefs=aberration_coefs,
            boundary=boundary,
            subtract_frame_mean=subtract_frame_mean,
            soft_edges=soft_edges,
            crop_bf_mask=crop_bf_mask,
            bf_mask_padding_px=bf_mask_padding_px,
            rng=rng,
            device=device,
            verbose=verbose,
        )

    @classmethod
    def from_dataset3d(
        cls,
        dataset: Dataset3d,
        positions: Dataset2d | torch.Tensor | NDArray,
        energy: float,
        semiangle_cutoff: float,
        rotation_angle: float,
        scan_sampling: Tuple[float, float] | Literal["auto"],
        aberration_coefs: dict = {},
        max_batch_size: int | None = None,
        fit_method: str = "plane",
        mode: str = "bilinear",
        force_measured_origin: Tuple[float, float] | torch.Tensor | NDArray | None = None,
        force_fitted_origin: Tuple[float, float] | torch.Tensor | NDArray | None = None,
        intensity_threshold: float = 0.5,
        boundary: Literal["wrap", "pad"] = "pad",
        subtract_frame_mean: bool = False,
        soft_edges: bool = True,
        crop_bf_mask: bool = True,
        bf_mask_padding_px: int = 1,
        rng: np.random.Generator | int | None = None,
        device: str | int = "cpu",
        verbose: int | bool = True,
        normalization_order: int = 0,
    ):
        """
        Build from an ungridded stack of diffraction patterns and their probe positions.

        Parameters
        ----------
        dataset : Dataset3d
            ``(N, Qx, Qy)`` diffraction patterns, reciprocal units ``"A^-1"`` or ``"mrad"``.
        positions : Dataset2d, torch.Tensor or ndarray
            ``(N, 2)`` probe positions in Angstrom, ordered ``(row, col)`` to match the
            diffraction axes. A ``Dataset2d`` must carry units ``"A"``.
        rotation_angle : float
            Detector rotation in degrees. Required: rotation is otherwise estimated from the
            curl of the center of mass over a 2D scan grid, which an ungridded scan lacks.
        scan_sampling : tuple of float or "auto"
            Canvas pixel size in Angstrom. ``"auto"`` uses the median nearest-neighbour
            position spacing and warns with the inferred value.

        Notes
        -----
        Positions are *not* rotated: the detector rotation already enters through the
        bright-field k-grid, and rotating the positions as well would double-count it.
        """
        positions_ang = cls._validate_positions(positions)
        if positions_ang.shape[0] != dataset.shape[0]:
            raise ValueError(
                f"`positions` has {positions_ang.shape[0]} rows but `dataset` has "
                f"{dataset.shape[0]} diffraction patterns."
            )

        if isinstance(scan_sampling, str):
            if scan_sampling != "auto":
                raise ValueError(
                    f"`scan_sampling` must be a pair or 'auto', got {scan_sampling!r}"
                )
            scan_sampling = cls._infer_scan_sampling(positions_ang)
            warnings.warn(
                f"Inferred scan_sampling={scan_sampling} Angstrom from the median "
                "nearest-neighbour position spacing.",
                stacklevel=2,
            )
        scan_sampling = tuple(float(s) for s in scan_sampling)

        shifted_tensor, rotation_angle = fit_and_shift_diffraction_origin(
            dataset,
            device=device,
            max_batch_size=max_batch_size,
            fit_method=fit_method,
            mode=mode,
            force_measured_origin=force_measured_origin,
            force_fitted_origin=force_fitted_origin,
            rotation_angle=rotation_angle,
            probe_positions=positions_ang,
        )

        bf_mask = bf_mask_from_mean_pattern(shifted_tensor, intensity_threshold)
        bf_mask_dataset = Dataset2d.from_array(
            bf_mask.cpu().numpy(),
            name="BF mask",
            units=dataset.units[-2:],
            sampling=dataset.sampling[-2:],
        )

        if normalization_order != 0:
            raise ValueError(
                "`normalization_order=1` fits a 2D linear background per bright-field image "
                "and needs a scan grid, which an ungridded scan does not have; use "
                "`normalization_order=0`."
            )

        vbf_stack = shifted_tensor[..., bf_mask].cpu()  # (N, N_bf)
        vbf_stack = normalize_vbf_stack(vbf_stack, normalization_order, vbf_stack.shape[:1])
        vbf_stack = vbf_stack.T.contiguous()  # (N_bf, N)

        # anchor to the position bounding box, then convert to canvas pixels
        positions_px = (positions_ang - positions_ang.min(axis=0)) / np.asarray(scan_sampling)
        scan_gpts = tuple(int(math.ceil(v)) + 1 for v in positions_px.max(axis=0))

        return cls(
            vbf_stack=vbf_stack,
            positions_px=positions_px,
            bf_mask_dataset=bf_mask_dataset,
            energy=energy,
            rotation_angle=rotation_angle,
            aberration_coefs=aberration_coefs,
            semiangle_cutoff=semiangle_cutoff,
            scan_sampling=scan_sampling,
            scan_units=("A", "A"),
            scan_gpts=scan_gpts,
            boundary=boundary,
            subtract_frame_mean=subtract_frame_mean,
            soft_edges=soft_edges,
            crop_bf_mask=crop_bf_mask,
            bf_mask_padding_px=bf_mask_padding_px,
            rng=rng,
            device=device,
            verbose=verbose,
            _token=cls._token,
        )

    @staticmethod
    def _raster_positions_px(scan_gpts: Tuple[int, int]) -> NDArray:
        """Integer ``(Rx*Ry, 2)`` raster positions in scan pixels, "ij" ordered."""
        ii, jj = np.meshgrid(np.arange(scan_gpts[0]), np.arange(scan_gpts[1]), indexing="ij")
        return np.stack((ii.ravel(), jj.ravel()), axis=-1).astype(np.float64)

    @staticmethod
    def _validate_positions(positions) -> NDArray:
        if isinstance(positions, Dataset2d):
            if str(positions.units[0]) != "A":
                raise ValueError(
                    f"`positions` must be given in 'A', got {tuple(positions.units)!r}"
                )
            positions = positions.array
        positions = np.asarray(
            positions.detach().cpu().numpy() if hasattr(positions, "detach") else positions,
            dtype=np.float64,
        )
        if positions.ndim != 2 or positions.shape[1] != 2:
            raise ValueError(f"`positions` must have shape (N, 2), got {positions.shape}")
        return positions

    @staticmethod
    def _infer_scan_sampling(
        positions_ang: NDArray, max_points: int = 4096
    ) -> Tuple[float, float]:
        """Median nearest-neighbour spacing, isotropic, from a subsample of positions."""
        pts = positions_ang
        if len(pts) > max_points:
            pts = pts[np.linspace(0, len(pts) - 1, max_points).astype(int)]
        dists = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=-1)
        np.fill_diagonal(dists, np.inf)
        spacing = float(np.median(dists.min(axis=1)))
        return (spacing, spacing)

    # ------------------------------------------------------------------
    # properties
    # ------------------------------------------------------------------

    @property
    def vbf_stack(self) -> torch.Tensor:
        """``(N_bf, N_pos)`` virtual bright-field stack, flattened over scan positions."""
        return self._vbf_stack

    @vbf_stack.setter
    def vbf_stack(self, value):
        stack = validate_tensor(value, "vbf_stack", dtype=torch.float).to(device=self.device)
        if stack.ndim != 2:
            raise ValueError(
                f"`vbf_stack` must have shape (N_bf, N_pos), got {tuple(stack.shape)}"
            )
        self._vbf_stack = stack

    @property
    def positions_px(self) -> torch.Tensor:
        """``(N_pos, 2)`` scan positions in canvas pixels at ``upsampling_factor=1``."""
        return self._positions_px

    @positions_px.setter
    def positions_px(self, value):
        positions = validate_tensor(value, "positions_px", dtype=torch.float64).to(
            device=self.device
        )
        if positions.ndim != 2 or positions.shape[1] != 2:
            raise ValueError(
                f"`positions_px` must have shape (N_pos, 2), got {tuple(positions.shape)}"
            )
        self._positions_px = positions

    @property
    def corrected_bf(self) -> torch.Tensor | None:
        """Reconstructed phase image, or ``None`` before :meth:`reconstruct`."""
        return self._corrected_bf

    @property
    def weights(self) -> torch.Tensor | None:
        """Accumulated splat weight per canvas pixel -- the montage's local support."""
        if self._sum_w is None:
            return None
        return self._sum_w.reshape(self._canvas_shape)

    @property
    def variance_map(self) -> torch.Tensor | None:
        """Per-pixel variance across bright-field images (see :meth:`variance_loss`)."""
        if self._sum_wv2 is None:
            return None
        _, var, _ = self._weighted_moments()
        return var.reshape(self._canvas_shape)

    @property
    def _obj_fov(self) -> tuple[float, float]:
        """Field of view of the canvas, in Angstrom.

        With ``boundary="pad"`` the canvas grows past the scan to cover the shifted
        positions, so it spans more than :attr:`fov`. Computed by :meth:`_return_canvas`
        alongside the canvas shape, so the two always agree.
        """
        if self._canvas_fov is None:
            return self.fov
        return self._canvas_fov

    # ------------------------------------------------------------------
    # preprocessing
    # ------------------------------------------------------------------

    def _preprocess(self):
        """
        Remove the scan mean of each bright-field image.

        This is the real-space equivalent of zeroing the DC bin of each image's scan-space
        Fourier transform, which is what ``DirectPtychography._preprocess`` does. It also
        centers the accumulated values on zero, which keeps the ``E[v^2] - E[v]^2`` variance
        accumulation well conditioned.
        """
        self._dc_per_image = self._vbf_stack.mean(dim=1)
        self._vbf_stack = self._vbf_stack - self._dc_per_image[:, None]

        if self.subtract_frame_mean:
            self._vbf_stack = self._vbf_stack - self._vbf_stack.mean(dim=0, keepdim=True)

        self._reset_reconstruction()
        return self

    def _reset_reconstruction(self):
        self._sum_w = None
        self._sum_wv = None
        self._sum_wv2 = None
        self._corrected_bf = None
        self._canvas_shape = None
        self._canvas_origin_px = None
        self._canvas_fov = None
        self._bf_weights = None

    # ------------------------------------------------------------------
    # reconstruction
    # ------------------------------------------------------------------

    def _return_shifts_px(self, rotation_angle, aberration_coefs, bf_mask, upsampling_factor):
        """``(num_bf, 2)`` parallax shifts in upsampled canvas pixels, plus the BF weight."""
        kxa, kya = spatial_frequencies(
            self.gpts,
            self.sampling,
            rotation_angle=_rotation_degrees_to_radians(rotation_angle),
            device=self.device,
        )
        k, phi = polar_coordinates(kxa, kya)

        dx, dy = aberration_surface_cartesian_gradients(
            k * self.wavelength,
            phi,
            aberration_coefs=aberration_coefs,
        )
        grad_k = torch.stack((dx[bf_mask], dy[bf_mask]), -1)

        upsampled_sampling = torch.as_tensor(
            [s / upsampling_factor for s in self.scan_sampling],
            device=self.device,
            dtype=torch.float64,
        )
        shifts_px = grad_k.to(torch.float64) / (2 * math.pi) / upsampled_sampling

        # matches DirectPtychography.reconstruct: soft_edges is left at evaluate_probe's
        # default rather than taking self.soft_edges, so the two normalizations agree
        cmplx_probe_k = evaluate_probe(
            k * self.wavelength,
            phi,
            self.semiangle_cutoff,
            self.angular_sampling,
            self.wavelength,
            aberration_coefs=aberration_coefs,
        )
        bf_weights = cmplx_probe_k[bf_mask].abs().square().sum()

        return shifts_px, bf_weights

    def _return_canvas(self, shifts_px, upsampling_factor, boundary, pad_px):
        """``(canvas_shape, canvas_origin_px, canvas_fov)`` for the requested boundary.

        The field of view is returned alongside the shape, rather than recomputed later,
        so the two cannot disagree about the upsampling factor.
        """
        positions_up = self._positions_px * upsampling_factor

        def with_fov(canvas_shape, origin):
            canvas_fov = tuple(
                n * s / upsampling_factor for n, s in zip(canvas_shape, self.scan_sampling)
            )
            return canvas_shape, origin, canvas_fov

        if boundary == "wrap":
            # spans exactly the scan field of view, at any upsampling factor
            canvas_shape = tuple(int(n) * upsampling_factor for n in self.scan_gpts)
            origin = torch.zeros(2, device=self.device, dtype=torch.float64)
            return with_fov(canvas_shape, origin)

        if boundary != "pad":
            raise ValueError(f"`boundary` must be 'wrap' or 'pad', got {boundary!r}")

        if pad_px is None:
            lo = torch.floor(positions_up.amin(0) + shifts_px.amin(0))
            hi = torch.ceil(positions_up.amax(0) + shifts_px.amax(0))
        else:
            lo = torch.floor(positions_up.amin(0)) - pad_px
            hi = torch.ceil(positions_up.amax(0)) + pad_px

        # +2 leaves room for the upper bilinear corner at the far edge
        canvas_shape = tuple(int(v) + 2 for v in (hi - lo))
        return with_fov(canvas_shape, lo)

    def reconstruct(
        self,
        bf_mask=None,
        override_aberration_coefs=None,
        upsampling_factor=None,
        override_rotation_angle=None,
        max_batch_size=None,
        deconvolution_kernel="parallax",
        q_highpass=None,
        q_lowpass=None,
        butterworth_order=12,
        parallax_flip_phase=True,
        verbose=None,
        use_initial_state=False,
        boundary=None,
        interpolation="bilinear",
        weight_normalize=None,
        weight_threshold=1e-2,
        pad_px=None,
        compute_variance=True,
        suppress_nyquist=False,
    ):
        """
        Accumulate the shadow montage and apply the post-hoc Fourier filters.

        Parameters
        ----------
        bf_mask : torch.Tensor, optional
            Subset of the bright-field mask to use. Must be strictly smaller than the mask
            used at initialization.
        override_aberration_coefs : dict, optional
            Aberration coefficients, overriding the hyperparameter state.
        upsampling_factor : int, optional
            Integer factor by which to refine the canvas relative to the scan sampling.
        override_rotation_angle : float, optional
            Detector rotation in degrees, overriding the hyperparameter state.
        max_batch_size : int, optional
            Number of bright-field pixels splatted at once. Defaults to a memory-bounded
            chunk of roughly four million ``(BF pixel, scan position)`` points.
        deconvolution_kernel : str
            Only the parallax aliases (``"prlx"``, ``"parallax"``, ``"tcbf"``, ...) are
            supported; other kernels are not real-space translations.
        q_highpass, q_lowpass : float, optional
            Butterworth filter cutoffs, applied once to the finished image.
        parallax_flip_phase : bool
            Apply the ``sign(sin(chi(q)))`` phase-flip filter.
        boundary : {"wrap", "pad"}, optional
            ``"wrap"`` wraps the montage periodically over the scan grid and reproduces
            ``DirectPtychography``; ``"pad"`` grows the canvas to cover the shifted positions
            and drops nothing. Defaults to the value chosen at construction.
        interpolation : {"bilinear", "nearest"}
            Sub-pixel deposition scheme. ``"nearest"`` reproduces the integer-rounded shifts
            of the streaming prototype.
        weight_normalize : bool, optional
            Divide by the accumulated weight rather than by the total bright-field weight.
            Defaults to ``True`` for ``"pad"`` and ``False`` for ``"wrap"``.

            For ``"wrap"`` with ``upsampling_factor > 1`` the accumulated weight is a comb of
            ones and zeros, so normalizing by it is meaningless -- leave it ``False``.
        weight_threshold : float
            Fraction of the peak weight below which the normalized image is tapered to zero,
            following ``bilinear_kde``. Only used when ``weight_normalize`` is true.
        pad_px : int, optional
            Freeze the ``"pad"`` canvas to the position bounding box plus this many pixels,
            instead of sizing it from the (aberration-dependent) shifts. Use this to keep the
            canvas fixed across hyperparameter trials.
        compute_variance : bool
            Accumulate the sum of squares needed by :meth:`variance_loss`.
        suppress_nyquist : bool
            Zero the Nyquist row and column of the phase-flip filter. Off by default, to
            match ``DirectPtychography``; turn it on for odd-order aberrations, where
            ``sign(sin(chi))`` is not symmetric and leaves a checkerboard artifact.

        Returns
        -------
        self
        """
        state = self.hyperparameter_state

        if verbose is None:
            verbose = self.verbose

        if use_initial_state:
            if verbose:
                print("Reconstructing with:\n\n", state.summarize(which="initial"))
            aberration_coefs = state.initial_aberrations
            rotation_angle = state.initial_rotation_angle
        else:
            if verbose:
                print(
                    "Reconstructing with:\n\n",
                    state.summarize(
                        which="current",
                        override_aberration_coefs=override_aberration_coefs,
                        override_rotation_angle=override_rotation_angle,
                    ),
                )
            aberration_coefs = state.current_aberrations(override_aberration_coefs)
            rotation_angle = state.current_rotation_angle(override_rotation_angle)

        if self._normalize_kernel_name(deconvolution_kernel) != "prlx":
            raise ValueError(
                f"{type(self).__name__} only implements the parallax kernel, got "
                f"{deconvolution_kernel!r}. SSB, OBF and matched-filter deconvolutions have "
                "bright-field-dependent Fourier multipliers that are not real-space "
                "translations; use DirectPtychography for those."
            )

        if upsampling_factor is None:
            upsampling_factor = 1
        upsampling_factor = math.ceil(upsampling_factor)

        if bf_mask is None:
            bf_mask = self.bf_mask
        bf = self._return_bf_context(bf_mask)

        if boundary is None:
            boundary = self.boundary
        if weight_normalize is None:
            weight_normalize = boundary == "pad"

        shifts_px, bf_weights = self._return_shifts_px(
            rotation_angle, aberration_coefs, bf.bf_mask, upsampling_factor
        )
        canvas_shape, canvas_origin, canvas_fov = self._return_canvas(
            shifts_px, upsampling_factor, boundary, pad_px
        )

        if max_batch_size is None:
            max_batch_size = max(1, _DEFAULT_POINTS_PER_BATCH // max(self.num_positions, 1))

        buffers = allocate_splat_buffers(
            canvas_shape, self.device, accumulate_squares=compute_variance
        )
        coords_base = self._positions_px * upsampling_factor - canvas_origin

        pbar = tqdm(range(bf.num_bf), disable=not verbose)
        batcher = SimpleBatcher(bf.num_bf, batch_size=max_batch_size, shuffle=False, rng=self.rng)

        for batch_idx in batcher:
            mapped_idx = bf.vbf_index_mapping[batch_idx]
            values = self._vbf_stack[mapped_idx]  # (B, N_pos)
            coords = coords_base[None] + shifts_px[batch_idx][:, None]  # (B, N_pos, 2)

            scatter_add_splat(
                values,
                coords,
                canvas_shape,
                boundary=boundary,
                interpolation=interpolation,
                out=buffers,
            )
            pbar.update(len(batch_idx))
        pbar.close()

        self._sum_w, self._sum_wv, self._sum_wv2 = buffers
        self._canvas_shape = canvas_shape
        self._canvas_origin_px = canvas_origin
        self._canvas_fov = canvas_fov
        self._bf_weights = bf_weights

        # normalization must precede filtering: dividing by the (spatially varying) weight
        # map is not linear, so it does not commute with the Fourier filters below
        if weight_normalize:
            mean, _, support = self._weighted_moments(weight_threshold)
            obj = (mean * support).reshape(canvas_shape)
        else:
            obj = self._sum_wv.reshape(canvas_shape) / bf_weights

        obj = self._apply_fourier_filters(
            obj,
            aberration_coefs=aberration_coefs,
            upsampling_factor=upsampling_factor,
            q_lowpass=q_lowpass,
            q_highpass=q_highpass,
            butterworth_order=butterworth_order,
            parallax_flip_phase=parallax_flip_phase,
            suppress_nyquist=suppress_nyquist,
        )
        self._corrected_bf = obj.to(torch.float32)

        # memory management
        gc.collect()
        torch.cuda.empty_cache()
        if hasattr(torch, "mps") and torch.backends.mps.is_available():
            torch.mps.empty_cache()
        gc.collect()

        return self

    def _apply_fourier_filters(
        self,
        obj,
        *,
        aberration_coefs,
        upsampling_factor,
        q_lowpass,
        q_highpass,
        butterworth_order,
        parallax_flip_phase,
        suppress_nyquist,
    ):
        """
        Apply the bright-field-index-independent filters once, on the summed image.

        ``DirectPtychography`` multiplies these into every bright-field image before its
        inverse transform; because they do not depend on the bright-field index, doing it
        once on the sum is exactly equivalent.
        """
        if not (parallax_flip_phase or q_lowpass or q_highpass or suppress_nyquist):
            return obj

        upsampled_sampling = tuple(s / upsampling_factor for s in self.scan_sampling)
        qxa, qya = spatial_frequencies(obj.shape, upsampled_sampling, device=self.device)
        q, theta = polar_coordinates(qxa, qya)

        # the filter is deliberately built at the grid's native precision rather than the
        # accumulator's: chi(q) reaches tens of radians, so sign(sin(chi)) is ill-conditioned
        # at its zero crossings and evaluating it in float64 would flip a handful of pixels
        # relative to DirectPtychography, which is a large perturbation per flipped mode
        filt = torch.ones_like(q)
        if parallax_flip_phase:
            chi_q = aberration_surface(
                q * self.wavelength,
                theta,
                self.wavelength,
                aberration_coefs=aberration_coefs,
            )
            filt = filt * torch.sign(torch.sin(chi_q))
        if q_lowpass:
            filt = filt / (1 + (q / q_lowpass) ** (2 * butterworth_order))
        if q_highpass:
            filt = filt * (1 - 1 / (1 + (q / q_highpass) ** (2 * butterworth_order)))
        if suppress_nyquist:
            n_rows, n_cols = obj.shape
            if n_rows % 2 == 0:
                filt[n_rows // 2, :] = 0.0
            if n_cols % 2 == 0:
                filt[:, n_cols // 2] = 0.0

        return torch.fft.ifft2(torch.fft.fft2(obj) * filt.to(obj.dtype)).real

    def _weighted_moments(self, weight_threshold: float = 1e-2):
        """``(mean, variance, support)`` per canvas pixel, as flat tensors."""
        w = self._sum_w
        if w is None or self._sum_wv is None:
            raise RuntimeError("Run reconstruct() before asking for the accumulated moments.")

        tiny = torch.finfo(w.dtype).tiny
        inv_w = 1 / w.clamp_min(tiny)

        mean = self._sum_wv * inv_w
        if self._sum_wv2 is None:
            var = torch.zeros_like(mean)
        else:
            var = (self._sum_wv2 * inv_w - mean.square()).clamp_min(0)

        w_max = w.max()
        if w_max <= 0:
            support = torch.zeros_like(w)
        else:
            support = (w / (weight_threshold * w_max)).clamp(max=1.0)

        return mean * (w > 0), var * (w > 0), support

    def variance_loss(self):
        """
        Weight-averaged variance across bright-field images, without storing the stack.

        Accumulating ``sum(w)``, ``sum(w*v)`` and ``sum(w*v**2)`` during the splat gives the
        per-pixel population variance across bright-field images directly, so no
        ``(N_bf, Ny, Nx)`` stack is needed.

        It differs from ``DirectPtychography.variance_loss`` in four documented ways:

        1. It is a weight-averaged mean over pixels rather than an unweighted one. The two
           coincide for ``boundary="wrap"`` on a complete grid with ``upsampling_factor=1``,
           where the accumulated weight is exactly ``num_bf`` everywhere.
        2. It lacks the ``1 / bf_weights**2`` scale, being computed on the raw values.
        3. It is computed *before* the phase-flip and Butterworth filters, which are applied
           post-hoc here but per bright-field image there. With ``parallax_flip_phase=True``
           this is a genuinely different objective.
        4. For ``upsampling_factor > 1`` it ignores unvisited canvas pixels instead of
           counting them as zeros.

        With ``interpolation="bilinear"`` and non-integer shifts it also folds in the
        within-interpolation spread, since ``sum(w*v**2)`` averages ``v**2`` over the
        neighbouring source pixels. Prefer ``upsampling_factor=1`` and
        ``parallax_flip_phase=False`` when driving a hyperparameter search.
        """
        if self._sum_w is None or self._sum_wv2 is None:
            return None

        _, var, _ = self._weighted_moments()
        w = self._sum_w
        denom = w.sum()
        if denom <= 0:
            return torch.tensor(torch.inf, dtype=w.dtype, device=self.device)
        return (var * w).sum() / denom
