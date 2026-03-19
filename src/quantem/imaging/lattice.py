import numpy as np
from numpy.typing import NDArray

from quantem.core.datastructures.dataset2d import Dataset2d
from quantem.core.io.serialize import AutoSerialize
from quantem.core.visualization import show_2d


class Lattice(AutoSerialize):
    """
    Atomic lattice fitting in 2D.
    """

    _token = object()

    def __init__(
        self,
        image: Dataset2d,
        _token: object | None = None,
    ):
        if _token is not self._token:
            raise RuntimeError("Use Lattice.from_data() to instantiate this class.")
        self._image: Dataset2d = image

    ### --- Constructors ---
    @classmethod
    def from_data(
        cls,
        image: Dataset2d | NDArray,
        normalize_min: bool = True,
        normalize_max: bool = True,
    ) -> "Lattice":
        """
        Create a Lattice instance from a 2D image-like input.

        Parameters:
        - image: A 2D numpy array or a Dataset2d instance representing the image.
        - normalize_min: If True, shift the image so its minimum becomes 0.
        - normalize_max: If True, scale the image by its maximum after min-shift
          so values are in [0, 1]. If the maximum is 0 or non-finite (NaN/Inf),
          scaling is skipped to avoid invalid operations.

        Notes:
        - Non-2D inputs and empty arrays raise a ValueError.
        - Inputs with boolean dtype are safely converted to float before normalization.
        - NaN values are ignored when computing min/max (using nanmin/nanmax). If the
          data is all-NaN, normalization is skipped.
        """
        if isinstance(image, Dataset2d):
            ds2d = image
            # Ensure numeric operations are valid (e.g., for bool dtype)
            ds2d.array = np.asarray(ds2d.array, dtype=float)
            # Validate shape
            if ds2d.array.ndim != 2:
                raise ValueError("Input image must be a 2D array.")
            if ds2d.array.size == 0:
                raise ValueError("Input image array must not be empty.")
        else:
            # Validate dimensionality and emptiness before any processing
            arr = np.asarray(image)
            if arr.ndim != 2:
                raise ValueError("Input image must be a 2D array.")
            if arr.size == 0:
                raise ValueError("Input image array must not be empty.")
            # Convert to float for safe arithmetic (handles bool arrays)
            arr = arr.astype(float, copy=False)
            if hasattr(Dataset2d, "from_array") and callable(getattr(Dataset2d, "from_array")):
                ds2d = Dataset2d.from_array(arr)  # type: ignore[attr-defined]
            else:
                ds2d = Dataset2d(arr)  # type: ignore[call-arg]

        # Normalization (robust to constant, NaN, and bool inputs)
        if normalize_min:
            # Use nanmin to ignore NaNs; if all-NaN, skip
            try:
                min_val = np.nanmin(ds2d.array)
                if np.isfinite(min_val):
                    ds2d.array = ds2d.array - min_val
            except ValueError:
                # Raised when all values are NaN; skip
                pass

        if normalize_max:
            # Use nanmax to ignore NaNs; skip division if max <= 0 or not finite
            try:
                max_val = np.nanmax(ds2d.array)
                if np.isfinite(max_val) and max_val > 0.0:
                    ds2d.array = ds2d.array / max_val
            except ValueError:
                # Raised when all values are NaN; skip
                pass

        return cls(image=ds2d, _token=cls._token)

    ### --- Properties ---
    @property
    def image(self) -> Dataset2d:
        return self._image

    @image.setter
    def image(self, value: Dataset2d | NDArray):
        if isinstance(value, Dataset2d):
            # Ensure numeric dtype to avoid boolean arithmetic issues downstream
            value.array = np.asarray(value.array, dtype=float)
            # Validate shape
            if value.array.ndim != 2:
                raise ValueError("Input image must be a 2D array.")
            if value.array.size == 0:
                raise ValueError("Input image array must not be empty.")
            self._image = value
        else:
            arr = np.asarray(value)
            if arr.ndim != 2:
                raise ValueError("Input image must be a 2D array.")
            if arr.size == 0:
                raise ValueError("Input image array must not be empty.")
            arr = arr.astype(float, copy=False)
            if hasattr(Dataset2d, "from_array") and callable(getattr(Dataset2d, "from_array")):
                self._image = Dataset2d.from_array(arr)  # type: ignore[attr-defined]
            else:
                self._image = Dataset2d(arr)  # type: ignore[call-arg]

    ### --- Functions ---
    def define_lattice(
        self,
        origin,
        u,
        v,
        refine_lattice: bool = True,
        block_size: int | None = None,
        plot_lattice: bool = True,
        bound_num_vectors: int | None = None,
        refine_maxiter: int = 200,
        **kwargs,
    ) -> "Lattice":
        """
        Define the lattice for the image using the origin and the u and v vectors starting from the origin.
        The lattice is defined as r = r0 + nu + mv.

        Parameters
        ----------
        origin : NDArray[2] | Sequence[float]
            Start point (r0) to define the lattice.
            Enter as (row, col) as a numpy array, list, or tuple.
            Ideally a lattice point.
        u : NDArray[2] | Sequence[float]
            Basis vector u to define the lattice.
            Enter as (row, col) as a numpy array, list, or tuple.
        v : NDArray[2] | Sequence[float]
            Basis vector v to define the lattice.
            Enter as (row, col) as a numpy array, list, or tuple.
        refine_lattice : bool, default=True
            If True, refines the values of r0, u, and v by maximizing the bilinear intensity sum.
        block_size : int | None , default=None
            Fit the lattice points in steps of block_size * lattice_vectors(u, v).
            For example, if block_size = 5, then the lattice points will be fit in steps of
            (-5, 5)u * (-5, 5)v -> (-10, 10)u * (-10, 10)v -> ...
            block_size = None means the entire image will be fit at once.
        plot_lattice : bool, default=True
            If True, the lattice vectors and lines will be plotted overlaid on the image.
        bound_num_vectors : int | None, default=None
            The maximum number of lattice vectors to plot in each direction.
            For example, if bound_num_vectors = 5, lattice lines between (-5, 5)u * (-5, 5)v will be plotted.
            If None, the plotting bounds are set to the image edges.
        refine_maxiter : int, default=200
            Maximum number of iterations for the lattice refinement optimizer (Powell method).
        **kwargs
            Additional keyword arguments forwarded to the plotting function (show_2d), e.g., cmap, title, etc.

        Returns
        -------
        self : Lattice
            Returns the same object, modified in-place.
            The final values of r0, u, v are stored in self._lat.
        """
        # Lattice
        self._lat = np.vstack(
            (
                np.array(origin),
                np.array(u),
                np.array(v),
            )
        )
        if not self._lat.shape == (3, 2):
            raise ValueError("origin, u, v must be in (row, col) format only.")

        # Refine lattice coordinates
        # Note that we currently assume corners are local maxima
        if refine_lattice:
            from scipy.optimize import minimize

            if block_size is not None and block_size < 0:
                raise ValueError("block_size must be positive or None.")

            H, W = self._image.shape
            im = np.asarray(self._image.array, dtype=float)
            r0, u, v = (np.asarray(x, dtype=float) for x in self._lat)

            corners = np.array(
                [
                    [0.0, 0.0],
                    [float(H), 0.0],
                    [0.0, float(W)],
                    [float(H), float(W)],
                ],
                dtype=float,
            )

            # a,b from corners; A = [u v] in columns (2x2), rhs = (corner - r0)
            A = np.column_stack((u, v))  # (2,2)
            ab = np.linalg.lstsq(A, (corners - r0[None, :]).T, rcond=None)[0]  # (2,4)

            # Getting the min and max values for the indices a, b from the corners
            a_min, a_max = int(np.floor(ab[0].min())), int(np.ceil(ab[0].max()))
            b_min, b_max = int(np.floor(ab[1].min())), int(np.ceil(ab[1].max()))

            max_ind = max(abs(a_min), a_max, abs(b_min), b_max)
            if not block_size:
                steps = [max_ind]
            else:
                steps = (
                    [*np.arange(0, max_ind + 1, block_size)[1:], max_ind]
                    if max_ind > 0
                    else [max_ind]
                )

            PENALTY = 1e10
            H_CLIP = H - 2
            W_CLIP = W - 2
            a_range = np.arange(max(a_min, -max_ind), min(a_max, max_ind) + 1, dtype=np.int32)
            b_range = np.arange(max(b_min, -max_ind), min(b_max, max_ind) + 1, dtype=np.int32)
            aa, bb = np.meshgrid(a_range, b_range, indexing="ij")

            # Pre-compute all masks and bases
            all_masks = {}
            all_bases = {}
            for curr_block_size in steps:
                a_min_blk = max(a_min, -curr_block_size)
                a_max_blk = min(a_max, curr_block_size)
                b_min_blk = max(b_min, -curr_block_size)
                b_max_blk = min(b_max, curr_block_size)
                mask = (
                    (aa >= a_min_blk) & (aa <= a_max_blk) & (bb >= b_min_blk) & (bb <= b_max_blk)
                )

                aa_masked = aa[mask]
                bb_masked = bb[mask]

                all_masks[curr_block_size] = mask
                all_bases[curr_block_size] = np.column_stack(
                    [np.ones(aa_masked.size), aa_masked.ravel(), bb_masked.ravel()]
                )

            # Pre-allocate cache
            max_points = max(basis.shape[0] for basis in all_bases.values())
            x0_cache = np.empty(max_points, dtype=np.int32)
            y0_cache = np.empty(max_points, dtype=np.int32)
            dx_cache = np.empty(max_points, dtype=np.float64)
            dy_cache = np.empty(max_points, dtype=np.float64)

            def bilinear_sum(im_: np.ndarray, xy: np.ndarray) -> float:
                """Sum of bilinearly interpolated intensities at (x,y) points."""

                n_points = xy.shape[0]
                if n_points == 0:
                    return 0.0

                x, y = xy[:, 0], xy[:, 1]

                # Filter points that are within valid bounds for bilinear interpolation
                # Need x in [0, H-2] and y in [0, W-2] so that x+1 and y+1 are valid
                valid_mask = (
                    (x >= 0)
                    & (x <= H_CLIP)
                    & (y >= 0)
                    & (y <= W_CLIP)
                    & np.isfinite(x)
                    & np.isfinite(y)
                )

                n_valid = np.sum(valid_mask)
                if n_valid == 0:
                    return -PENALTY

                x_valid = x[valid_mask]
                y_valid = y[valid_mask]

                # Use pre-allocated arrays
                x0, y0 = x0_cache[:n_valid], y0_cache[:n_valid]
                dx, dy = dx_cache[:n_valid], dy_cache[:n_valid]

                np.floor(x_valid, out=dx)
                x0[:] = dx.astype(np.int32)
                np.floor(y_valid, out=dy)
                y0[:] = dy.astype(np.int32)

                np.subtract(x_valid, x0, out=dx)
                np.subtract(y_valid, y0, out=dy)

                Ia = im_[x0, y0]
                Ib = im_[x0 + 1, y0]
                Ic = im_[x0, y0 + 1]
                Id = im_[x0 + 1, y0 + 1]

                return np.sum(
                    Ia * (1 - dx) * (1 - dy)
                    + Ib * dx * (1 - dy)
                    + Ic * (1 - dx) * dy
                    + Id * dx * dy
                )

            current_basis = None

            def objective(theta: np.ndarray) -> float:
                """Function to be minimized"""
                # theta is 6-vector -> (3,2) matrix [[r0],[u],[v]]
                lat = theta.reshape(3, 2)
                xy = current_basis @ lat  # (N,2) with columns (x,y)
                # Negative: maximize intensity sum by minimizing its negative
                return -bilinear_sum(im, xy)

            minimize_options = {
                "maxiter": int(refine_maxiter),
                "xtol": 1e-3,
                "ftol": 1e-3,
                "disp": False,
            }

            lat_flat = self._lat.astype(np.float32).reshape(-1)

            for curr_block_size in steps:
                current_basis = all_bases[curr_block_size]

                res = minimize(
                    objective,
                    lat_flat,
                    method="Powell",
                    options=minimize_options,
                )

                # Update for next iteration
                lat_flat = res.x
                self._lat = res.x.reshape(3, 2)

        # plotting
        if plot_lattice:
            fig, ax = show_2d(
                self._image.array,
                returnfig=True,
                **kwargs,
            )

            # Put the image at lowest zorder so overlays sit on top
            if ax.images:
                ax.images[-1].set_zorder(0)

            H, W = self._image.shape
            r0, u, v = (np.asarray(x, dtype=float) for x in self._lat)

            # Origin marker (TOP of stack)
            ax.scatter(
                r0[1],
                r0[0],  # (y, x)
                s=60,
                edgecolor=(0, 0, 0),
                facecolor=(0, 0.5, 0),
                marker="s",
                zorder=30,
            )

            # Lattice vectors as arrows
            n_vec = int(bound_num_vectors) if bound_num_vectors is not None else 1

            # draw n_vec arrows for u (red)
            for k in range(1, n_vec + 1):
                tip = r0 + k * u
                ax.arrow(
                    r0[1],
                    r0[0],  # base (y, x)
                    (tip - r0)[1],
                    (tip - r0)[0],  # delta (y, x)
                    length_includes_head=True,
                    head_width=4.0,
                    head_length=6.0,
                    linewidth=2.0,
                    color="red",
                    zorder=20,
                )

            # draw n_vec arrows for v (cyan)
            for k in range(1, n_vec + 1):
                tip = r0 + k * v
                ax.arrow(
                    r0[1],
                    r0[0],
                    (tip - r0)[1],
                    (tip - r0)[0],
                    length_includes_head=True,
                    head_width=4.0,
                    head_length=6.0,
                    linewidth=2.0,
                    color=(0.0, 0.7, 1.0),
                    zorder=20,
                )

            # Solve for a,b at plot corners (bounds)
            if bound_num_vectors is None:
                corners = np.array(
                    [
                        [0.0, 0.0],
                        [float(H), 0.0],
                        [0.0, float(W)],
                        [float(H), float(W)],
                    ]
                )
            else:
                n = float(bound_num_vectors)
                corners = np.array(
                    [
                        r0 - n * u,
                        r0 - n * v,
                        r0 + n * u,
                        r0 + n * v,
                    ],
                    dtype=float,
                )

            # a,b from corners; A = [u v] in columns (2x2), rhs = (corner - r0)
            A = np.column_stack((u, v))
            ab = np.linalg.lstsq(A, (corners - r0[None, :]).T, rcond=None)[0]

            a_min, a_max = int(np.floor(np.min(ab[0]))), int(np.ceil(np.max(ab[0])))
            b_min, b_max = int(np.floor(np.min(ab[1]))), int(np.ceil(np.max(ab[1])))

            # Clipping rectangle (image or custom)
            if bound_num_vectors is None:
                x_lo, x_hi = 0.0, float(H)
                y_lo, y_hi = 0.0, float(W)
            else:
                # Bounds are the min/max over the provided corners
                x_lo, x_hi = float(np.min(corners[:, 0])), float(np.max(corners[:, 0]))
                y_lo, y_hi = float(np.min(corners[:, 1])), float(np.max(corners[:, 1]))

            def clipped_segment(base: np.ndarray, direction: np.ndarray):
                """Clip base + t*direction to rectangle [x_lo,x_hi] x [y_lo,y_hi]."""
                x0, y0 = base
                dx, dy = direction
                t0, t1 = -np.inf, np.inf
                eps = 1e-12

                # x in [x_lo, x_hi]
                if abs(dx) < eps:
                    if not (x_lo <= x0 <= x_hi):
                        return None
                else:
                    tx0 = (x_lo - x0) / dx
                    tx1 = (x_hi - x0) / dx
                    t_enter, t_exit = (tx0, tx1) if tx0 <= tx1 else (tx1, tx0)
                    t0, t1 = max(t0, t_enter), min(t1, t_exit)

                # y in [y_lo, y_hi]
                if abs(dy) < eps:
                    if not (y_lo <= y0 <= y_hi):
                        return None
                else:
                    ty0 = (y_lo - y0) / dy
                    ty1 = (y_hi - y0) / dy
                    t_enter, t_exit = (ty0, ty1) if ty0 <= ty1 else (ty1, ty0)
                    t0, t1 = max(t0, t_enter), min(t1, t_exit)

                if t0 > t1:
                    return None

                p1 = base + t0 * direction  # (x, y)
                p2 = base + t1 * direction
                return p1, p2

            # Lattice lines (zorder above image)
            # Using x=rows, y=cols: plot(y, x)

            # Lines parallel to v (vary a)
            for a in range(a_min, a_max + 1):
                base = r0 + a * u
                seg = clipped_segment(base, v)
                if seg is None:
                    continue
                (x1, y1), (x2, y2) = seg
                ax.plot([y1, y2], [x1, x2], color=(0.0, 0.7, 1.0), lw=1, clip_on=True, zorder=10)

            # Lines parallel to u (vary b)
            for b in range(b_min, b_max + 1):
                base = r0 + b * v
                seg = clipped_segment(base, u)
                if seg is None:
                    continue
                (x1, y1), (x2, y2) = seg
                ax.plot([y1, y2], [x1, x2], color="red", lw=1, clip_on=True, zorder=10)

            # Axes limits (x=rows vertical; y=cols horizontal)
            ax.set_xlim(y_lo, y_hi)
            ax.set_ylim(x_hi, x_lo)

        return self
