"""
Array utilities for widgets. Supports NumPy + PyTorch input.
"""

from typing import Literal
import numpy as np

try:
    import torch
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False


ArrayBackend = Literal["numpy", "torch", "unknown"]


def get_array_backend(data) -> ArrayBackend:
    """Detect array backend. Returns 'numpy', 'torch', or 'unknown'."""
    if _HAS_TORCH and isinstance(data, torch.Tensor):
        return "torch"
    if isinstance(data, np.ndarray):
        return "numpy"
    return "unknown"


def to_numpy(data, dtype: np.dtype | None = None) -> np.ndarray:
    """Convert NumPy or PyTorch array to NumPy.

    Parameters
    ----------
    data : np.ndarray or torch.Tensor
        Input array.
    dtype : np.dtype, optional
        Target dtype.

    Returns
    -------
    np.ndarray

    Examples
    --------
    >>> import numpy as np
    >>> to_numpy(np.zeros((4, 4)))
    >>> import torch
    >>> to_numpy(torch.zeros(4, 4))

    Raises
    ------
    TypeError
        If `data` is not a NumPy array or PyTorch tensor.
    """
    backend = get_array_backend(data)
    if backend == "torch":
        result = data.detach().cpu().numpy()
    elif backend == "numpy":
        result = data
    else:
        # Try np.asarray as last-resort fallback for things like Dataset arrays
        try:
            result = np.asarray(data)
        except Exception as e:
            raise TypeError(
                f"to_numpy expected a NumPy array or PyTorch tensor, got {type(data).__name__}. "
                f"Convert your input via np.asarray(...) or tensor.cpu().numpy() first."
            ) from e
    if dtype is not None:
        result = np.asarray(result, dtype=dtype)
    return result


def _resize_image(img: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    """Center-pad an image to (target_h, target_w) with zeros.

    Used to align gallery images of different shapes to a common canvas.
    """
    h, w = img.shape[-2:]
    if h == target_h and w == target_w:
        return img
    pad_top = (target_h - h) // 2
    pad_bot = target_h - h - pad_top
    pad_left = (target_w - w) // 2
    pad_right = target_w - w - pad_left
    return np.pad(img, ((pad_top, pad_bot), (pad_left, pad_right)), mode="constant", constant_values=0)


def apply_shift(img: np.ndarray, dy: float, dx: float) -> np.ndarray:
    """Sub-pixel image shift via bilinear interpolation. Used for diff alignment."""
    if not _HAS_TORCH:
        # Fallback: integer roll only
        return np.roll(img, (int(round(dy)), int(round(dx))), axis=(-2, -1))
    t = torch.from_numpy(img).float()
    if t.ndim == 2:
        t = t.unsqueeze(0).unsqueeze(0)
    h, w = t.shape[-2:]
    y = torch.arange(h, dtype=torch.float32) - dy
    x = torch.arange(w, dtype=torch.float32) - dx
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    grid = torch.stack(((xx / (w - 1)) * 2 - 1, (yy / (h - 1)) * 2 - 1), dim=-1).unsqueeze(0)
    out = torch.nn.functional.grid_sample(t, grid, mode="bilinear", padding_mode="border", align_corners=True)
    return out.squeeze().numpy()


def bin2d(img: np.ndarray, factor: int) -> np.ndarray:
    """Reduce 2D image by integer binning factor. Mean of f×f blocks."""
    if factor <= 1:
        return img
    h, w = img.shape[-2:]
    h2, w2 = h - h % factor, w - w % factor
    img = img[..., :h2, :w2]
    return img.reshape(*img.shape[:-2], h2 // factor, factor, w2 // factor, factor).mean(axis=(-3, -1))
