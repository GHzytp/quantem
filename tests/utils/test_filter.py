"""Tests for `filter_hot_pixels`.

A microscopist running 4D-STEM sees most detector pixels read out at low
counts (~1-100), but a few are stuck near saturation (~60000) regardless of
incident intensity. They must be removed before virtual imaging or dp_max
analysis.
"""

import torch

from quantem.core.utils.filter import filter_hot_pixels


def test_filter_hot_pixels_replaces_stuck_detector_pixels_with_local_median():
    """Stuck pixels should drop from 60000 back into the local bulk regime (<1000)."""
    ds = torch.randint(1, 101, size=(64, 64, 32, 32), dtype=torch.int32)
    # Assume these 3 places have hot pixels that we later want to remove
    hot_coords = [(5, 7), (18, 24), (29, 3)]
    for r, c in hot_coords:
        ds[:, :, r, c] = 60000
    filtered = filter_hot_pixels(ds.numpy())
    dp_max = filtered.max(axis=(0, 1))
    # no pixels should have value 101
    assert dp_max.max() < 101, f"hot pixels still present, dp_max max={dp_max.max()}"
