from dataclasses import dataclass, fields
from typing import Literal, Optional

from quantem.core.visualization.custom_normalizations import NormalizationConfig
from quantem.core.visualization.visualization_utils import ScalebarConfig


class ShowParams:
    """
    Container for ``show_2d`` parameter dataclasses.

    Nested classes
    --------------
    Norm
        Normalization configuration (interval + stretch).
    Scalebar
        Scale bar overlay configuration.

    Examples
    --------
    >>> show_2d(img, norm=ShowParams.Norm(power=0.5))
    >>> show_2d(img, scalebar=ShowParams.Scalebar(sampling=0.5, units="Å"))
    >>> show_2d(dp, norm=ShowParams.Norm.log_auto(), cbar=True, cmap="turbo")
    """

    @dataclass
    class Norm:
        """
        Normalization parameters for ``show_2d``.

        Controls how pixel values are mapped to the [0, 1] display range via
        an *interval* (which values to keep) and a *stretch* (non-linear
        transfer function).

        If ``vmin`` or ``vmax`` is set and ``interval_type`` is left as the
        default ``"quantile"``, it is automatically changed to ``"manual"``.
        Likewise, setting ``vcenter`` to a non-zero value or providing
        ``half_range`` auto-selects ``"centered"``.

        Parameters
        ----------
        interval_type : ``"quantile"`` | ``"manual"`` | ``"centered"``
            How to determine the data range.
        stretch_type : ``"linear"`` | ``"power"`` | ``"logarithmic"`` | ``"asinh"``
            Transfer function applied after interval mapping.
        lower_quantile : float
            Lower quantile for ``"quantile"`` interval. Default 0.02.
        upper_quantile : float
            Upper quantile for ``"quantile"`` interval. Default 0.98.
        vmin : float or None
            Explicit minimum for ``"manual"`` interval.
        vmax : float or None
            Explicit maximum for ``"manual"`` interval.
        vcenter : float
            Centre value for ``"centered"`` interval. Default 0.0.
        half_range : float or None
            Symmetric half-range for ``"centered"`` interval.
        power : float
            Exponent for ``"power"`` stretch (e.g. 0.5 = sqrt). Default 1.0.
        logarithmic_index : float
            Index *a* for ``"logarithmic"`` stretch: ``log(a*x+1)/log(a+1)``.
            Default 1000.
        asinh_linear_range : float
            Transition parameter *a* for ``"asinh"`` stretch. Default 0.1.

        Examples
        --------
        >>> ShowParams.Norm()                        # quantile + linear (default)
        >>> ShowParams.Norm(power=0.5)               # quantile + sqrt stretch
        >>> ShowParams.Norm(vmin=0, vmax=1000)       # auto → manual range
        >>> ShowParams.Norm.log_auto()               # quantile + log stretch
        >>> ShowParams.Norm.centered(half_range=5)   # centered ± 5, linear
        """

        interval_type: Literal["quantile", "manual", "centered"] = "quantile"
        stretch_type: Literal["linear", "power", "logarithmic", "asinh"] = "linear"
        lower_quantile: float = 0.02
        upper_quantile: float = 0.98
        vmin: Optional[float] = None
        vmax: Optional[float] = None
        vcenter: float = 0.0
        half_range: Optional[float] = None
        power: float = 1.0
        logarithmic_index: float = 1000.0
        asinh_linear_range: float = 0.1

        def __post_init__(self) -> None:
            if self.interval_type != "quantile":
                return
            if self.vmin is not None or self.vmax is not None:
                self.interval_type = "manual"
            elif self.vcenter != 0.0 or self.half_range is not None:
                self.interval_type = "centered"

        def to_config(self) -> NormalizationConfig:
            """Convert to a ``NormalizationConfig``."""
            return NormalizationConfig(**{f.name: getattr(self, f.name) for f in fields(self)})

        # ---- presets (mirror NORMALIZATION_PRESETS) ----

        @classmethod
        def linear_auto(cls, **kw) -> "ShowParams.Norm":
            """Quantile interval + linear stretch (the default)."""
            return cls(**kw)

        @classmethod
        def minmax(cls, **kw) -> "ShowParams.Norm":
            """Full min/max interval + linear stretch."""
            return cls(interval_type="manual", **kw)

        @classmethod
        def centered(
            cls, vcenter: float = 0.0, half_range: Optional[float] = None, **kw
        ) -> "ShowParams.Norm":
            """Centered interval + linear stretch."""
            return cls(interval_type="centered", vcenter=vcenter, half_range=half_range, **kw)

        @classmethod
        def log_auto(cls, **kw) -> "ShowParams.Norm":
            """Quantile interval + logarithmic stretch."""
            return cls(stretch_type="logarithmic", **kw)

        @classmethod
        def log_minmax(cls, **kw) -> "ShowParams.Norm":
            """Full min/max interval + logarithmic stretch."""
            return cls(interval_type="manual", stretch_type="logarithmic", **kw)

        @classmethod
        def power_sqrt(cls, **kw) -> "ShowParams.Norm":
            """Quantile interval + square-root (power=0.5) stretch."""
            return cls(stretch_type="power", power=0.5, **kw)

        @classmethod
        def power_squared(cls, **kw) -> "ShowParams.Norm":
            """Quantile interval + squared (power=2.0) stretch."""
            return cls(stretch_type="power", power=2.0, **kw)

        @classmethod
        def asinh_centered(cls, vcenter: float = 0.0, **kw) -> "ShowParams.Norm":
            """Centered interval + asinh stretch."""
            return cls(interval_type="centered", stretch_type="asinh", vcenter=vcenter, **kw)

    @dataclass
    class Scalebar:
        """
        Scale bar parameters for ``show_2d``.

        Parameters
        ----------
        sampling : float
            Physical units per pixel. Default 1.0.
        units : str
            Unit label displayed on the scale bar (e.g. ``"Å"``, ``"nm"``,
            ``"1/Å"``). Default ``"pixels"``.
        length : float or None
            Fixed scale bar length in physical units. ``None`` auto-estimates
            a "nice" length.
        width_px : float
            Thickness of the bar in image pixels. Default 1.
        pad_px : float
            Padding between bar and plot edge in image pixels. Default 0.5.
        color : str
            Bar and label colour. Default ``"white"``.
        loc : ``"lower right"`` | ``"lower left"`` | ``"upper right"`` | ``"upper left"``
            Anchor location. Default ``"lower right"``.
        fontsize : int
            Font size of the scale bar label in points. Default 12.
        bold : bool
            Whether to render the label in bold. Default True.

        Examples
        --------
        >>> ShowParams.Scalebar(sampling=0.5, units="Å")
        >>> ShowParams.Scalebar(sampling=0.02, units="1/Å", color="black", fontsize=16)
        """

        sampling: float = 1.0
        units: str = "pixels"
        length: Optional[float] = None
        width_px: float = 1
        pad_px: float = 0.5
        color: str = "white"
        loc: Literal["lower right", "lower left", "upper right", "upper left"] = "lower right"
        fontsize: int = 12
        bold: bool = False

        def to_config(self) -> ScalebarConfig:
            """Convert to a ``ScalebarConfig``."""
            return ScalebarConfig(**{f.name: getattr(self, f.name) for f in fields(self)})
