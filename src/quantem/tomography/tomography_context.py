from dataclasses import dataclass
from typing import Optional
from quantem.core.ml.constraints import BaseContext

import torch


@dataclass
class ReconstructionContext(BaseContext):
    """
    Handles all reconstruction parameters to be passed into object models.

    Subclasses will pick whatever parameter they need
        - Pixelated reads ".volume"
        - INR reads ".coords" and recomputes via the model.
        - TensorDecomp reads ".coords" and ".pred" (and ".all densities")
    """

    coords: Optional[torch.Tensor] = None
    pred: Optional[torch.Tensor] = None
    all_densities: Optional[torch.Tensor] = None
    obj: Optional[torch.Tensor] = None
