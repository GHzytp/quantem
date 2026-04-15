from abc import ABC, abstractmethod
from typing import Dict

import torch


class PPLR(ABC):
    """
    Abstract base class for models that require multi-scale parameter optimization.
    """
    @abstractmethod
    def get_optimization_parameters(self) -> Dict[str, list[torch.nn.Parameter]]:
        pass