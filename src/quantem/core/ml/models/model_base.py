from abc import ABC, abstractmethod
from typing import Dict

import torch


class PPLR(ABC):
    """
    Abstract base class for models that require multi-scale parameter optimization.
    """
    @abstractmethod
    def get_params(self) -> Dict[str, list[torch.nn.Parameter]]:
        """
        This abstract method should return a dictionary of parameters based on a key.

        For example if your nn.Module has multiple optimizable parameter groups, 
        you can return a dictionary with the keys "grids" and "sigma_net" (KPlanes example).
        """
        pass

    @property
    @abstractmethod
    def param_keys(self) -> list[str]:
        """
        This abstract property should return a list of available parameter keys.
        """
        pass