from abc import abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Generator, Iterator, Literal, Sequence

from quantem.core import config

if TYPE_CHECKING:
    import torch
else:
    if config.get("has_torch"):
        import torch


class OptimizerParams:
    """
    Nested class for optimizer parameters.
    """

    @dataclass
    class Adam:
        lr: float = 1e-3
        betas: tuple[float, float] = (0.9, 0.999)
        eps: float = 1e-8
        weight_decay: float = 0
        _name: str = "adam"

        def params(self) -> dict:
            return {
                "lr": self.lr,
                "betas": self.betas,
                "eps": self.eps,
                "weight_decay": self.weight_decay,
            }

    @dataclass
    class AdamW:
        lr: float = 1e-3
        betas: tuple[float, float] = (0.9, 0.999)
        eps: float = 1e-8
        weight_decay: float = 0
        _name: str = "adamw"

        def params(self) -> dict:
            return {
                "lr": self.lr,
                "betas": self.betas,
                "eps": self.eps,
                "weight_decay": self.weight_decay,
            }

    @dataclass
    class SGD:
        lr: float = 1e-3
        momentum: float = 0
        dampening: float = 0
        weight_decay: float = 0
        nesterov: bool = False
        _name: str = "sgd"

        def params(self) -> dict:
            return {
                "lr": self.lr,
                "momentum": self.momentum,
                "dampening": self.dampening,
                "weight_decay": self.weight_decay,
                "nesterov": self.nesterov,
            }

    @dataclass
    class NoneOptimizer:
        _name: str = "none"

        def params(self) -> dict:
            return {}

    @classmethod
    def parse_dict(cls, d: dict):
        """
        Parse dictionary to a optimizer params object.
        """
        if d["name"] == "adam":
            d.pop("name")
            return OptimizerParams.Adam(**d)
        elif d["name"] == "adamw":
            d.pop("name")
            return OptimizerParams.AdamW(**d)
        elif d["name"] == "sgd":
            d.pop("name")
            return OptimizerParams.SGD(**d)
        else:
            raise ValueError(f"Unknown optimizer type: {d['name']}")


OptimizerType = (
    OptimizerParams.Adam
    | OptimizerParams.AdamW
    | OptimizerParams.SGD
    | OptimizerParams.NoneOptimizer
)


class SchedulerParams:
    """
    Nested class for scheduler parameters.
    """

    @dataclass
    class Plateau:
        mode: Literal["min", "max"] = "min"
        min_lr_factor: float = 1 / 20
        min_lr: float | None = None
        factor: float = 0.5
        patience: int = 10
        threshold: float = 1e-5
        cooldown: int = 50
        _name: str = "plateau"

        def params(self, base_LR: float) -> dict:
            if self.min_lr is None:
                self.min_lr = self.min_lr_factor * base_LR
            return {
                "mode": self.mode,
                "factor": self.factor,
                "patience": self.patience,
                "threshold": self.threshold,
                "min_lr": self.min_lr,
                "cooldown": self.cooldown,
            }

    @dataclass
    class Exponential:
        gamma: float = 0.9
        factor: float | None = 0.5
        num_iter: int | None = None
        _name: str = "exponential"

        def params(self, base_LR: float) -> dict:
            return {
                "gamma": self.gamma,
                "factor": self.factor,
            }

    @dataclass
    class Cyclic:
        base_lr_factor: float = 1 / 4
        max_lr_factor: float = 4
        base_lr: float | None = None
        max_lr: float | None = None
        step_size_up: int = 100
        step_size_down: int = 100
        mode: Literal["triangular2", "triangular", "exp_range"] = "triangular2"
        cycle_momentum: bool = False
        _name: str = "cyclic"

        def params(self, base_LR: float) -> dict:
            if self.base_lr is None:
                self.base_lr = self.base_lr_factor * base_LR
            if self.max_lr is None:
                self.max_lr = self.max_lr_factor * base_LR
            return {
                "base_lr": self.base_lr,
                "max_lr": self.max_lr,
                "step_size_up": self.step_size_up,
                "step_size_down": self.step_size_down,
                "mode": self.mode,
                "cycle_momentum": self.cycle_momentum,
            }

    @dataclass
    class Linear:
        total_iters: int
        start_factor: float = 0.1
        end_factor: float = 1.0
        _name: str = "linear"

        def params(self, base_LR: float) -> dict:
            return {
                "start_factor": self.start_factor,
                "end_factor": self.end_factor,
                "total_iters": self.total_iters,
            }

    @dataclass
    class CosineAnnealing:
        T_max: int
        eta_min: float = 1e-7
        _name: str = "cosine_annealing"

        def params(self, base_LR: float) -> dict:
            return {
                "T_max": self.T_max,
                "eta_min": self.eta_min,
            }

    @dataclass
    class NoneScheduler:
        _name: str = "none"

        def params(self, base_LR: float) -> dict:
            return {}

    @classmethod
    def parse_dict(cls, d: dict):
        """
        Parse dictionary to a scheduler params object.
        """
        if d["name"] == "plateau":
            d.pop("name")
            return SchedulerParams.Plateau(**d)
        elif d["name"] == "exponential":
            d.pop("name")
            return SchedulerParams.Exponential(**d)
        elif d["name"] == "cyclic":
            d.pop("name")
            return SchedulerParams.Cyclic(**d)
        elif d["name"] == "linear":
            d.pop("name")
            return SchedulerParams.Linear(**d)
        elif d["name"] == "cosine_annealing":
            d.pop("name")
            return SchedulerParams.CosineAnnealing(**d)
        elif d["name"] == "none":
            d.pop("name")
            return SchedulerParams.NoneScheduler()
        else:
            raise ValueError(f"Unknown scheduler type: {d['name']}")


SchedulerType = (
    SchedulerParams.Plateau
    | SchedulerParams.Exponential
    | SchedulerParams.Cyclic
    | SchedulerParams.Linear
    | SchedulerParams.CosineAnnealing
    | SchedulerParams.NoneScheduler
)


class OptimizerMixin:
    """
    Mixin class for handling optimizer and scheduler management.
    Each model (object, probe, dataset) can inherit from this to manage its own optimizers.
    """

    DEFAULT_OPTIMIZER_TYPE = "adamw"

    def __init__(self):
        """Initialize the optimizer mixin."""
        self._optimizer = None
        self._scheduler = None
        self._optimizer_params: OptimizerType = OptimizerParams.NoneOptimizer()
        self._scheduler_params: SchedulerType = SchedulerParams.NoneScheduler()
        # Don't call super().__init__() in mixin classes to avoid MRO issues

    @property
    def optimizer(self) -> "torch.optim.Optimizer | None":
        """Get the optimizer for this model."""
        return self._optimizer

    @property
    def scheduler(self) -> "torch.optim.lr_scheduler.LRScheduler | None":
        """Get the scheduler for this model."""
        return self._scheduler

    @property
    def optimizer_params(self) -> OptimizerType:
        """Get the optimizer parameters."""
        return self._optimizer_params

    @optimizer_params.setter
    def optimizer_params(self, params: OptimizerType | dict):
        """Set the optimizer parameters."""
        if isinstance(params, dict):
            params = OptimizerParams.parse_dict(d=params)
        if not isinstance(params, OptimizerType):
            raise TypeError(f"optimizer parameters must be a OptimizerType, got {type(params)}")
        self._optimizer_params = params

    @property
    def scheduler_params(self) -> SchedulerType:
        """Get the scheduler parameters."""
        return self._scheduler_params

    @scheduler_params.setter
    def scheduler_params(self, params: SchedulerType | dict):
        """Set the scheduler parameters."""
        if isinstance(params, dict):
            params = SchedulerParams.parse_dict(d=params)
        if not isinstance(params, SchedulerType):
            raise TypeError(f"scheduler parameters must be a SchedulerType, got {type(params)}")
        self._scheduler_params = params

    @abstractmethod
    def get_optimization_parameters(
        self,
    ) -> "torch.Tensor | Sequence[torch.Tensor] | Iterator[torch.Tensor]":
        """
        Get the parameters that should be optimized for this model.
        This could be replaced with just module.parameters(), but this allows for flexibility
        in the future to allow for per parameter LRs.
        """
        raise NotImplementedError("Subclasses must implement get_optimization_parameters")

    def set_optimizer(self, opt_params: OptimizerType | dict | None = None) -> None:
        """
        Set the optimizer for this model.
        Currently supports single LR for all parameters, TODO allow for per parameter LRs by
        updating get_optimization_parameters to return a list of parameters and their LRs.
        """
        if opt_params is not None:
            self.optimizer_params = opt_params

        if not self._optimizer_params:
            self._optimizer = None
            return

        if isinstance(self._optimizer_params, OptimizerParams.NoneOptimizer):
            self.remove_optimizer()
            return

        params = self.get_optimization_parameters()
        if isinstance(params, torch.Tensor):
            params = [params]
        elif isinstance(params, Generator):
            params = list(params)

        # Ensure parameters require gradients
        for p in params:
            p.requires_grad_(True)

        match self._optimizer_params:
            case OptimizerParams.Adam():
                self._optimizer = torch.optim.Adam(params, **self._optimizer_params.params())
            case OptimizerParams.AdamW():
                self._optimizer = torch.optim.AdamW(params, **self._optimizer_params.params())
            case OptimizerParams.SGD():
                self._optimizer = torch.optim.SGD(params, **self._optimizer_params.params())
            case _:
                raise NotImplementedError(f"Unknown optimizer type: {self._optimizer_params}")

    def set_scheduler(
        self,
        scheduler_params: SchedulerType | dict | None = None,
    ) -> None:
        """Set the scheduler for this model."""
        if scheduler_params is not None:
            self.scheduler_params = scheduler_params

        if not self._scheduler_params or self._optimizer is None:
            self._scheduler = None
            return

        optimizer = self._optimizer
        base_LR = optimizer.param_groups[0]["lr"]

        params = self._scheduler_params.params(base_LR)
        match self.scheduler_params:
            case SchedulerParams.NoneScheduler():
                self._scheduler = None
            case SchedulerParams.Cyclic():
                self._scheduler = torch.optim.lr_scheduler.CyclicLR(
                    optimizer,
                    **params,
                )
            case SchedulerParams.Plateau():
                self._scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer,
                    **params,
                )
            case SchedulerParams.Exponential():
                self._scheduler = torch.optim.lr_scheduler.ExponentialLR(
                    optimizer,
                    **params,
                )
            case SchedulerParams.Linear():
                self._scheduler = torch.optim.lr_scheduler.LinearLR(
                    optimizer,
                    **params,
                )
            case SchedulerParams.CosineAnnealing():
                self._scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer,
                    **params,
                )
            case _:
                raise ValueError(f"Unknown scheduler type: {self.scheduler_params}")

    def step_optimizer(self) -> None:
        """Step the optimizer if it exists."""
        if self._optimizer is not None:
            self._optimizer.step()

    def zero_optimizer_grad(self) -> None:
        """Zero gradients if optimizer exists."""
        if self._optimizer is not None:
            self._optimizer.zero_grad()

    def step_scheduler(self, loss: float | None = None) -> None:
        """Step the scheduler if it exists."""
        if self._scheduler is not None:
            if isinstance(self._scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                if loss is not None:
                    self._scheduler.step(loss)
            else:
                self._scheduler.step()

    def has_optimizer(self) -> bool:
        """Check if this model has an active optimizer."""
        return self._optimizer is not None

    def get_current_lr(self) -> float:
        """Get the current learning rate."""
        if self._optimizer is not None:
            return self._optimizer.param_groups[0]["lr"]
        return 0.0

    def remove_optimizer(self) -> None:
        """Remove the optimizer and scheduler."""
        self._optimizer = None
        self._optimizer_params = OptimizerParams.NoneOptimizer()
        self._scheduler = None
        self._scheduler_params = SchedulerParams.NoneScheduler()

    def reset_optimizer(self) -> None:
        """Reset the optimizer and scheduler."""
        self.set_optimizer(self._optimizer_params)
        self.set_scheduler(self._scheduler_params)

    def reconnect_optimizer_to_parameters(self) -> None:
        """
        Reconnect optimizer to parameters after device changes.
        This is needed because AutoSerialize loads to CPU, but optimizers
        need to reference tensors on the current device.
        """
        if self._optimizer is None:
            return

        current_params = self.get_optimization_parameters()
        if isinstance(current_params, torch.Tensor):
            current_params = [current_params]
        elif isinstance(current_params, Generator):
            current_params = list(current_params)

        optimizable_params = [
            p for p in current_params if isinstance(p, torch.Tensor) and p.is_leaf
        ]

        if not optimizable_params:
            print(
                f"souldn't be getting here! No optimizable parameters found for {self.__class__.__name__}, removing optimizer"
            )
            self.remove_optimizer()
            return

        for p in optimizable_params:
            p.requires_grad_(True)

        # Preserve optimizer state and param_group settings
        old_state = self._optimizer.state.copy()
        current_param_group = self._optimizer.param_groups[0].copy()

        # Reconnect to new parameters
        self._optimizer.param_groups.clear()
        self._optimizer.add_param_group({"params": optimizable_params})

        # Update state mapping and move tensors to correct device
        new_state = {}
        device = optimizable_params[0].device
        for i, old_param in enumerate(old_state.keys()):
            if i < len(optimizable_params):
                new_param = optimizable_params[i]
                new_state[new_param] = {}
                for key, value in old_state[old_param].items():
                    if isinstance(value, torch.Tensor):
                        new_state[new_param][key] = value.to(device)
                    else:
                        new_state[new_param][key] = value

        self._optimizer.state.clear()
        self._optimizer.state.update(new_state)

        # Restore param_group settings (LR, betas, etc.) but keep new parameters
        self._optimizer.param_groups[0].update(
            {k: v for k, v in current_param_group.items() if k != "params"}
        )

        # Reconnect scheduler
        if self._scheduler is not None and self._optimizer is not None:
            self._scheduler.optimizer = self._optimizer
        return
