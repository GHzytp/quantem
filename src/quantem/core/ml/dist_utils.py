"""
Standalone distributed training utilities for ptychography.

These are kept separate from ddp.py (which imports tomography types) so they
can be used by diffractive_imaging without circular imports.
"""

from __future__ import annotations

import os

import torch
import torch.distributed as dist


def is_distributed_launch() -> bool:
    """True when launched via torchrun / torch.distributed.launch (RANK env var is set)."""
    return "RANK" in os.environ


def init_process_group(
    rank: int,
    world_size: int,
    backend: str = "nccl",
    master_addr: str = "127.0.0.1",
    master_port: str = "29500",
) -> None:
    """Initialize the distributed process group from within an mp.spawn worker."""
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = master_port
    dist.init_process_group(
        backend=backend,
        rank=rank,
        world_size=world_size,
    )
    if backend == "nccl":
        torch.cuda.set_device(rank)


def get_rank() -> int:
    """Return the current process rank (0 if not in a distributed context)."""
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    return 0


def get_world_size() -> int:
    """Return the world size (1 if not in a distributed context)."""
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size()
    return 1


def all_reduce_params(*params: torch.Tensor, op: dist.ReduceOp = dist.ReduceOp.AVG) -> None:
    """Average the .grad tensors of the given parameters across all ranks in-place."""
    for p in params:
        if p.grad is not None:
            _ = dist.all_reduce(p.grad, op=op)  # type: ignore[arg-type]


def broadcast_params(*params: torch.Tensor, src: int = 0) -> None:
    """Broadcast .data of each parameter from rank src to all other ranks."""
    for p in params:
        _ = dist.broadcast(p.data, src=src)