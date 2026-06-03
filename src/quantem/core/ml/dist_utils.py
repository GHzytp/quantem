"""
Standalone distributed training utilities for ptychography.

These are kept separate from ddp.py (which imports tomography types) so they
can be used by diffractive_imaging without circular imports.
"""

from __future__ import annotations

import os
from typing import Any

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
    local_device: int | None = None,
) -> None:
    """Initialize the distributed process group from within an mp.spawn worker.

    ``local_device`` is the physical CUDA device index this rank should bind to
    (e.g. with ``GPU_IDS=[2, 3]``, rank 0 should get ``local_device=2``).
    NCCL allocates communicator buffers on the *current* CUDA device at
    ``init_process_group`` time, so the device must be set *before* that call
    or the buffers will land on whichever device was current — typically
    ``cuda:0``. Falling back to ``rank`` matches PyTorch's
    ``LOCAL_RANK == device_index`` convention used by ``torchrun`` when each
    process maps to a contiguous device starting at 0.
    """
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = master_port
    if backend == "nccl":
        device_index = local_device if local_device is not None else rank
        torch.cuda.set_device(device_index)
    dist.init_process_group(
        backend=backend,
        rank=rank,
        world_size=world_size,
    )


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


def all_reduce_params(*params: torch.Tensor, op: Any = dist.ReduceOp.AVG) -> None:
    """Average the .grad tensors of the given parameters across all ranks in-place."""
    for p in params:
        if p.grad is not None:
            _ = dist.all_reduce(p.grad, op=op)


def broadcast_params(*params: torch.Tensor, src: int = 0) -> None:
    """Broadcast .data of each parameter from rank src to all other ranks."""
    for p in params:
        _ = dist.broadcast(p.data, src=src)


def worker_init_fn(worker_id: int) -> None:
    """Hide CUDA from DataLoader workers so they only touch CPU-resident tensors."""
    os.environ["CUDA_VISIBLE_DEVICES"] = ""


def spawn_distributed_workers(
    worker_fn, devices: list[int], *worker_args, start_method: str = "forkserver"
) -> None:
    """Launch one worker per device via torch.multiprocessing.start_processes.

    worker_fn must be a module-level callable with signature
    (rank, world_size, *worker_args) — matches the mp.start_processes contract,
    which passes rank as the first arg automatically.
    """
    import torch.multiprocessing as mp

    mp.start_processes(  # type: ignore
        worker_fn,
        args=(len(devices), *worker_args),
        nprocs=len(devices),
        join=True,
        start_method=start_method,
    )
