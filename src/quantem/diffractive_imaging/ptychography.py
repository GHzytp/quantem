import contextlib
import copy
import gc
import os
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Self, Sequence, cast
from warnings import warn

import numpy as np
from tqdm.auto import tqdm

from quantem.core import config
from quantem.core.io.serialize import load as autoserialize_load
from quantem.core.ml.dist_utils import (
    init_process_group,
    is_distributed_launch,
)
from quantem.diffractive_imaging.dataset_models import DatasetModelType
from quantem.diffractive_imaging.detector_models import DetectorModelType
from quantem.diffractive_imaging.logger_ptychography import LoggerPtychography
from quantem.diffractive_imaging.object_models import ObjectModelType, ObjectPixelated
from quantem.diffractive_imaging.probe_models import ProbeModelType, ProbeParametric
from quantem.diffractive_imaging.ptycho_utils import SimpleBatcher
from quantem.diffractive_imaging.ptychography_base import PtychographyBase
from quantem.diffractive_imaging.ptychography_opt import PtychographyOpt
from quantem.diffractive_imaging.ptychography_visualizations import PtychographyVisualizations

if TYPE_CHECKING:
    import torch
    import torch.distributed as dist
    import torch.multiprocessing as mp
else:
    if config.get("has_torch"):
        import torch
        import torch.distributed as dist
        import torch.multiprocessing as mp


def _ddp_ptycho_worker(
    rank: int,
    world_size: int,
    ptycho_path: str,
    devices: list[int],
    recon_kwargs: dict[str, Any],
    result_path: str,
) -> None:
    """Module-level worker for mp.start_processes — must live at module scope to be picklable.

    Receives a file path rather than the Ptychography object directly so that no
    large tensors cross the process boundary via pickle (which triggers PyTorch's
    shared-memory tensor mechanism and fails in some Linux environments).
    """
    device_id = devices[rank]
    init_process_group(rank, world_size, backend="nccl" if torch.cuda.is_available() else "gloo")

    ptycho = torch.load(ptycho_path, map_location="cpu", weights_only=False)
    ptycho.dset.shard(rank, world_size)
    ptycho.to(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")

    if dist.is_available() and dist.is_initialized():
        ptycho._broadcast_parameters(src=0)

    ptycho._reconstruct_inner(**recon_kwargs, _dist_rank=rank, _dist_world_size=world_size)

    if rank == 0:
        torch.save(
            {
                "obj_state": {k: v.cpu() for k, v in ptycho.obj_model.state_dict().items()},
                "probe_state": {k: v.cpu() for k, v in ptycho.probe_model.state_dict().items()},
                "iter_losses": ptycho._iter_losses,
                "iter_val_losses": ptycho._iter_val_losses,
            },
            result_path,
        )

    dist.destroy_process_group()


class Ptychography(PtychographyOpt, PtychographyVisualizations, PtychographyBase):
    """
    A class for performing phase retrieval using the Ptychography algorithm.
    """

    @classmethod
    def from_models(
        cls,
        dset: DatasetModelType,
        obj_model: ObjectModelType,
        probe_model: ProbeModelType,
        detector_model: DetectorModelType,
        logger: LoggerPtychography | None = None,
        device: str | int = "cpu",  # "gpu" | "cpu" | "cuda:X"
        verbose: int | bool = True,
        rng: np.random.Generator | int | None = None,
    ) -> Self:
        return cls(
            dset=dset,
            obj_model=obj_model,
            probe_model=probe_model,
            detector_model=detector_model,
            logger=logger,
            device=device,
            verbose=verbose,
            rng=rng,
            _token=cls._token,
        )

    @classmethod
    def from_ptychography(
        cls,
        ptycho: Self,
        obj_model: ObjectModelType | None = None,
        probe_model: ProbeModelType | None = None,
        logger: LoggerPtychography | None = None,
    ) -> Self:
        _tmp_logger = ptycho.logger
        ptycho.logger = None
        cloned = ptycho.clone()
        ptycho.logger = _tmp_logger
        if obj_model is not None:
            cloned.obj_model = obj_model
        if probe_model is not None:
            cloned.probe_model = probe_model
        if logger is not None:
            cloned.logger = logger

        cloned.reset_recon()
        return cloned

    # region --- explicit properties and setters ---

    @property
    def autograd(self) -> bool:
        return self._autograd

    @autograd.setter
    def autograd(self, autograd: bool) -> None:
        self._autograd = bool(autograd)

    # endregion --- explicit properties and setters ---

    # region --- methods ---
    # TODO reset RNG as well
    def reset_recon(self) -> None:
        super().reset_recon()
        self.obj_model.reset_optimizer()
        self.probe_model.reset_optimizer()
        self.dset.reset_optimizer()

    def _record_iter(self, iter_loss: float) -> None:
        self._iter_losses.append(iter_loss)
        optimizers = self.optimizers
        all_keys = set(self._iter_lrs.keys()) | set(optimizers.keys())
        for key in all_keys:
            if key in self._iter_lrs.keys():
                if key in optimizers.keys():
                    self._iter_lrs[key].append(optimizers[key].param_groups[0]["lr"])
                else:
                    self._iter_lrs[key].append(0.0)
            else:  # new optimizer
                # For new optimizers, backfill with 0.0 LR for previous iterations
                current_iter = self.num_iters - 1  # -1 because loss was just appended
                prev_lrs = [0.0] * current_iter
                prev_lrs.append(optimizers[key].param_groups[0]["lr"])
                self._iter_lrs[key] = prev_lrs

    def _reset_iter_constraints(self) -> None:
        """Reset constraint loss accumulation for all models."""
        self.obj_model.reset_iter_constraint_losses()
        self.probe_model.reset_iter_constraint_losses()
        self.dset.reset_iter_constraint_losses()

    def _soft_constraints(self) -> torch.Tensor:
        """Calculate soft constraints by calling apply_soft_constraints on each model."""
        total_loss = torch.tensor(0, device=self.device, dtype=self._dtype_real)

        obj_loss = self.obj_model.apply_soft_constraints(
            self.obj_model.obj, mask=self.obj_model.mask
        )
        total_loss += obj_loss

        probe_loss = self.probe_model.apply_soft_constraints(self.probe_model.probe)
        total_loss += probe_loss

        dataset_loss = self.dset.apply_soft_constraints(self.dset.descan_shifts)
        total_loss += dataset_loss

        return total_loss

    # endregion --- methods ---

    # region --- reconstruction ---

    def reconstruct(
        self,
        num_iters: int = 0,
        reset: bool = False,
        optimizer_params: dict | None = None,
        scheduler_params: dict | None = None,
        constraints: dict = {},
        batch_size: int | None = None,
        store_snapshots: bool | None = None,
        store_snapshots_every: int | None = None,
        device: Literal["cpu", "gpu"] | int | list[int] | None = None,
        autograd: bool = True,
        loss_type: Literal[
            "l2_amplitude", "l1_amplitude", "l2_intensity", "l1_intensity", "poisson"
        ] = "l2_amplitude",
    ) -> Self:
        """Run iterative ptychography reconstruction.

        ``device`` accepts:
          - ``None``             — keep current device
          - ``"cpu"`` / ``"gpu"`` — existing string form
          - ``int``              — specific GPU index, e.g. ``device=2`` → cuda:2
          - ``list[int]``        — multi-GPU, e.g. ``device=[0,1,2,3]``

        Multi-GPU (``device`` is a list) launches worker processes via ``mp.spawn`` when called
        from a notebook, or uses the existing distributed process group when launched with
        ``torchrun``. Only autograd mode is supported for multi-GPU in this release.
        """
        self._check_preprocessed()

        # Route to multi-GPU path when a list of device IDs is given
        if isinstance(device, list):
            if not autograd:
                raise ValueError("Multi-GPU reconstruction requires autograd=True.")
            if not is_distributed_launch():
                return self._spawn_reconstruct(
                    devices=device,
                    num_iters=num_iters,
                    reset=reset,
                    optimizer_params=optimizer_params,
                    scheduler_params=scheduler_params,
                    constraints=constraints,
                    batch_size=batch_size,
                    store_snapshots=store_snapshots,
                    store_snapshots_every=store_snapshots_every,
                    autograd=autograd,
                    loss_type=loss_type,
                )
            # torchrun: fall through — process group already initialised externally

        # Handle torchrun distributed launch (RANK env var present)
        if is_distributed_launch():
            rank = int(os.environ["RANK"])
            world_size = int(os.environ["WORLD_SIZE"])
            if not torch.distributed.is_initialized():
                torch.distributed.init_process_group(
                    backend="nccl" if torch.cuda.is_available() else "gloo",
                    init_method="env://",
                )
            local_rank = int(os.environ.get("LOCAL_RANK", rank))
            dev = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
            self.to(dev)
            self.dset.shard(rank, world_size)
            self._broadcast_parameters(src=0)
        else:
            rank, world_size = 0, 1
            if device is not None and not isinstance(device, list):
                self.to(device)

        return self._reconstruct_inner(
            num_iters=num_iters,
            reset=reset,
            optimizer_params=optimizer_params,
            scheduler_params=scheduler_params,
            constraints=constraints,
            batch_size=batch_size,
            store_snapshots=store_snapshots,
            store_snapshots_every=store_snapshots_every,
            autograd=autograd,
            loss_type=loss_type,
            _dist_rank=rank,
            _dist_world_size=world_size,
        )

    def _reconstruct_inner(
        self,
        num_iters: int = 0,
        reset: bool = False,
        optimizer_params: dict | None = None,
        scheduler_params: dict | None = None,
        constraints: dict = {},
        batch_size: int | None = None,
        store_snapshots: bool | None = None,
        store_snapshots_every: int | None = None,
        autograd: bool = True,
        loss_type: Literal[
            "l2_amplitude", "l1_amplitude", "l2_intensity", "l1_intensity", "poisson"
        ] = "l2_amplitude",
        _dist_rank: int = 0,
        _dist_world_size: int = 1,
    ) -> Self:
        """Core reconstruction loop. Called by reconstruct() for all launch modes."""
        self.batch_size = batch_size
        self.store_snapshot_every = store_snapshots_every
        if store_snapshots_every is not None and store_snapshots is None:
            self.store_snapshots = True
        else:
            self.store_snapshots = store_snapshots

        if reset:
            self.reset_recon()
        self.constraints = constraints

        new_scheduler = reset
        if optimizer_params is not None:
            self.optimizer_params = optimizer_params
            self.set_optimizers()
            new_scheduler = True

        if scheduler_params is not None:
            self.scheduler_params = scheduler_params
            new_scheduler = True

        if new_scheduler:
            self.set_schedulers(self.scheduler_params, num_iter=num_iters)

        self.dset._set_targets(loss_type)
        self.compute_propagator_arrays()  # required to avoid issue if stopped learning probe tilt

        # Compute the global scan count once — needed to keep loss scale consistent across world sizes.
        # In single-GPU mode num_gpts is already global; in distributed mode each rank holds a shard,
        # so we sum across ranks to get the exact total.
        if _dist_world_size > 1 and dist.is_available() and dist.is_initialized():
            n_local = torch.tensor(self.dset.num_gpts, device=self.device, dtype=torch.long)
            dist.all_reduce(n_local, op=dist.ReduceOp.SUM)
            global_n = int(n_local.item())
        else:
            global_n = self.dset.num_gpts

        batcher = SimpleBatcher(
            self.dset.num_gpts,
            self.batch_size,
            rng=self.rng,
            val_ratio=self.val_ratio,
            val_mode=self.val_mode,
        )
        pbar = tqdm(range(num_iters), disable=not self.verbose or _dist_rank != 0)

        for a0 in pbar:
            consistency_loss = 0.0
            total_loss = 0.0
            self._reset_iter_constraints()

            for batch_indices in batcher:
                self.zero_grad_all()
                patch_indices, _positions_px, positions_px_fractional, descan_shifts = (
                    self.dset.forward(batch_indices, self.obj_padding_px)
                )
                shifted_probes = self.probe_model.forward(positions_px_fractional)
                obj_patches = self.obj_model.forward(patch_indices)
                propagated_probes, overlap = self.forward_operator(
                    obj_patches, shifted_probes, descan_shifts
                )
                pred_intensities = self.detector_model.forward(overlap)

                batch_consistency_loss, targets = self.error_estimate(
                    pred_intensities,
                    batch_indices,
                    loss_type=loss_type,
                    global_n=global_n,
                )

                batch_soft_constraint_loss = self._soft_constraints()
                batch_loss = batch_consistency_loss + batch_soft_constraint_loss

                self.backward(
                    batch_loss,
                    autograd,
                    obj_patches,
                    propagated_probes,
                    overlap,
                    patch_indices,
                    targets,
                )
                if _dist_world_size > 1:
                    self._all_reduce_gradients()
                self.step_optimizers()
                consistency_loss += batch_consistency_loss.item()
                total_loss += batch_loss.item()

            num_batches = len(batcher)
            total_loss = total_loss / num_batches
            consistency_loss = consistency_loss / num_batches

            # Average loss across ranks so rank-0 reports the global mean
            if _dist_world_size > 1:
                loss_t = torch.tensor(
                    [total_loss, consistency_loss], device=self.device, dtype=torch.float64
                )
                dist.all_reduce(loss_t, op=dist.ReduceOp.AVG)
                total_loss, consistency_loss = loss_t[0].item(), loss_t[1].item()

            # Validation pass (no gradient, no optimizer steps)
            val_loss = None
            if batcher.has_validation:
                val_consistency_loss = 0.0
                val_batches = 0
                with torch.no_grad():
                    for batch_indices in batcher.iter_val():
                        patch_indices, _positions_px, positions_px_fractional, descan_shifts = (
                            self.dset.forward(batch_indices, self.obj_padding_px)
                        )
                        shifted_probes = self.probe_model.forward(positions_px_fractional)
                        obj_patches = self.obj_model.forward(patch_indices)
                        _propagated_probes, overlap = self.forward_operator(
                            obj_patches, shifted_probes, descan_shifts
                        )
                        pred_intensities = self.detector_model.forward(overlap)
                        batch_val_loss, _ = self.error_estimate(
                            pred_intensities, batch_indices, loss_type=loss_type, global_n=global_n
                        )
                        val_consistency_loss += batch_val_loss.item()
                        val_batches += 1
                if val_batches > 0:
                    val_loss = val_consistency_loss / val_batches
                    if _dist_rank == 0:
                        self._iter_val_losses.append(val_loss)

            if _dist_rank == 0:
                self._record_iter(total_loss)  # TODO record val loss as well

            # Step schedulers with current loss
            self.step_schedulers(total_loss)

            if _dist_rank == 0 and self.store_snapshots and (a0 % self.store_snapshot_every) == 0:
                self._store_current_iter_snapshot()

            if _dist_rank == 0 and self.logger is not None:
                self.logger.log_iter(
                    self.obj_model,
                    self.probe_model,
                    self.dset,
                    self.num_iters - 1,
                    consistency_loss,
                    num_batches,
                    self._get_current_lrs(),
                )

            if _dist_rank == 0:
                if val_loss is not None:
                    pbar.set_description(
                        f"Iter {a0 + 1}/{num_iters}, Loss: {total_loss:.3e}, Val: {val_loss:.3e}"
                    )
                else:
                    pbar.set_description(f"Iter {a0 + 1}/{num_iters}, Loss: {total_loss:.3e}")

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if hasattr(torch, "mps") and torch.backends.mps.is_available():
            torch.mps.empty_cache()
        gc.collect()

        return self

    def _spawn_reconstruct(self, devices: list[int], **recon_kwargs) -> Self:
        """Notebook multi-GPU: spawn one worker process per device via forkserver.

        State is saved to a temp file so that no tensors cross the process boundary
        via pickle.  PyTorch's ForkingPickler automatically moves all CPU tensors to
        shared memory when pickling for multiprocessing, which fails on some Linux
        systems (EINVAL from ftruncate).  Passing only a file path (a plain string)
        avoids that mechanism entirely.
        """
        self.to("cpu")

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            ptycho_path = str(tmpdir_path / "ptycho_state.pt")
            result_path = str(tmpdir_path / "result.pt")

            torch.save(self, ptycho_path, pickle_protocol=4)

            # forkserver: workers fork from a clean pre-started server (no inherited
            # CUDA, no Jupyter FDs).  Only plain Python scalars/strings cross the
            # process boundary, so tensor pickling is never triggered.
            mp.start_processes(
                _ddp_ptycho_worker,
                args=(len(devices), ptycho_path, devices, recon_kwargs, result_path),
                nprocs=len(devices),
                join=True,
                start_method="forkserver",
            )
            result = torch.load(result_path, map_location="cpu", weights_only=False)

        self.obj_model.load_state_dict(result["obj_state"])
        self.probe_model.load_state_dict(result["probe_state"])
        self._iter_losses.extend(result["iter_losses"])
        self._iter_val_losses.extend(result["iter_val_losses"])
        return self

    def _get_current_lrs(self) -> dict[str, float]:
        return {
            param_name: optimizer.param_groups[0]["lr"]
            for param_name, optimizer in self.optimizers.items()
            if optimizer is not None
        }

    def backward(
        self,
        loss: torch.Tensor,
        autograd: bool,
        obj_patches: torch.Tensor,
        propagated_probes: torch.Tensor,
        overlap: torch.Tensor,
        patch_indices: torch.Tensor,
        amplitudes: torch.Tensor,
    ):
        if autograd:
            loss.backward()
            # scaling pixelated ad gradients to closer match analytic
            if isinstance(self.obj_model, ObjectPixelated):
                obj_grad_scale = self.dset.upsample_factor**2 / 2  # factor of 2 from l2 grad
                self.obj_model._obj.grad.mul_(obj_grad_scale)  # type:ignore

            if isinstance(self.probe_model, ProbeParametric):
                probe_grad_scale = np.sqrt(self.probe_model._mean_diffraction_intensity)
                for par in self.probe_model.params:
                    par.grad.mul_(probe_grad_scale)  # type:ignore

        else:
            gradient = self.gradient_step(amplitudes, overlap)
            prop_gradient = self.obj_model.backward(
                gradient,
                obj_patches,
                propagated_probes,
                self._propagators,
                patch_indices,
            )
            self.probe_model.backward(prop_gradient, obj_patches)

    def gradient_step(self, amplitudes, overlap):
        """Computes analytical gradient using the Fourier projection modified overlap"""
        modified_overlap = self.fourier_projection(amplitudes, overlap)
        ## mod_overlap shape: (nprobes, batch_size, roi_shape[0], roi_shape[1])
        ## grad shape: (nprobes, batch_size, roi_shape[0], roi_shape[1])
        return modified_overlap - overlap

    def fourier_projection(self, measured_amplitudes, overlap_array):
        """Replaces the Fourier amplitude of overlap with the measured data."""
        # corner centering measured amplitudes
        measured_amplitudes = torch.fft.fftshift(measured_amplitudes, dim=(-2, -1))
        fourier_overlap = torch.fft.fft2(overlap_array, norm="ortho")
        if self.num_probes == 1:  # faster
            fourier_modified_overlap = measured_amplitudes * torch.exp(
                1.0j * torch.angle(fourier_overlap)
            )
        else:  # necessary for mixed state # TODO check this with normalization
            farfield_amplitudes = self.estimate_amplitudes(overlap_array, corner_centered=True)
            farfield_amplitudes[farfield_amplitudes == 0] = torch.inf
            amplitude_modification = measured_amplitudes / farfield_amplitudes
            fourier_modified_overlap = amplitude_modification[None] * fourier_overlap

        return torch.fft.ifft2(fourier_modified_overlap, norm="ortho")

    # endregion --- reconstruction ---

    def save(
        self,
        path: str | Path,
        mode: Literal["w", "o"] = "w",
        store: Literal["auto", "zip", "dir"] = "auto",
        skip: str | type | Sequence[str | type] = (),
        compression_level: int | None = 4,
        save_raw_data: bool = False,
        verbose: int | bool = True,
    ):
        """
        Save the ptychography object, optionally excluding raw dataset data.

        By default, this method saves the ptychography object without the raw dataset
        to save space and allow for dataset reloading. Use save_raw_data=True if you
        want to include the complete dataset.

        When saving without raw data, the system automatically saves:
        - Dataset file path and file type
        - All preprocessing parameters (CoM fitting, rotation, padding, etc.)
        - Reconstruction state (losses, constraints, etc.)

        On load, if no dataset is provided, the system will automatically:
        - Reload the dataset from the saved file path
        - Reapply all preprocessing with the exact same parameters
        - Restore the reconstruction state

        Parameters
        ----------
        path : str | Path
            Path to save the object
        mode : Literal["w", "o"]
            Write mode ('w' for write, 'o' for overwrite)
        store : Literal["auto", "zip", "dir"]
            Storage format
        skip : str | type | Sequence[str | type]
            Additional items to skip during serialization
        compression_level : int | None
            Compression level for zip storage
        save_raw_data : bool
            Whether to save the raw dataset data (default: False)

        Examples
        --------
        # Save without raw data (default behavior) - includes dataset metadata
        ptycho.save("my_reconstruction.zip")

        # Save with raw data included
        ptycho.save("my_reconstruction_with_data.zip", save_raw_data=True)

        # Load a saved reconstruction - automatically reloads dataset
        loaded_ptycho = Ptychography.from_file("my_reconstruction.zip")

        # Load and move to GPU
        loaded_ptycho = Ptychography.from_file("my_reconstruction.zip", device="gpu")

        # Load with custom dataset (overrides automatic reloading)
        loaded_ptycho = Ptychography.from_file("my_reconstruction.zip", dset=my_dataset)

        """
        if isinstance(skip, (str, type)):
            skip = [skip]
        skip = list(skip)

        # Always skip raw dataset data unless explicitly requested
        if not save_raw_data:
            skip.extend(
                [
                    "_dset",  # Skip the dataset object itself
                    "dset",  # Skip dataset references
                ]
            )

            # Save dataset metadata for automatic reloading
            self._dataset_metadata = {
                "file_path": str(self.dset.dset.file_path) if self.dset.dset.file_path else None,
                "preprocessing_params": self.dset._preprocessing_params,
                "learned_scan_positions_px": self.dset.scan_positions_px.data.cpu(),
                "learned_descan_shifts": self.dset.descan_shifts.data.cpu(),
            }

        # Add other common skips for ptychography objects
        skips = skip

        current_device = self.device
        self.to("cpu")

        if self.verbose and verbose:
            print(f"Saving ptychography object to {Path(path).resolve()}")

        super().save(
            path,
            mode=mode,
            store=store,
            skip=skips,
            compression_level=compression_level,
        )

        self.to(current_device)  # TODO figure out why this isn't working for DDIP sometimes?

        # Clean up temporary metadata
        if not save_raw_data and hasattr(self, "_dataset_metadata"):
            delattr(self, "_dataset_metadata")

    @classmethod
    def from_file(
        cls,
        path: str | Path,
        dset: DatasetModelType | None = None,
        device: str | int | None = None,
        verbose: int | bool | None = None,
        auto_reload_dataset: bool = True,
    ):
        """
        Load a ptychography object from a saved file.

        Parameters
        ----------
        path : str | Path
            Path to the saved ptychography object
        dset : DatasetModelType | None
            Dataset to use (if None and auto_reload_dataset=True, will try to reload from saved metadata)
        device : str | int | None
            Device to load the object on
        verbose : int | bool | None
            Verbosity level
        rng : np.random.Generator | int | None
            Random number generator
        auto_reload_dataset : bool
            Whether to automatically reload and preprocess the dataset from saved metadata

        Returns
        -------
        Ptychography
            Loaded ptychography object
        """
        # Load the base object without the dataset
        ptycho = cls._recursive_load_from_path(path)

        if not isinstance(ptycho, Ptychography):
            raise ValueError("Loaded object is not a Ptychography object")

        # If no dataset was provided, try to reload it from saved metadata
        if dset is None and auto_reload_dataset and not hasattr(ptycho, "dset"):
            if hasattr(ptycho, "_dataset_metadata") and ptycho._dataset_metadata:
                metadata = ptycho._dataset_metadata
                file_path = metadata.get("file_path")

                if file_path:
                    # Import here to avoid circular imports
                    from quantem.core.io.file_readers import read_4dstem
                    from quantem.diffractive_imaging.dataset_models import (
                        PtychographyDatasetRaster,
                    )

                    # Reload the dataset
                    print(f"reloading dataset from {file_path}", end="\r")
                    try:
                        raw_dset = read_4dstem(file_path)
                    except (ValueError, ModuleNotFoundError) as _e:
                        try:
                            raw_dset = autoserialize_load(file_path)
                            raw_dset.file_path = file_path  # legacy support
                        except Exception as e:
                            raise ValueError(
                                f"Could not automatically reload dataset from {file_path}: {e}"
                            )

                    dset = PtychographyDatasetRaster.from_dataset4dstem(
                        raw_dset, verbose=verbose or 1
                    )
                    # Apply preprocessing with saved parameters
                    preprocessing_params = metadata.get("preprocessing_params", {})
                    _v = dset.verbose
                    dset.verbose = 0
                    dset.preprocess(**preprocessing_params)
                    dset.verbose = _v

                    print(f"Successfully reloaded dataset from {file_path}")
                else:
                    dset = None
            else:
                print("Warning: No dataset metadata found in saved object.")
                dset = None
        elif dset is not None:
            dset._set_initial_scan_positions_px(ptycho.obj_padding_px)
            dset._set_patch_indices(ptycho.obj_padding_px)
            if hasattr(ptycho, "_dataset_metadata") and ptycho._dataset_metadata:
                metadata = ptycho._dataset_metadata
                # preserve learned scan positions and descan shifts
                if "learned_scan_positions_px" in metadata:
                    dset.scan_positions_px.data = metadata["learned_scan_positions_px"]
                if "learned_descan_shifts" in metadata:
                    dset.descan_shifts.data = metadata["learned_descan_shifts"]

        # check if dset was attached to ptycho object
        if dset is not None:
            ptycho.dset = dset
        elif not (hasattr(ptycho, "_dset") and ptycho._dset is not None):
            warn(
                "No dataset provided and could not automatically reload dataset.\n"
                "Please provide a dataset parameter or ensure the object was saved with dataset metadata.\n"
                "Many functionalities will not work without the dataset attached."
            )
            # raise ValueError(
            #     "No dataset provided and could not automatically reload dataset. "
            #     "Please provide a dataset parameter or ensure the object was saved with dataset metadata."
            # )

        if device is not None:
            ptycho.to(device)

        return ptycho

    @classmethod
    def _recursive_load_from_path(cls, path: str | Path):
        """Helper method to load an object from a path using AutoSerialize."""
        return autoserialize_load(path)

    def clone(self, device: str | int = "cpu") -> Self:  # TODO make this faster
        """
        Create a deep-copy clone of this Ptychography instance.

        The clone is placed on CPU by default (device="cpu"). You can override
        the output device by passing a different device string.

        This method first attempts a Python deepcopy for speed. If that fails
        (e.g., due to non-copyable objects), it falls back to serializing the
        object to a temporary file and reloading it, which is robust and includes
        the dataset by default.
        """
        try:
            cloned: Self = copy.deepcopy(self)
        except Exception:
            # Robust fallback: save then reload including raw dataset data so that
            # the in-memory state is fully preserved without relying on external files.
            tmp_path = (
                Path(tempfile.gettempdir()) / f"ptycho_clone_{self.rng.integers(int(1e7))}.zip"
            )
            try:
                self.save(tmp_path, mode="o", store="zip", save_raw_data=True, verbose=0)
                cloned = cast(
                    Self, Ptychography.from_file(tmp_path, device=None, auto_reload_dataset=False)
                )
            finally:
                with contextlib.suppress(Exception):
                    tmp_path.unlink()

        if self.logger is not None:
            cloned.logger = self.logger.clone()

        cloned.to(device)
        return cloned
