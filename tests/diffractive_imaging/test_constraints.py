"""Tests for the ptychography constraint dataclass API."""

import numpy as np
import pytest

from quantem.core.datastructures import Dataset4dstem
from quantem.diffractive_imaging import (
    DetectorPixelated,
    ObjectPixelated,
    ProbePixelated,
    PtychoDatasetConstraintParams,
    PtychoObjConstraintParams,
    PtychoProbeConstraintParams,
    Ptychography,
    PtychographyDatasetRaster,
)

N_SCAN = 8
N_DET = 16
PROBE_ENERGY = 80e3
PROBE_SEMIANGLE = 20
PROBE_DEFOCUS = 100


@pytest.fixture
def ptycho():
    rng = np.random.default_rng(42)
    array = rng.random((N_SCAN, N_SCAN, N_DET, N_DET)).astype(np.float32)
    dset = Dataset4dstem.from_array(
        array,
        name="test",
        sampling=[1.0, 1.0, 0.05, 0.05],
        units=["A", "A", "A^-1", "A^-1"],
    )
    pdset = PtychographyDatasetRaster.from_dataset4dstem(dset)
    pdset.preprocess(com_fit_function="constant", plot_rotation=False, plot_com=False)
    obj = ObjectPixelated.from_uniform(obj_type="pure_phase", num_slices=1)
    probe = ProbePixelated.from_params(
        probe_params={
            "energy": PROBE_ENERGY,
            "defocus": PROBE_DEFOCUS,
            "semiangle_cutoff": PROBE_SEMIANGLE,
        }
    )
    p = Ptychography.from_models(
        dset=pdset,
        obj_model=obj,
        probe_model=probe,
        detector_model=DetectorPixelated(),
        verbose=False,
        rng=42,
    )
    p.preprocess(obj_padding_px=(4, 4))
    return p


# --- parse_dict tests ---------------------------------------------------------


class TestParseDict:
    def test_object_raster_by_name(self):
        c = PtychoObjConstraintParams.parse_dict({"name": "raster", "tv_weight_z": 5.0})
        assert isinstance(c, PtychoObjConstraintParams.Raster)
        assert c.tv_weight_z == 5.0
        assert c.positivity is True  # default preserved

    def test_object_inr_by_type(self):
        c = PtychoObjConstraintParams.parse_dict({"type": "inr"})
        assert isinstance(c, PtychoObjConstraintParams.INR)

    def test_object_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown object constraint type"):
            PtychoObjConstraintParams.parse_dict({"name": "nope"})

    def test_object_missing_name_raises(self):
        with pytest.raises(ValueError, match="Must provide either 'name' or 'type'"):
            PtychoObjConstraintParams.parse_dict({"tv_weight_z": 5.0})

    def test_probe_raster_with_fields(self):
        c = PtychoProbeConstraintParams.parse_dict(
            {"name": "raster", "center_probe": True, "tv_weight": 0.1}
        )
        assert isinstance(c, PtychoProbeConstraintParams.Raster)
        assert c.center_probe is True
        assert c.tv_weight == 0.1

    def test_dataset_raster_default(self):
        c = PtychoDatasetConstraintParams.parse_dict({"name": "raster"})
        assert isinstance(c, PtychoDatasetConstraintParams.Raster)
        assert c.clip_scan_positions is True  # default preserved


# --- Constraint typo catching -------------------------------------------------


class TestTypoCatching:
    def test_setting_unknown_field_via_dict_raises(self, ptycho):
        with pytest.raises(KeyError, match="Invalid constraint key"):
            ptycho.obj_model.constraints = {"not_a_real_field": True}

    def test_add_constraint_unknown_key_raises(self, ptycho):
        with pytest.raises(KeyError, match="Invalid constraint key"):
            ptycho.obj_model.add_constraint("not_a_real_field", True)


# --- Round-trip: pass dataclass via reconstruct(), read back through getter ---


class TestRoundtrip:
    def test_obj_constraints_dataclass(self, ptycho):
        obj_c = PtychoObjConstraintParams.Raster(tv_weight_z=2.5, identical_slices=True)
        ptycho.constraints = {"object": obj_c}
        assert ptycho.obj_model.constraints is obj_c
        assert ptycho.obj_model.constraints.tv_weight_z == 2.5
        assert ptycho.obj_model.constraints.identical_slices is True

    def test_probe_constraints_dataclass(self, ptycho):
        probe_c = PtychoProbeConstraintParams.Raster(center_probe=True, tv_weight=0.05)
        ptycho.constraints = {"probe": probe_c}
        assert ptycho.probe_model.constraints is probe_c

    def test_dataset_constraints_dataclass(self, ptycho):
        dset_c = PtychoDatasetConstraintParams.Raster(descan_tv_weight=0.01)
        ptycho.constraints = {"dataset": dset_c}
        assert ptycho.dset.constraints is dset_c

    def test_dict_form_still_works(self, ptycho):
        """Backward compatibility: nested-dict form sets individual fields."""
        ptycho.constraints = {
            "object": {"tv_weight_z": 3.0, "positivity": False},
            "probe": {"tv_weight": 0.02},
        }
        assert ptycho.obj_model.constraints.tv_weight_z == 3.0
        assert ptycho.obj_model.constraints.positivity is False
        assert ptycho.probe_model.constraints.tv_weight == 0.02


# --- Reconstruct() kwargs and mutual exclusion --------------------------------


class TestReconstructKwargs:
    def test_obj_constraints_kwarg_applied(self, ptycho):
        from quantem.core.ml import OptimizerParams

        obj_c = PtychoObjConstraintParams.Raster(tv_weight_z=1.5)
        ptycho.reconstruct(
            num_iters=1,
            reset=True,
            optimizer_params={"object": OptimizerParams.Adam(lr=1e-2)},
            obj_constraints=obj_c,
            batch_size=4,
            device="cpu",
        )
        assert ptycho.obj_model.constraints.tv_weight_z == 1.5

    def test_dict_kwarg_parsed(self, ptycho):
        from quantem.core.ml import OptimizerParams

        ptycho.reconstruct(
            num_iters=1,
            reset=True,
            optimizer_params={"object": OptimizerParams.Adam(lr=1e-2)},
            obj_constraints={"name": "raster", "surface_zero_weight": 0.7},
            batch_size=4,
            device="cpu",
        )
        assert ptycho.obj_model.constraints.surface_zero_weight == 0.7

    def test_mutual_exclusion_with_legacy_dict(self, ptycho):
        with pytest.raises(ValueError, match="provided via both"):
            ptycho.reconstruct(
                num_iters=0,
                constraints={"object": {"tv_weight_z": 1.0}},
                obj_constraints=PtychoObjConstraintParams.Raster(tv_weight_z=2.0),
            )
