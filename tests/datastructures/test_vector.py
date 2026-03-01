import zipfile

import numpy as np
import pytest

from quantem.core.io.serialize import load
from quantem.core.datastructures.vector import Vector



def make_line_vector() -> Vector:
    v = Vector.from_shape(
        shape=(4,),
        fields=["intensity", "kx", "ky"],
        units=["a.u.", "px", "px"],
        name="line",
    )
    v[0] = np.array([[1.0, 10.0, 100.0], [2.0, 20.0, 200.0]])
    v[1] = np.array([[3.0, 30.0, 300.0]])
    v[2] = np.array([[4.0, 40.0, 400.0], [5.0, 50.0, 500.0]])
    v[3] = np.array([[6.0, 60.0, 600.0]])
    return v



def make_grid_vector() -> Vector:
    v = Vector.from_shape(shape=(3, 2), fields=["intensity", "kx", "ky"])
    for i in range(3):
        for j in range(2):
            base = float(i * 10 + j)
            v[i, j] = np.array([[base, base + 100.0, base + 200.0]])
    return v


class TestVector:
    def test_initialization_and_len(self):
        v1 = Vector.from_shape(shape=(2, 3), fields=["a", "b", "c"])
        assert v1.shape == (2, 3)
        assert len(v1) == 2
        assert v1.num_fields == 3
        assert v1.fields == ["a", "b", "c"]
        assert v1.units == ["none", "none", "none"]
        assert v1.name == "2d ragged array"
        assert v1[0, 0].array.shape == (0, 3)
        np.testing.assert_array_equal(v1[0, 0].flatten(), v1[0, 0].array)

        v2 = Vector.from_shape(shape=(2, 3), num_fields=2)
        assert v2.fields == ["field_0", "field_1"]

        with pytest.raises(TypeError):
            len(v1[0, 0])

        with pytest.raises(ValueError, match="Must specify either 'fields' or 'num_fields'."):
            Vector.from_shape(shape=(2, 3))

        with pytest.raises(ValueError, match="does not match length of fields"):
            Vector.from_shape(shape=(2, 3), num_fields=2, fields=["a", "b", "c"])

        with pytest.raises(ValueError, match="Duplicate field names"):
            Vector.from_shape(shape=(2, 3), fields=["a", "a"])

        assert str(v1) == (
            "quantem.Vector, shape=(2, 3), name=2d ragged array\n"
            "  fields = ['a', 'b', 'c']\n"
            "  units: ['none', 'none', 'none']"
        )

    def test_indexing_and_array_contract(self):
        v = make_grid_vector()

        assert isinstance(v[:2, 1], Vector)
        assert v[:2, 1].shape == (2,)
        assert v[1].shape == (2,)
        assert v[1, 1].shape == ()
        np.testing.assert_array_equal(v[-1, -1].array, np.array([[21.0, 121.0, 221.0]]))

        with pytest.raises(ValueError):
            _ = v[:, 1].array

        result = v[[-1, 0], 1]
        assert result.shape == (2,)
        np.testing.assert_array_equal(result[0].array, np.array([[21.0, 121.0, 221.0]]))
        np.testing.assert_array_equal(result[1].array, np.array([[1.0, 101.0, 201.0]]))

    def test_select_fields_and_chaining_equivalence(self):
        v = make_line_vector()

        selected = v.select_fields("kx")
        assert selected.fields == ["kx"]
        assert selected.units == ["px"]
        assert selected.shape == v.shape

        np.testing.assert_array_equal(
            v.select_fields("kx")[2].array,
            v[2].select_fields("kx").array,
        )

        with pytest.raises(KeyError):
            v.select_fields("missing")

        with pytest.raises(TypeError):
            _ = v["kx"]

        with pytest.raises(TypeError):
            _ = v[1, "kx"]

    def test_array_mutation_writes_through_for_single_field(self):
        v = make_line_vector()
        cell = v.select_fields("kx")[1].array
        cell[0, 0] = 99.0
        assert v[1].array[0, 1] == 99.0

    def test_field_arithmetic_with_scalar_and_ndarray(self):
        v = make_line_vector()

        kx = v.select_fields("kx")
        kx += 10
        np.testing.assert_array_equal(v.select_fields("kx").flatten(), np.array([[20.0], [30.0], [40.0], [50.0], [60.0], [70.0]]))

        v.select_fields("kx")[...] += np.arange(6)
        np.testing.assert_array_equal(v.select_fields("kx").flatten(), np.array([[20.0], [31.0], [42.0], [53.0], [64.0], [75.0]]))

        summed = v.select_fields("intensity") + v.select_fields("ky")
        np.testing.assert_array_equal(
            summed.flatten(),
            np.array([[101.0], [202.0], [303.0], [404.0], [505.0], [606.0]]),
        )

    def test_field_assignment_from_vector_expression(self):
        v = make_line_vector()
        scale = 2.5

        v[:2].select_fields("intensity")[...] = v[2:4].select_fields("intensity") * scale
        np.testing.assert_array_equal(
            v[:2].select_fields("intensity").flatten(),
            np.array([[10.0], [12.5], [15.0]]),
        )

    def test_field_assignment_requires_matching_per_cell_row_counts(self):
        v = make_line_vector()
        with pytest.raises(ValueError, match="Per-cell row counts must match"):
            v[:2].select_fields("intensity")[...] = v[1:3].select_fields("intensity")

    def test_full_cell_assignment_allows_row_count_changes(self):
        v = make_line_vector()

        v[1] = v[0]
        assert v[1].array.shape == (2, 3)
        np.testing.assert_array_equal(v[1].array, v[0].array)

        v[0:2] = v[1:3]
        assert v[0].array.shape == (2, 3)
        assert v[1].array.shape == (2, 3)

        broadcast_cell = np.array([[9.0, 8.0, 7.0]])
        v[[0, 3]] = broadcast_cell
        np.testing.assert_array_equal(v[0].array, broadcast_cell)
        np.testing.assert_array_equal(v[3].array, broadcast_cell)

    def test_boolean_indexing_is_axis_wise(self):
        v = make_grid_vector()

        rows = np.array([True, False, True])
        cols = np.array([False, True])
        selected = v[rows, cols]

        assert selected.shape == (2, 1)
        np.testing.assert_array_equal(selected[0, 0].array, np.array([[1.0, 101.0, 201.0]]))
        np.testing.assert_array_equal(selected[1, 0].array, np.array([[21.0, 121.0, 221.0]]))

        with pytest.raises(IndexError):
            _ = v[np.array([[True, False], [False, True]])]

    def test_empty_selection_is_valid_and_no_op_for_scalar_math(self):
        v = make_grid_vector()
        before = v.copy().flatten()

        empty = v[[], :]
        assert empty.shape == (0, 2)
        assert empty.flatten().shape == (0, 3)

        empty.select_fields("kx")[...] += 1
        np.testing.assert_array_equal(v.flatten(), before)

    def test_add_fields_defaults_expression_and_multiple_values(self):
        v = make_line_vector()

        v.add_fields(("h", "k"))
        assert v.fields == ["intensity", "kx", "ky", "h", "k"]
        assert np.isnan(v[0].array[:, 3:5]).all()

        v.add_fields("field_out", v.select_fields("kx") + v.select_fields("ky"))
        np.testing.assert_array_equal(
            v.select_fields("field_out").flatten(),
            np.array([[110.0], [220.0], [330.0], [440.0], [550.0], [660.0]]),
        )

        v2 = make_line_vector()
        v2.add_fields(("h", "k"), (1.0, np.array([5.0, 6.0, 7.0, 8.0, 9.0, 10.0])))
        np.testing.assert_array_equal(v2.select_fields("h").flatten(), np.ones((6, 1)))
        np.testing.assert_array_equal(
            v2.select_fields("k").flatten(),
            np.array([[5.0], [6.0], [7.0], [8.0], [9.0], [10.0]]),
        )

        with pytest.raises(ValueError, match="all fields are selected"):
            v2.select_fields("kx").add_fields("bad")

    def test_remove_fields_preserves_remaining_data(self):
        v = make_line_vector()
        v.add_fields("extra", 1.0)
        v.remove_fields(("kx", "extra"))

        assert v.fields == ["intensity", "ky"]
        np.testing.assert_array_equal(
            v[0].array,
            np.array([[1.0, 100.0], [2.0, 200.0]]),
        )

    def test_copy_is_deep(self):
        v = make_line_vector()
        v_copy = v.select_fields(["intensity", "kx"]).copy()

        v_copy[0].array[0, 0] = -1.0
        assert v[0].array[0, 0] == 1.0
        assert v_copy.fields == ["intensity", "kx"]
        assert v_copy.shape == (4,)

    def test_from_data_supports_nested_fixed_grid(self):
        data = [
            [np.array([[1.0, 2.0]]), np.array([[3.0, 4.0], [5.0, 6.0]])],
            [np.array([[7.0, 8.0]]), np.array([[9.0, 10.0]])],
        ]
        v = Vector.from_data(data=data, fields=["a", "b"], units=["u1", "u2"], name="nested")

        assert v.shape == (2, 2)
        assert v.fields == ["a", "b"]
        assert v.units == ["u1", "u2"]
        assert v.name == "nested"
        np.testing.assert_array_equal(v[0, 1].array, np.array([[3.0, 4.0], [5.0, 6.0]]))

        tuple_cells = [
            ([1.0, 2.0], [3.0, 4.0]),
            ([5.0, 6.0], [7.0, 8.0], [9.0, 10.0]),
        ]
        tuple_vector = Vector.from_data(data=tuple_cells, fields=["a", "b"])
        assert tuple_vector.shape == (2,)
        np.testing.assert_array_equal(tuple_vector[0].array, np.array([[1.0, 2.0], [3.0, 4.0]]))
        np.testing.assert_array_equal(
            tuple_vector[1].array,
            np.array([[5.0, 6.0], [7.0, 8.0], [9.0, 10.0]]),
        )

        with pytest.raises(TypeError, match="Data must be a list"):
            Vector.from_data(data=np.array([1, 2, 3]))  # type: ignore[arg-type]

        with pytest.raises(ValueError, match="same number of fields"):
            Vector.from_data(data=[np.array([[1.0, 2.0]]), np.array([[1.0, 2.0, 3.0]])])

    def test_save_and_load_round_trip(self, tmp_path):
        v = make_grid_vector()
        v.add_fields("extra", v.select_fields("intensity") + 1.0)

        path = tmp_path / "vector_test.zip"
        v.save(path, mode="o", compression_level=4)

        with zipfile.ZipFile(path) as zf:
            names = [info.filename for info in zf.infolist()]
        assert len(names) < 30
        assert "_state/data/zarr.json" in names
        assert all(not name.startswith("_selection_coords/") for name in names)

        loaded = load(path)
        assert isinstance(loaded, Vector)
        assert loaded.shape == v.shape
        assert loaded.fields == v.fields
        assert loaded.units == v.units
        np.testing.assert_array_equal(loaded[2, 1].array, v[2, 1].array)
