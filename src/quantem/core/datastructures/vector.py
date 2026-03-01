from __future__ import annotations

import copy
from typing import Any, Sequence, overload

import numpy as np
from numpy.typing import NDArray

from quantem.core.io.serialize import AutoSerialize
from quantem.core.utils.validators import (
    validate_fields,
    validate_num_fields,
    validate_shape,
    validate_vector_units,
)


class Vector(AutoSerialize):
    """Ragged cell data on a fixed grid.

    Storage is compact and AutoSerialize-friendly:
    - one 2D numeric row buffer for all ragged rows
    - one start array and one length array for cell boundaries
    - optional selection state for views
    """

    def __init__(
        self,
        shape: tuple[int, ...],
        fields: Sequence[str],
        units: Sequence[str] | None = None,
        name: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        root_shape = validate_shape(shape)
        root_fields = validate_fields(list(fields))
        root_units = validate_vector_units(
            list(units) if units is not None else None,
            len(root_fields),
        )

        self._state = {
            "shape": root_shape,
            "fields": list(root_fields),
            "units": list(root_units),
            "name": name or f"{len(root_shape)}d ragged array",
            "metadata": dict(metadata or {}),
            "data": np.empty((0, len(root_fields)), dtype=float),
            "cell_starts": np.zeros(_cell_count(root_shape), dtype=np.int64),
            "cell_lengths": np.zeros(_cell_count(root_shape), dtype=np.int64),
        }
        self._selection_shape = root_shape
        self._selection_indices: NDArray[np.int64] | None = None
        self._selected_fields: tuple[str, ...] | None = None

    @classmethod
    def _from_view(
        cls,
        state: dict[str, Any],
        selection_shape: tuple[int, ...],
        selection_indices: NDArray[np.int64] | None,
        selected_fields: tuple[str, ...] | None,
    ) -> "Vector":
        obj = cls.__new__(cls)
        obj._state = state
        obj._selection_shape = selection_shape
        obj._selection_indices = None if selection_indices is None else selection_indices.astype(np.int64, copy=False)
        obj._selected_fields = selected_fields
        return obj

    @classmethod
    def from_shape(
        cls,
        shape: tuple[int, ...],
        num_fields: int | None = None,
        fields: Sequence[str] | None = None,
        units: Sequence[str] | None = None,
        name: str | None = None,
    ) -> "Vector":
        if fields is not None:
            root_fields = validate_fields(list(fields))
            if num_fields is not None and len(root_fields) != num_fields:
                raise ValueError(
                    f"num_fields ({num_fields}) does not match length of fields ({len(root_fields)})"
                )
        elif num_fields is not None:
            count = validate_num_fields(num_fields)
            root_fields = [f"field_{i}" for i in range(count)]
        else:
            raise ValueError("Must specify either 'fields' or 'num_fields'.")

        return cls(shape=shape, fields=root_fields, units=units, name=name)

    @classmethod
    def from_data(
        cls,
        data: list[Any],
        num_fields: int | None = None,
        fields: Sequence[str] | None = None,
        units: Sequence[str] | None = None,
        name: str | None = None,
    ) -> "Vector":
        root_shape, cell_arrays = _normalize_nested_data(data)
        inferred_counts = {array.shape[1] for array in cell_arrays}
        if len(inferred_counts) > 1:
            raise ValueError("All cell arrays must have the same number of fields.")
        inferred_fields = cell_arrays[0].shape[1] if cell_arrays else 0

        if fields is not None:
            root_fields = validate_fields(list(fields))
            if len(root_fields) != inferred_fields:
                raise ValueError(
                    f"num_fields ({inferred_fields}) does not match length of fields ({len(root_fields)})"
                )
        elif num_fields is not None:
            count = validate_num_fields(num_fields)
            if count != inferred_fields:
                raise ValueError(
                    f"Provided num_fields ({count}) does not match inferred ({inferred_fields})."
                )
            root_fields = [f"field_{i}" for i in range(count)]
        else:
            root_fields = [f"field_{i}" for i in range(inferred_fields)]

        vector = cls(shape=root_shape, fields=root_fields, units=units, name=name)
        vector._replace_cells(np.arange(len(cell_arrays), dtype=np.int64), cell_arrays)
        return vector

    @property
    def shape(self) -> tuple[int, ...]:
        return self._selection_shape

    @property
    def fields(self) -> list[str]:
        if self._selected_fields is None:
            return list(self._state["fields"])
        return list(self._selected_fields)

    @property
    def units(self) -> list[str]:
        lookup = {
            field: unit
            for field, unit in zip(self._state["fields"], self._state["units"])
        }
        return [lookup[field] for field in self.fields]

    @property
    def num_fields(self) -> int:
        return len(self.fields)

    @property
    def name(self) -> str:
        return self._state["name"]

    @name.setter
    def name(self, value: str) -> None:
        self._state["name"] = str(value)

    @property
    def metadata(self) -> dict[str, Any]:
        return self._state["metadata"]

    @property
    def array(self) -> NDArray[np.generic]:
        if self.shape != ():
            raise ValueError(".array is only valid when the selection contains exactly one cell.")
        cell = self._cell_matrix(self._selected_cell_indices()[0])
        cols = self._field_indices()
        if cols.size == self._full_num_fields:
            return cell
        if cols.size == 1:
            col = int(cols[0])
            return cell[:, col : col + 1]
        if _is_contiguous(cols):
            return cell[:, int(cols[0]) : int(cols[-1]) + 1]
        return cell[:, cols].copy()

    def __len__(self) -> int:
        if self.shape == ():
            raise TypeError("len() of unsized 0D Vector")
        return self.shape[0]

    def __repr__(self) -> str:
        return "\n".join(
            [
                f"quantem.Vector, shape={self.shape}, name={self.name}",
                f"  fields = {self.fields}",
                f"  units: {self.units}",
            ]
        )

    __str__ = __repr__

    def copy(self) -> "Vector":
        copied = self.__class__(
            shape=self.shape,
            fields=self.fields,
            units=self.units,
            name=self.name,
            metadata=copy.deepcopy(self.metadata),
        )
        target_cells = copied._selected_cell_indices()
        source_arrays = [self._selected_cell_matrix(index).copy() for index in self._selected_cell_indices()]
        copied._replace_cells(target_cells, source_arrays)
        return copied

    def flatten(self) -> NDArray[np.generic]:
        arrays = [
            self._selected_cell_matrix(index)
            for index in self._selected_cell_indices()
            if self._cell_row_count(index) > 0
        ]
        if arrays:
            return np.vstack(arrays)

        dtype = self._state["data"].dtype if self._state["data"].ndim == 2 else float
        return np.empty((0, self.num_fields), dtype=dtype)

    def select_fields(self, field_names: str | Sequence[str]) -> "Vector":
        selected = _normalize_field_names(field_names)
        available = set(self.fields)
        missing = [field for field in selected if field not in available]
        if missing:
            raise KeyError(f"Unknown field(s): {missing}")

        if selected == tuple(self._state["fields"]):
            selected_fields = None
        else:
            selected_fields = selected
        return self._from_view(
            self._state,
            self.shape,
            self._selection_indices,
            selected_fields,
        )

    def add_fields(
        self,
        names: str | Sequence[str],
        values: Any | None = None,
        units: str | Sequence[str] | None = None,
    ) -> None:
        self._require_full_field_view("add_fields")
        new_fields = _normalize_field_names(names)
        if any(field in self._state["fields"] for field in new_fields):
            raise ValueError("One or more new field names already exist.")

        new_units = _normalize_units(units, len(new_fields))
        old_fields = list(self._state["fields"])
        self._state["fields"].extend(new_fields)
        self._state["units"].extend(new_units)
        self._expand_storage(len(new_fields))

        if values is None:
            return

        target = self.select_fields(list(new_fields))
        if len(new_fields) > 1 and isinstance(values, (list, tuple)) and len(values) == len(new_fields):
            for field, value in zip(new_fields, values):
                target.select_fields(field)[...] = value
        else:
            target[...] = values

        if self._selected_fields is not None and tuple(old_fields) == self._selected_fields:
            self._selected_fields = None

    def remove_fields(self, names: str | Sequence[str]) -> None:
        self._require_full_field_view("remove_fields")
        to_remove = set(_normalize_field_names(names))
        old_fields = self._state["fields"]
        old_units = self._state["units"]

        missing = [field for field in to_remove if field not in old_fields]
        if missing:
            raise KeyError(f"Unknown field(s): {missing}")
        if len(to_remove) == len(old_fields):
            raise ValueError("Cannot remove all fields from a Vector.")

        keep = [i for i, field in enumerate(old_fields) if field not in to_remove]
        self._state["fields"] = [old_fields[i] for i in keep]
        self._state["units"] = [old_units[i] for i in keep]
        self._state["data"] = self._state["data"][:, keep]

        if self._selected_fields is not None:
            self._selected_fields = tuple(field for field in self._selected_fields if field in self._state["fields"])
            if len(self._selected_fields) == len(self._state["fields"]):
                self._selected_fields = None

    def get_data(self, *indices: Any) -> NDArray[np.generic] | list[NDArray[np.generic]]:
        if len(indices) != len(self.shape):
            raise ValueError(f"Expected {len(self.shape)} indices, got {len(indices)}")
        selection = self[indices if len(indices) != 1 else indices[0]]
        if selection.shape == ():
            return selection.array
        return [selection._selected_cell_matrix(index).copy() for index in selection._selected_cell_indices()]

    def set_data(self, value: Any, *indices: Any) -> None:
        if len(indices) != len(self.shape):
            raise ValueError(f"Expected {len(self.shape)} indices, got {len(indices)}")
        self[indices if len(indices) != 1 else indices[0]] = value

    @overload
    def __getitem__(self, idx: Any) -> "Vector": ...

    def __getitem__(self, idx: Any) -> "Vector":
        if _looks_like_field_selector(idx):
            raise TypeError("Use select_fields(...) for field selection.")
        if idx is Ellipsis:
            return self

        selection_shape, selection_indices = _select_linear_indices(
            self.shape,
            self._selected_cell_indices(),
            idx,
        )
        return self._from_view(
            self._state,
            selection_shape,
            selection_indices,
            self._selected_fields,
        )

    def __setitem__(self, idx: Any, value: Any) -> None:
        if idx is Ellipsis:
            target = self
        else:
            target = self[idx]
        target._assign(value)

    def __add__(self, other: Any) -> "Vector":
        return self._binary_op(other, np.add)

    def __sub__(self, other: Any) -> "Vector":
        return self._binary_op(other, np.subtract)

    def __mul__(self, other: Any) -> "Vector":
        return self._binary_op(other, np.multiply)

    def __truediv__(self, other: Any) -> "Vector":
        return self._binary_op(other, np.divide)

    def __radd__(self, other: Any) -> "Vector":
        return self._binary_op(other, np.add, reverse=True)

    def __rmul__(self, other: Any) -> "Vector":
        return self._binary_op(other, np.multiply, reverse=True)

    def __rsub__(self, other: Any) -> "Vector":
        return self._binary_op(other, np.subtract, reverse=True)

    def __rtruediv__(self, other: Any) -> "Vector":
        return self._binary_op(other, np.divide, reverse=True)

    def __iadd__(self, other: Any) -> "Vector":
        self._inplace_op(other, np.add)
        return self

    def __isub__(self, other: Any) -> "Vector":
        self._inplace_op(other, np.subtract)
        return self

    def __imul__(self, other: Any) -> "Vector":
        self._inplace_op(other, np.multiply)
        return self

    def __itruediv__(self, other: Any) -> "Vector":
        self._inplace_op(other, np.divide)
        return self

    @property
    def _full_num_fields(self) -> int:
        return len(self._state["fields"])

    def _field_indices(self) -> NDArray[np.int64]:
        if self._selected_fields is None:
            return np.arange(self._full_num_fields, dtype=np.int64)

        lookup = {field: i for i, field in enumerate(self._state["fields"])}
        try:
            return np.array([lookup[field] for field in self._selected_fields], dtype=np.int64)
        except KeyError as exc:
            raise KeyError(f"Unknown field(s): {[str(exc.args[0])]}") from exc

    def _require_full_field_view(self, operation: str) -> None:
        if self._selected_fields is not None:
            raise ValueError(f"{operation} is only allowed when all fields are selected.")

    def _selected_cell_indices(self) -> NDArray[np.int64]:
        if self._selection_indices is None:
            return np.arange(_cell_count(self._state["shape"]), dtype=np.int64)
        return self._selection_indices

    def _is_full_field_selection(self) -> bool:
        return self._selected_fields is None

    def _cell_row_count(self, linear_index: int) -> int:
        return int(self._state["cell_lengths"][linear_index])

    def _cell_matrix(self, linear_index: int) -> NDArray[np.generic]:
        start = int(self._state["cell_starts"][linear_index])
        length = int(self._state["cell_lengths"][linear_index])
        if length == 0:
            return self._state["data"][0:0]
        return self._state["data"][start : start + length]

    def _selected_cell_matrix(self, linear_index: int) -> NDArray[np.generic]:
        cell = self._cell_matrix(linear_index)
        cols = self._field_indices()
        if cols.size == self._full_num_fields:
            return cell
        if cols.size == 1:
            col = int(cols[0])
            return cell[:, col : col + 1]
        if _is_contiguous(cols):
            return cell[:, int(cols[0]) : int(cols[-1]) + 1]
        return cell[:, cols].copy()

    def _replace_cells(self, targets: NDArray[np.int64], arrays: Sequence[NDArray[np.generic]]) -> None:
        if len(targets) != len(arrays):
            raise ValueError("Target cell count does not match source cell count.")
        if len(targets) == 0:
            return

        normalized = [_coerce_cell_array(array, self._full_num_fields) for array in arrays]
        payloads = [array for array in normalized if array.shape[0] > 0]
        if payloads:
            appended = np.vstack(payloads)
            data = self._state["data"]
            if data.shape[0] == 0:
                self._state["data"] = appended.copy()
            else:
                self._state["data"] = np.concatenate((data, appended), axis=0)

        cursor = self._state["data"].shape[0] - sum(array.shape[0] for array in normalized)
        for target, array in zip(targets, normalized):
            self._state["cell_starts"][target] = cursor
            self._state["cell_lengths"][target] = array.shape[0]
            cursor += array.shape[0]

        self._maybe_compact_storage()

    def _expand_storage(self, num_new_fields: int) -> None:
        data = self._state["data"]
        if data.shape[0] == 0:
            dtype = np.result_type(data.dtype, float)
            self._state["data"] = np.empty((0, data.shape[1] + num_new_fields), dtype=dtype)
            return

        dtype = np.result_type(data.dtype, float)
        filler = np.full((data.shape[0], num_new_fields), np.nan, dtype=dtype)
        self._state["data"] = np.concatenate((data.astype(dtype, copy=False), filler), axis=1)

    def _maybe_compact_storage(self) -> None:
        data = self._state["data"]
        used_rows = int(self._state["cell_lengths"].sum())
        if data.shape[0] <= used_rows + 1024:
            return
        if used_rows == 0:
            self._state["data"] = np.empty((0, self._full_num_fields), dtype=data.dtype)
            self._state["cell_starts"].fill(0)
            return
        if data.shape[0] <= 2 * used_rows:
            return

        compacted = np.empty((used_rows, self._full_num_fields), dtype=data.dtype)
        starts = np.zeros_like(self._state["cell_starts"])
        cursor = 0
        for linear_index in range(_cell_count(self._state["shape"])):
            length = self._cell_row_count(linear_index)
            starts[linear_index] = cursor
            if length > 0:
                cell = self._cell_matrix(linear_index)
                compacted[cursor : cursor + length] = cell
                cursor += length
        self._state["data"] = compacted
        self._state["cell_starts"] = starts

    def _assign(self, value: Any) -> None:
        if self._is_full_field_selection():
            self._assign_full_cells(value)
        else:
            self._assign_selected_fields(value)

    def _assign_full_cells(self, value: Any) -> None:
        targets = self._selected_cell_indices()
        if isinstance(value, Vector):
            source_cells = value._selected_cell_indices()
            if len(targets) != len(source_cells):
                raise ValueError(f"Expected {len(targets)} cells, got {len(source_cells)}")
            if value.num_fields != self.num_fields:
                raise ValueError(
                    f"Expected {self.num_fields} fields, got {value.num_fields}"
                )
            arrays = [value._selected_cell_matrix(index).copy() for index in source_cells]
            self._replace_cells(targets, arrays)
            return

        array = _coerce_cell_array(value, self.num_fields)
        self._replace_cells(targets, [array] * len(targets))

    def _assign_selected_fields(self, value: Any) -> None:
        targets = self._selected_cell_indices()
        field_indices = self._field_indices()
        row_counts = [self._cell_row_count(index) for index in targets]
        total_rows = sum(row_counts)

        if isinstance(value, Vector):
            source_cells = value._selected_cell_indices()
            if len(targets) != len(source_cells):
                raise ValueError(f"Expected {len(targets)} cells, got {len(source_cells)}")
            if value.num_fields != self.num_fields:
                raise ValueError(
                    f"Expected {self.num_fields} fields, got {value.num_fields}"
                )
            source_counts = [value._cell_row_count(index) for index in source_cells]
            if row_counts != source_counts:
                raise ValueError("Per-cell row counts must match for field-selected assignment.")
            snapshots = [value._selected_cell_matrix(index).copy() for index in source_cells]
            for target, array in zip(targets, snapshots):
                cell = self._cell_matrix(int(target))
                if array.shape[0] > 0:
                    cell[:, field_indices] = array
            return

        if np.isscalar(value):
            for target in targets:
                cell = self._cell_matrix(int(target))
                if cell.shape[0] > 0:
                    cell[:, field_indices] = value
            return

        broadcast = _broadcast_field_values(value, total_rows, self.num_fields)
        cursor = 0
        for target, rows in zip(targets, row_counts):
            chunk = broadcast[cursor : cursor + rows]
            cell = self._cell_matrix(int(target))
            if rows > 0:
                cell[:, field_indices] = chunk
            cursor += rows

    def _binary_op(self, other: Any, op: Any, reverse: bool = False) -> "Vector":
        result = self.copy()
        result._inplace_op(other, op, reverse=reverse)
        return result

    def _inplace_op(self, other: Any, op: Any, reverse: bool = False) -> None:
        targets = self._selected_cell_indices()
        field_indices = self._field_indices()
        row_counts = [self._cell_row_count(index) for index in targets]
        total_rows = sum(row_counts)

        if isinstance(other, Vector):
            source_cells = other._selected_cell_indices()
            if len(targets) != len(source_cells):
                raise ValueError(f"Expected {len(targets)} cells, got {len(source_cells)}")
            if other.num_fields != self.num_fields:
                raise ValueError(f"Expected {self.num_fields} fields, got {other.num_fields}")
            source_counts = [other._cell_row_count(index) for index in source_cells]
            if row_counts != source_counts:
                raise ValueError("Per-cell row counts must match for Vector arithmetic.")
            snapshots = [other._selected_cell_matrix(index).copy() for index in source_cells]
            for target, rhs in zip(targets, snapshots):
                cell = self._cell_matrix(int(target))
                lhs = cell[:, field_indices]
                cell[:, field_indices] = op(rhs, lhs) if reverse else op(lhs, rhs)
            return

        if np.isscalar(other):
            for target in targets:
                cell = self._cell_matrix(int(target))
                lhs = cell[:, field_indices]
                if lhs.shape[0] > 0:
                    cell[:, field_indices] = op(other, lhs) if reverse else op(lhs, other)
            return

        broadcast = _broadcast_field_values(other, total_rows, self.num_fields)
        cursor = 0
        for target, rows in zip(targets, row_counts):
            chunk = broadcast[cursor : cursor + rows]
            cell = self._cell_matrix(int(target))
            lhs = cell[:, field_indices]
            if rows > 0:
                cell[:, field_indices] = op(chunk, lhs) if reverse else op(lhs, chunk)
            cursor += rows


def _cell_count(shape: tuple[int, ...]) -> int:
    return int(np.prod(shape, dtype=np.int64)) if shape else 1


def _normalize_field_names(field_names: str | Sequence[str]) -> tuple[str, ...]:
    if isinstance(field_names, str):
        normalized = (field_names,)
    else:
        normalized = tuple(field_names)
    if not normalized:
        raise ValueError("At least one field name is required.")
    validate_fields(list(normalized))
    return normalized



def _normalize_units(units: str | Sequence[str] | None, count: int) -> list[str]:
    if units is None:
        return ["none"] * count
    if isinstance(units, str):
        if count != 1:
            raise ValueError("A single unit can only be provided for a single field.")
        return [units]
    normalized = list(units)
    if len(normalized) != count:
        raise ValueError(f"Expected {count} units, got {len(normalized)}")
    return normalized



def _looks_like_field_selector(idx: Any) -> bool:
    if isinstance(idx, str):
        return True
    if isinstance(idx, tuple) and any(_looks_like_field_selector(item) for item in idx):
        return True
    if isinstance(idx, (list, tuple)) and idx and all(isinstance(item, str) for item in idx):
        return True
    return False



def _coerce_cell_array(value: Any, num_fields: int) -> NDArray[np.generic]:
    if isinstance(value, Vector):
        if value.shape != ():
            raise ValueError("Expected a 0D Vector for single-cell assignment.")
        array = value.array.copy()
    else:
        array = np.asarray(value)

    if array.ndim == 0:
        raise ValueError("Cell assignment requires a 2D array.")
    if array.ndim == 1:
        if array.size == 0:
            array = np.empty((0, num_fields), dtype=float)
        elif num_fields == 1:
            array = array.reshape(-1, 1)
        else:
            array = array.reshape(1, -1)
    if array.ndim != 2:
        raise ValueError("Cell assignment requires a 2D array.")
    if array.shape[1] != num_fields:
        raise ValueError(f"Expected {num_fields} fields, got {array.shape[1]}")
    return array



def _normalize_nested_data(data: list[Any]) -> tuple[tuple[int, ...], list[NDArray[np.generic]]]:
    if not isinstance(data, list):
        raise TypeError("Data must be a list")
    if not data:
        return (0,), []
    return _flatten_fixed_grid(data)



def _flatten_fixed_grid(node: Any) -> tuple[tuple[int, ...], list[NDArray[np.generic]]]:
    if isinstance(node, np.ndarray):
        return (), [_coerce_inferred_cell_array(node)]
    if not isinstance(node, (list, tuple)):
        raise TypeError("Data must be a nested list/tuple of cell arrays or row sequences.")
    if _looks_like_cell_rows(node):
        return (), [_coerce_inferred_cell_array(node)]
    if len(node) == 0:
        return (0,), []

    child_shape: tuple[int, ...] | None = None
    cells: list[NDArray[np.generic]] = []
    for child in node:
        shape, child_cells = _flatten_fixed_grid(child)
        if child_shape is None:
            child_shape = shape
        elif child_shape != shape:
            raise ValueError("All nested fixed-grid branches must have matching shapes.")
        cells.extend(child_cells)

    assert child_shape is not None
    return (len(node),) + child_shape, cells



def _looks_like_cell_rows(node: Sequence[Any]) -> bool:
    if len(node) == 0:
        return True
    return all(_is_row_like(item) for item in node)



def _is_row_like(item: Any) -> bool:
    if isinstance(item, np.ndarray):
        return item.ndim == 1
    if not isinstance(item, (list, tuple)):
        return False
    return all(np.isscalar(value) for value in item)



def _coerce_inferred_cell_array(value: Any) -> NDArray[np.generic]:
    array = np.asarray(value)
    if array.ndim == 0:
        raise ValueError("Cell data must be 1D or 2D.")
    if array.ndim == 1:
        if array.size == 0:
            return np.empty((0, 0), dtype=float)
        return array.reshape(1, -1)
    if array.ndim != 2:
        raise ValueError("Cell data must be 1D or 2D.")
    return array



def _select_linear_indices(
    shape: tuple[int, ...],
    current_indices: NDArray[np.int64],
    idx: Any,
) -> tuple[tuple[int, ...], NDArray[np.int64]]:
    if shape == ():
        if idx in ((), Ellipsis):
            return (), np.array([int(current_indices[0])], dtype=np.int64)
        raise IndexError("Too many indices for 0D Vector")

    index_tuple = _normalize_index_tuple(idx, len(shape))
    current_grid = current_indices.reshape(shape)

    axis_positions: list[NDArray[np.int64]] = []
    out_shape: list[int] = []
    scalar_axes: list[bool] = []
    for axis, axis_index in enumerate(index_tuple):
        positions, is_scalar = _positions_for_axis(axis_index, shape[axis])
        axis_positions.append(positions)
        scalar_axes.append(is_scalar)
        if not is_scalar:
            out_shape.append(len(positions))

    if all(scalar_axes):
        scalar_key = tuple(int(positions[0]) for positions in axis_positions)
        value = int(current_grid[scalar_key])
        return (), np.array([value], dtype=np.int64)

    mesh_inputs = [positions if not is_scalar else positions[:1] for positions, is_scalar in zip(axis_positions, scalar_axes)]
    grids = np.meshgrid(*mesh_inputs, indexing="ij")
    selected = np.asarray(current_grid[tuple(grids)], dtype=np.int64).reshape(-1)
    return tuple(out_shape), selected



def _normalize_index_tuple(idx: Any, ndim: int) -> tuple[Any, ...]:
    if idx is Ellipsis:
        return (slice(None),) * ndim
    if not isinstance(idx, tuple):
        idx = (idx,)

    ellipsis_count = sum(item is Ellipsis for item in idx)
    if ellipsis_count > 1:
        raise IndexError("An index can only have a single ellipsis.")
    if ellipsis_count == 1:
        ellipsis_pos = idx.index(Ellipsis)
        fill = ndim - (len(idx) - 1)
        idx = idx[:ellipsis_pos] + (slice(None),) * fill + idx[ellipsis_pos + 1 :]
    if len(idx) > ndim:
        raise IndexError(f"Too many indices for Vector: expected {ndim}, got {len(idx)}")
    if len(idx) < ndim:
        idx = idx + (slice(None),) * (ndim - len(idx))
    return idx



def _positions_for_axis(axis_index: Any, size: int) -> tuple[NDArray[np.int64], bool]:
    if isinstance(axis_index, (bool, np.bool_)):
        raise TypeError("Boolean scalars are not valid Vector indices.")

    if isinstance(axis_index, (int, np.integer)):
        index = int(axis_index)
        if index < 0:
            index += size
        if index < 0 or index >= size:
            raise IndexError("Vector index out of range")
        return np.array([index], dtype=np.int64), True

    if isinstance(axis_index, slice):
        return np.arange(size, dtype=np.int64)[axis_index], False

    array = np.asarray(axis_index)
    if array.ndim == 0:
        if np.issubdtype(array.dtype, np.integer):
            return _positions_for_axis(int(array.item()), size)
        raise TypeError(f"Unsupported index type: {type(axis_index)!r}")

    if array.dtype == bool or np.issubdtype(array.dtype, np.bool_):
        if array.ndim != 1:
            raise IndexError("Full-grid boolean masks are not supported.")
        if array.shape[0] != size:
            raise IndexError(f"Boolean mask length {array.shape[0]} does not match axis length {size}")
        return np.flatnonzero(array).astype(np.int64, copy=False), False

    if array.ndim != 1:
        raise IndexError("Fancy indexing arrays must be one-dimensional.")
    if array.size == 0:
        return np.array([], dtype=np.int64), False
    if not np.issubdtype(array.dtype, np.integer):
        raise TypeError("Fancy indices must be integers or booleans.")

    positions = array.astype(np.int64, copy=True)
    positions[positions < 0] += size
    if np.any((positions < 0) | (positions >= size)):
        raise IndexError("Vector index out of range")
    return positions, False



def _broadcast_field_values(value: Any, total_rows: int, num_fields: int) -> NDArray[np.generic]:
    array = np.asarray(value)
    if array.ndim == 0:
        return np.broadcast_to(array.reshape(1, 1), (total_rows, num_fields))
    if num_fields == 1 and array.ndim == 1:
        if total_rows == 0 and array.shape[0] == 0:
            return array.reshape(0, 1)
        if array.shape[0] != total_rows:
            raise ValueError(f"Expected {total_rows} values, got {array.shape[0]}")
        return array.reshape(total_rows, 1)
    try:
        return np.broadcast_to(array, (total_rows, num_fields))
    except ValueError as exc:
        raise ValueError(
            f"Cannot broadcast value with shape {array.shape} to ({total_rows}, {num_fields})"
        ) from exc



def _is_contiguous(indices: NDArray[np.int64]) -> bool:
    if indices.size <= 1:
        return True
    return bool(np.all(indices[1:] - indices[:-1] == 1))
