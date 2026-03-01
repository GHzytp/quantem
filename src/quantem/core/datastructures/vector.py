from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Iterable, Sequence, overload

import numpy as np
from numpy.typing import NDArray

from quantem.core.io.serialize import AutoSerialize
from quantem.core.utils.validators import (
    validate_fields,
    validate_num_fields,
    validate_shape,
    validate_vector_units,
)


@dataclass
class _VectorState:
    shape: tuple[int, ...]
    data: NDArray[np.object_]
    fields: list[str]
    units: list[str]
    name: str
    metadata: dict[str, Any]


class Vector(AutoSerialize):
    """Ragged cell data on a fixed grid.

    A ``Vector`` has fixed grid dimensions (``shape``). Each fixed-grid cell stores a
    NumPy array with shape ``(n_rows, num_fields)``. Selections always return new
    ``Vector`` views over the same backing store, while raw NumPy extraction is explicit
    via ``.array`` and ``.flatten()``.
    """

    def __init__(
        self,
        shape: tuple[int, ...],
        fields: Sequence[str],
        units: Sequence[str] | None = None,
        name: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        validated_shape = validate_shape(shape)
        validated_fields = validate_fields(list(fields))
        validated_units = validate_vector_units(list(units) if units is not None else None, len(validated_fields))
        self._state = _VectorState(
            shape=validated_shape,
            data=_empty_storage(validated_shape, len(validated_fields)),
            fields=list(validated_fields),
            units=list(validated_units),
            name=name or f"{len(validated_shape)}d ragged array",
            metadata=dict(metadata or {}),
        )
        self._coords = _root_coords(validated_shape)
        self._field_names = tuple(validated_fields)

    @classmethod
    def _from_state(
        cls,
        state: _VectorState,
        coords: NDArray[np.object_],
        field_names: Sequence[str],
    ) -> "Vector":
        obj = cls.__new__(cls)
        obj._state = state
        obj._coords = coords
        obj._field_names = tuple(field_names)
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
            validated_fields = validate_fields(list(fields))
            if num_fields is not None and len(validated_fields) != num_fields:
                raise ValueError(
                    f"num_fields ({num_fields}) does not match length of fields ({len(validated_fields)})"
                )
        elif num_fields is not None:
            validated_count = validate_num_fields(num_fields)
            validated_fields = [f"field_{i}" for i in range(validated_count)]
        else:
            raise ValueError("Must specify either 'fields' or 'num_fields'.")

        return cls(
            shape=shape,
            fields=validated_fields,
            units=units,
            name=name,
        )

    @classmethod
    def from_data(
        cls,
        data: list[Any],
        num_fields: int | None = None,
        fields: Sequence[str] | None = None,
        units: Sequence[str] | None = None,
        name: str | None = None,
    ) -> "Vector":
        inferred_shape, normalized_cells = _normalize_nested_data(data)
        inferred_num_fields = normalized_cells[0].shape[1] if normalized_cells else 0

        if fields is not None:
            validated_fields = validate_fields(list(fields))
            if len(validated_fields) != inferred_num_fields:
                raise ValueError(
                    f"num_fields ({inferred_num_fields}) does not match length of fields ({len(validated_fields)})"
                )
        elif num_fields is not None:
            validated_num_fields = validate_num_fields(num_fields)
            if validated_num_fields != inferred_num_fields:
                raise ValueError(
                    f"Provided num_fields ({validated_num_fields}) does not match inferred ({inferred_num_fields})."
                )
            validated_fields = [f"field_{i}" for i in range(validated_num_fields)]
        else:
            validated_fields = [f"field_{i}" for i in range(inferred_num_fields)]

        vector = cls(
            shape=inferred_shape,
            fields=validated_fields,
            units=units,
            name=name,
        )
        for coord, array in zip(np.ndindex(inferred_shape), normalized_cells):
            vector._state.data[coord] = array.copy()
        return vector

    @property
    def shape(self) -> tuple[int, ...]:
        return self._coords.shape

    @property
    def fields(self) -> list[str]:
        return list(self._field_names)

    @property
    def units(self) -> list[str]:
        lookup = {name: unit for name, unit in zip(self._state.fields, self._state.units)}
        return [lookup[name] for name in self._field_names]

    @property
    def num_fields(self) -> int:
        return len(self._field_names)

    @property
    def name(self) -> str:
        return self._state.name

    @name.setter
    def name(self, value: str) -> None:
        self._state.name = str(value)

    @property
    def metadata(self) -> dict[str, Any]:
        return self._state.metadata

    @property
    def array(self) -> NDArray[np.generic]:
        if self.shape != ():
            raise ValueError(".array is only valid when the selection contains exactly one cell.")
        coord = self._coords[()]
        cell = self._state.data[coord]
        indices = self._field_indices()
        if len(indices) == len(self._state.fields) and indices == list(range(len(self._state.fields))):
            return cell
        if len(indices) == 1:
            idx = indices[0]
            return cell[:, idx : idx + 1]
        if _is_contiguous(indices):
            return cell[:, indices[0] : indices[-1] + 1]
        return cell[:, indices].copy()

    def __len__(self) -> int:
        if self.shape == ():
            raise TypeError("len() of unsized 0D Vector")
        return self.shape[0]

    def __repr__(self) -> str:
        return f"quantem.Vector(shape={self.shape}, fields={self.fields}, name={self.name!r})"

    __str__ = __repr__

    def copy(self) -> "Vector":
        copied = self.__class__._from_state(
            _VectorState(
                shape=self.shape,
                data=_empty_storage(self.shape, self.num_fields),
                fields=self.fields,
                units=self.units,
                name=self.name,
                metadata=copy.deepcopy(self.metadata),
            ),
            _root_coords_allow_empty(self.shape),
            self.fields,
        )
        for coord, array in zip(np.ndindex(self.shape) if self.shape != () else [()], self._iter_selected_arrays()):
            copied._state.data[coord] = array.copy()
        return copied

    def flatten(self) -> NDArray[np.generic]:
        arrays = [array for array in self._iter_selected_arrays() if array.shape[0] > 0]
        if arrays:
            return np.vstack(arrays)
        dtype = float
        for array in self._iter_selected_arrays():
            dtype = array.dtype
            break
        return np.empty((0, self.num_fields), dtype=dtype)

    def select_fields(self, field_names: str | Sequence[str]) -> "Vector":
        normalized = _normalize_field_names(field_names)
        available = set(self._field_names)
        missing = [name for name in normalized if name not in available]
        if missing:
            raise KeyError(f"Unknown field(s): {missing}")
        return self._from_state(self._state, self._coords, normalized)

    def add_fields(
        self,
        names: str | Sequence[str],
        values: Any | None = None,
        units: str | Sequence[str] | None = None,
    ) -> None:
        new_names = _normalize_field_names(names)
        if any(name in self._state.fields for name in new_names):
            raise ValueError("One or more new field names already exist.")

        new_units = _normalize_units(units, len(new_names))
        old_fields = list(self._state.fields)
        self._state.fields.extend(new_names)
        self._state.units.extend(new_units)

        old_width = len(old_fields)
        new_width = len(self._state.fields)
        for coord in np.ndindex(self._state.shape) if self._state.shape != () else [()]:
            current = self._state.data[coord]
            promoted = np.result_type(current.dtype, float)
            expanded = np.full((current.shape[0], new_width), np.nan, dtype=promoted)
            expanded[:, :old_width] = current
            self._state.data[coord] = expanded

        if list(self._field_names) == old_fields:
            self._field_names = tuple(self._state.fields)

        target = self._from_state(self._state, self._coords, new_names)
        if values is None:
            return

        if len(new_names) > 1 and isinstance(values, (list, tuple)) and len(values) == len(new_names):
            for name, value in zip(new_names, values):
                target.select_fields(name)._assign(value)
        else:
            target._assign(values)

    def remove_fields(self, names: str | Sequence[str]) -> None:
        to_remove = _normalize_field_names(names)
        missing = [name for name in to_remove if name not in self._state.fields]
        if missing:
            raise KeyError(f"Unknown field(s): {missing}")
        if len(to_remove) == len(self._state.fields):
            raise ValueError("Cannot remove all fields from a Vector.")

        remove_set = set(to_remove)
        keep_names = [name for name in self._state.fields if name not in remove_set]
        keep_indices = [self._state.fields.index(name) for name in keep_names]
        keep_units = [self._state.units[index] for index in keep_indices]

        for coord in np.ndindex(self._state.shape) if self._state.shape != () else [()]:
            self._state.data[coord] = self._state.data[coord][:, keep_indices]

        self._state.fields = keep_names
        self._state.units = keep_units
        self._field_names = tuple(name for name in self._field_names if name in keep_names)

    def get_data(self, *indices: Any) -> NDArray[np.generic] | list[NDArray[np.generic]]:
        if len(indices) != len(self.shape):
            raise ValueError(f"Expected {len(self.shape)} indices, got {len(indices)}")
        selection = self[indices if len(indices) != 1 else indices[0]]
        if selection.shape == ():
            return selection.array
        return [array.copy() for array in selection._iter_selected_arrays()]

    def set_data(self, value: Any, *indices: Any) -> None:
        if len(indices) != len(self.shape):
            raise ValueError(f"Expected {len(self.shape)} indices, got {len(indices)}")
        self[indices if len(indices) != 1 else indices[0]] = value

    @overload
    def __getitem__(self, idx: Any) -> "Vector": ...

    def __getitem__(self, idx: Any) -> "Vector":
        if _is_field_selector(idx):
            raise TypeError("Use select_fields(...) for field selection.")
        coords = _select_coords(self._coords, idx)
        return self._from_state(self._state, coords, self._field_names)

    def __setitem__(self, idx: Any, value: Any) -> None:
        if _is_field_selector(idx):
            raise TypeError("Use select_fields(...) for field selection.")
        self[idx]._assign(value)

    def __add__(self, other: Any) -> "Vector":
        return self._binary_op(other, np.add)

    def __sub__(self, other: Any) -> "Vector":
        return self._binary_op(other, np.subtract)

    def __mul__(self, other: Any) -> "Vector":
        return self._binary_op(other, np.multiply)

    def __truediv__(self, other: Any) -> "Vector":
        return self._binary_op(other, np.divide)

    def __pow__(self, other: Any) -> "Vector":
        return self._binary_op(other, np.power)

    def __iadd__(self, other: Any) -> "Vector":
        return self._binary_op_inplace(other, np.add)

    def __isub__(self, other: Any) -> "Vector":
        return self._binary_op_inplace(other, np.subtract)

    def __imul__(self, other: Any) -> "Vector":
        return self._binary_op_inplace(other, np.multiply)

    def __itruediv__(self, other: Any) -> "Vector":
        return self._binary_op_inplace(other, np.divide)

    def __ipow__(self, other: Any) -> "Vector":
        return self._binary_op_inplace(other, np.power)

    def _binary_op(self, other: Any, op: Any) -> "Vector":
        result = self.copy()
        result._binary_op_inplace(other, op)
        return result

    def _binary_op_inplace(self, other: Any, op: Any) -> "Vector":
        row_counts = self._row_counts()
        targets = list(self._iter_coords())

        if isinstance(other, Vector):
            self._validate_vector_rhs(other, require_matching_rows=True)
            source_arrays = list(other._iter_selected_arrays())
            for coord, current, rhs in zip(targets, self._iter_selected_arrays(), source_arrays):
                updated = current.copy()
                op(updated, rhs, out=updated, casting="same_kind")
                self._write_selected_array(coord, updated)
            return self

        rhs_matrix = _broadcast_rhs(np.asarray(other), sum(row_counts), self.num_fields)
        cursor = 0
        for coord, current, row_count in zip(targets, self._iter_selected_arrays(), row_counts):
            chunk = rhs_matrix[cursor : cursor + row_count]
            cursor += row_count
            updated = current.copy()
            op(updated, chunk, out=updated, casting="same_kind")
            self._write_selected_array(coord, updated)
        return self

    def _assign(self, value: Any) -> None:
        if self._is_full_field_selection():
            self._assign_full_cells(value)
        else:
            self._assign_selected_fields(value)

    def _assign_full_cells(self, value: Any) -> None:
        targets = list(self._iter_coords())
        if isinstance(value, Vector):
            self._validate_vector_rhs(value, require_matching_rows=False)
            for target, source in zip(targets, value._iter_selected_arrays()):
                self._state.data[target] = source.copy()
            return

        array = _coerce_cell_array(value, self.num_fields)
        for target in targets:
            self._state.data[target] = array.copy()

    def _assign_selected_fields(self, value: Any) -> None:
        targets = list(self._iter_coords())
        row_counts = self._row_counts()
        if isinstance(value, Vector):
            self._validate_vector_rhs(value, require_matching_rows=True)
            for coord, source in zip(targets, value._iter_selected_arrays()):
                self._write_selected_array(coord, source)
            return

        if np.isscalar(value):
            for coord in targets:
                current = self._selected_array_for_coord(coord)
                fill = np.full(current.shape, value)
                self._write_selected_array(coord, fill)
            return

        rhs_matrix = _broadcast_rhs(np.asarray(value), sum(row_counts), self.num_fields)
        cursor = 0
        for coord, row_count in zip(targets, row_counts):
            chunk = rhs_matrix[cursor : cursor + row_count]
            cursor += row_count
            self._write_selected_array(coord, chunk)

    def _validate_vector_rhs(self, other: "Vector", require_matching_rows: bool) -> None:
        if self.num_fields != other.num_fields:
            raise ValueError(f"Expected {self.num_fields} fields, got {other.num_fields}")
        if self._coords.size != other._coords.size:
            raise ValueError(f"Expected {self._coords.size} cells, got {other._coords.size}")
        if require_matching_rows:
            left = self._row_counts()
            right = other._row_counts()
            if left != right:
                raise ValueError(f"Per-cell row counts must match: {left} != {right}")

    def _row_counts(self) -> list[int]:
        return [self._state.data[coord].shape[0] for coord in self._iter_coords()]

    def _iter_coords(self) -> Iterable[tuple[int, ...]]:
        return iter(self._coords.flat)

    def _iter_selected_arrays(self) -> Iterable[NDArray[np.generic]]:
        for coord in self._iter_coords():
            yield self._selected_array_for_coord(coord)

    def _selected_array_for_coord(self, coord: tuple[int, ...]) -> NDArray[np.generic]:
        cell = self._state.data[coord]
        indices = self._field_indices()
        if self._is_full_field_selection():
            return cell
        if len(indices) == 1:
            idx = indices[0]
            return cell[:, idx : idx + 1]
        if _is_contiguous(indices):
            return cell[:, indices[0] : indices[-1] + 1]
        return cell[:, indices].copy()

    def _write_selected_array(self, coord: tuple[int, ...], values: NDArray[np.generic]) -> None:
        cell = self._state.data[coord]
        if self._is_full_field_selection():
            replacement = _coerce_cell_array(values, self.num_fields)
            self._state.data[coord] = replacement.copy()
            return

        if values.ndim != 2 or values.shape[1] != self.num_fields:
            raise ValueError(
                f"Expected array with shape (_, {self.num_fields}), got {values.shape}"
            )
        if values.shape[0] != cell.shape[0]:
            raise ValueError(
                f"Expected {cell.shape[0]} rows for in-place field update, got {values.shape[0]}"
            )
        cell[:, self._field_indices()] = values

    def _field_indices(self) -> list[int]:
        lookup = {name: idx for idx, name in enumerate(self._state.fields)}
        try:
            return [lookup[name] for name in self._field_names]
        except KeyError as exc:
            raise KeyError(f"Unknown field '{exc.args[0]}'") from exc

    def _is_full_field_selection(self) -> bool:
        return list(self._field_names) == self._state.fields


def _empty_storage(shape: tuple[int, ...], num_fields: int) -> NDArray[np.object_]:
    storage = np.empty(shape if shape != () else (), dtype=object)
    for coord in np.ndindex(shape) if shape != () else [()]:
        storage[coord] = np.empty((0, num_fields), dtype=float)
    return storage


def _root_coords(shape: tuple[int, ...]) -> NDArray[np.object_]:
    return _root_coords_allow_empty(shape)


def _root_coords_allow_empty(shape: tuple[int, ...]) -> NDArray[np.object_]:
    coords = np.empty(shape if shape != () else (), dtype=object)
    for coord in np.ndindex(shape) if shape != () else [()]:
        coords[coord] = coord
    return coords


def _normalize_field_names(field_names: str | Sequence[str]) -> list[str]:
    if isinstance(field_names, str):
        names = [field_names]
    elif isinstance(field_names, Sequence):
        names = [str(name) for name in field_names]
    else:
        raise TypeError("Field names must be a string or a sequence of strings.")
    if not names:
        raise ValueError("Must select at least one field.")
    if len(set(names)) != len(names):
        raise ValueError("Duplicate field names are not allowed.")
    return names


def _normalize_units(units: str | Sequence[str] | None, count: int) -> list[str]:
    if units is None:
        return ["none"] * count
    if isinstance(units, str):
        if count != 1:
            raise ValueError("A single unit string is only valid for a single new field.")
        return [units]
    normalized = [str(unit) for unit in units]
    if len(normalized) != count:
        raise ValueError(f"Expected {count} units, got {len(normalized)}")
    return normalized


def _coerce_cell_array(value: Any, num_fields: int) -> NDArray[np.generic]:
    array = np.asarray(value)
    if array.ndim != 2 or array.shape[1] != num_fields:
        raise ValueError(f"Expected a numpy array with shape (_, {num_fields}), got {array.shape}")
    if not np.issubdtype(array.dtype, np.number):
        raise TypeError("Cell arrays must contain numeric values.")
    return array


def _broadcast_rhs(value: NDArray[np.generic], total_rows: int, num_fields: int) -> NDArray[np.generic]:
    if num_fields == 1 and value.ndim == 1:
        value = value.reshape(-1, 1)
    try:
        return np.broadcast_to(value, (total_rows, num_fields))
    except ValueError as exc:
        raise ValueError(
            f"RHS is not broadcast-compatible with flattened target shape ({total_rows}, {num_fields})."
        ) from exc


def _is_field_selector(idx: Any) -> bool:
    if isinstance(idx, str):
        return True
    if isinstance(idx, list | tuple) and idx:
        if all(isinstance(item, str) for item in idx):
            return True
        return any(isinstance(item, str) for item in idx)
    return False


def _is_contiguous(indices: Sequence[int]) -> bool:
    if not indices:
        return False
    return list(indices) == list(range(indices[0], indices[0] + len(indices)))


def _select_coords(coords: NDArray[np.object_], idx: Any) -> NDArray[np.object_]:
    shape = coords.shape
    selectors, scalar_axes = _normalize_indices(shape, idx)
    out_shape = tuple(len(selector) for selector, scalar in zip(selectors, scalar_axes) if not scalar)
    result = np.empty(out_shape if out_shape != () else (), dtype=object)

    if out_shape == ():
        source = tuple(selector[0] for selector in selectors)
        result[()] = coords[source]
        return result

    for out_index in np.ndindex(out_shape):
        out_cursor = 0
        source = []
        for axis, selector in enumerate(selectors):
            if scalar_axes[axis]:
                source.append(selector[0])
            else:
                source.append(selector[out_index[out_cursor]])
                out_cursor += 1
        result[out_index] = coords[tuple(source)]
    return result


def _normalize_indices(shape: tuple[int, ...], idx: Any) -> tuple[list[np.ndarray], list[bool]]:
    if shape == ():
        if idx in ((), Ellipsis, slice(None), None):
            return [], []
        raise IndexError("Too many indices for a 0D Vector.")

    normalized = idx if isinstance(idx, tuple) else (idx,)
    normalized = _expand_ellipsis(normalized, len(shape))
    if len(normalized) > len(shape):
        raise IndexError(f"Expected at most {len(shape)} indices, got {len(normalized)}")
    normalized = normalized + (slice(None),) * (len(shape) - len(normalized))

    selectors: list[np.ndarray] = []
    scalar_axes: list[bool] = []
    for axis_value, axis_size in zip(normalized, shape):
        selector, scalar = _normalize_axis_index(axis_value, axis_size)
        selectors.append(selector)
        scalar_axes.append(scalar)
    return selectors, scalar_axes


def _expand_ellipsis(idx: tuple[Any, ...], ndim: int) -> tuple[Any, ...]:
    if not any(item is Ellipsis for item in idx):
        return idx
    if sum(item is Ellipsis for item in idx) > 1:
        raise IndexError("Only one ellipsis is allowed.")
    ellipsis_pos = next(i for i, item in enumerate(idx) if item is Ellipsis)
    fill = ndim - (len(idx) - 1)
    return idx[:ellipsis_pos] + (slice(None),) * fill + idx[ellipsis_pos + 1 :]


def _normalize_axis_index(value: Any, axis_size: int) -> tuple[np.ndarray, bool]:
    if isinstance(value, (int, np.integer)):
        index = int(value)
        if index < 0:
            index += axis_size
        if index < 0 or index >= axis_size:
            raise IndexError(f"Index {value} out of bounds for axis with size {axis_size}")
        return np.array([index], dtype=int), True

    if isinstance(value, slice):
        return np.arange(axis_size)[value], False

    if isinstance(value, np.ndarray) and value.ndim == 0:
        return _normalize_axis_index(value.item(), axis_size)

    array = np.asarray(value)
    if array.ndim != 1:
        raise IndexError("Only 1D per-axis fancy or boolean indexing is supported.")
    if array.size == 0:
        return np.array([], dtype=int), False

    if array.dtype == bool or np.issubdtype(array.dtype, np.bool_):
        if array.shape[0] != axis_size:
            raise IndexError(
                f"Boolean mask length {array.shape[0]} does not match axis size {axis_size}"
            )
        return np.flatnonzero(array), False

    if not np.issubdtype(array.dtype, np.integer):
        raise TypeError(f"Unsupported index type {type(value).__name__}")

    normalized = array.astype(int, copy=True)
    normalized[normalized < 0] += axis_size
    if np.any((normalized < 0) | (normalized >= axis_size)):
        raise IndexError(f"Index out of bounds for axis with size {axis_size}")
    return normalized, False


def _normalize_nested_data(data: list[Any]) -> tuple[tuple[int, ...], list[NDArray[np.generic]]]:
    if not isinstance(data, list):
        raise TypeError("Data must be a list.")
    if not data:
        raise ValueError("Data list cannot be empty.")

    leaves: list[NDArray[np.generic]] = []
    num_fields: int | None = None

    def walk(node: Any) -> tuple[int, ...]:
        nonlocal num_fields
        if _is_leaf_cell(node):
            array = np.asarray(node)
            if array.ndim != 2:
                raise ValueError(f"Cell arrays must be 2D, got shape {array.shape}")
            if not np.issubdtype(array.dtype, np.number):
                raise TypeError("Cell arrays must contain numeric values.")
            if num_fields is None:
                num_fields = array.shape[1]
            elif array.shape[1] != num_fields:
                raise ValueError("All data arrays must have same number of fields.")
            leaves.append(array)
            return ()

        if not isinstance(node, list):
            raise TypeError("Data elements must be numpy arrays, numeric 2D lists, or nested lists thereof.")
        if not node:
            raise ValueError("Nested data lists cannot be empty.")
        child_shapes = [walk(child) for child in node]
        first = child_shapes[0]
        if any(shape != first for shape in child_shapes[1:]):
            raise ValueError("Nested data structure must have a consistent fixed-grid shape.")
        return (len(node), *first)

    inferred_shape = walk(data)
    return inferred_shape, leaves


def _is_leaf_cell(node: Any) -> bool:
    if isinstance(node, np.ndarray):
        return True
    if not isinstance(node, list):
        return False
    try:
        array = np.asarray(node)
    except Exception:
        return False
    return array.ndim == 2 and np.issubdtype(array.dtype, np.number)
