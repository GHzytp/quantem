import csv
from pathlib import Path
from typing import Optional, Union


def _parse_float(row: dict[str, str], keys: tuple[str, ...]) -> Optional[float]:
    for key in keys:
        value = row.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if not text:
            continue
        try:
            return float(text)
        except ValueError:
            continue
    return None


def load_xray_lines_database(path: Union[Path, str]) -> dict[str, dict[str, dict[str, float]]]:
    """Load X-ray lines CSV into the legacy element->line metadata mapping."""
    elements: dict[str, dict[str, dict[str, float]]] = {}
    duplicate_counts: dict[tuple[str, str], int] = {}

    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            element = str(row.get("element", "")).strip()
            line_name = str(row.get("line", "")).strip()
            if not element or not line_name:
                continue

            energy_kev = _parse_float(row, ("energy_keV", "energy (keV)", "energy"))
            if energy_kev is None:
                energy_ev = _parse_float(row, ("energy_eV", "energy (eV)"))
                if energy_ev is None:
                    continue
                energy_kev = energy_ev / 1000.0

            # Use the normalized CSV column as the X-ray line weight.
            weight = _parse_float(row, ("col4_norm", "weight", "relative_intensity"))
            if weight is None:
                weight = 0.0

            element_lines = elements.setdefault(element, {})
            key = (element, line_name)
            if line_name in element_lines:
                duplicate_counts[key] = duplicate_counts.get(key, 1) + 1
                line_name = f"{line_name}__{duplicate_counts[key]}"

            element_lines[line_name] = {
                "energy (keV)": float(energy_kev),
                "weight": float(weight),
            }

    return elements
