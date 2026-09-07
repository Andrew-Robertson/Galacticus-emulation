"""Utilities for writing Galacticus parameter-change files."""

from __future__ import annotations

from pathlib import Path
import xml.etree.ElementTree as ET


def build_changes_tree(
    parameter_path_value_pairs: list[tuple[str, object]],
    random_seed: int | None = None,
) -> ET.ElementTree:
    """Build a Galacticus parameter changes XML tree."""
    changes_root = ET.Element("changes")
    for parameter_path, parameter_value in parameter_path_value_pairs:
        ET.SubElement(
            changes_root,
            "change",
            type="update",
            path=str(parameter_path),
            value=str(parameter_value),
        )

    if random_seed is not None:
        rng_change = ET.SubElement(
            changes_root,
            "change",
            type="replaceOrAppend",
            path="randomNumberGenerator",
        )
        rng_element = ET.SubElement(rng_change, "randomNumberGenerator", value="GSL")
        ET.SubElement(rng_element, "seed", value=str(random_seed))

    tree = ET.ElementTree(changes_root)
    ET.indent(tree, space="  ")
    return tree


def write_changes_file(
    parameter_path_value_pairs: list[tuple[str, object]],
    output_path: str | Path,
    random_seed: int | None = None,
) -> Path:
    """Write a Galacticus parameter changes file to disk."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tree = build_changes_tree(
        parameter_path_value_pairs=parameter_path_value_pairs,
        random_seed=random_seed,
    )
    tree.write(output_path, encoding="UTF-8", xml_declaration=True)
    return output_path
