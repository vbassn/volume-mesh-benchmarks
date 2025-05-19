"""Utility to convert HDF5-based meshes into VTK ``.vtu`` files.

This script relies on :mod:`meshio` to load a mesh stored in a ``.h5`` file
and write it back as an unstructured VTK file. All point, cell and field data
supported by :mod:`meshio` is preserved.

Example
-------
    python h5_to_vtu.py input.h5 output.vtu
"""

from __future__ import annotations

import argparse
from typing import Optional

try:  # pragma: no cover - meshio may not be available
    import meshio
except ModuleNotFoundError as exc:  # pragma: no cover - meshio is optional
    raise ImportError("The meshio package is required to run this script.") from exc


def convert(h5_file: str, vtu_file: str, file_format: Optional[str] = None) -> None:
    """Convert ``h5_file`` into ``vtu_file`` using ``meshio``.

    Parameters
    ----------
    h5_file : str
        Path to the input ``.h5`` file.
    vtu_file : str
        Path to the output ``.vtu`` file.
    file_format : Optional[str]
        Optional format string understood by :func:`meshio.read`.
    """
    mesh = meshio.read(h5_file, file_format=file_format)
    meshio.write(vtu_file, mesh, binary=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert .h5 mesh to .vtu.")
    parser.add_argument("input_file", help="Path to the input .h5 file.")
    parser.add_argument("output_file", help="Path of the generated .vtu file.")
    parser.add_argument(
        "--format",
        dest="file_format",
        help=(
            "Optional meshio format name."
            " Use this when the input file is not automatically detected."
        ),
    )
    args = parser.parse_args()
    convert(args.input_file, args.output_file, args.file_format)


if __name__ == "__main__":  # pragma: no cover - manual invocation
    main()
