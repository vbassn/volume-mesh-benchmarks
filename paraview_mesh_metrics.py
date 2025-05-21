"""Utility to compute mesh quality metrics using ParaView's Python API.

This module defines a helper function that loads a ``.vtu`` file and
computes several quality metrics using ParaView's built-in ``MeshQuality``
filter.
"""

from typing import Dict, Any, List
import argparse
import json

try:
    from paraview.simple import XMLUnstructuredGridReader, MeshQuality, Delete
    from paraview.servermanager import Fetch
except ModuleNotFoundError as exc:  # pragma: no cover - ParaView may not be installed
    raise ImportError(
        "ParaView is required to use this module. Make sure paraview.simple is available"
    ) from exc


def _compute_metric(reader: Any, measure: str) -> List[float]:
    """Internal helper that returns the raw quality values for *measure*.

    Parameters
    ----------
    reader : Any
        The ParaView reader object for the input ``.vtu`` file.
    measure : str
        Name of the quality measure (e.g. ``'Aspect Ratio'``).

    Returns
    -------
    List[float]
        List containing the quality value for each cell in the mesh.
    """
    quality = MeshQuality(Input=reader)
    quality.TetQualityMeasure = measure
    quality.SaveCellQuality = 1

    data = Fetch(quality)
    array = data.GetCellData().GetArray("Quality")
    values = [array.GetValue(i) for i in range(array.GetNumberOfTuples())]

    Delete(quality)
    return values


def compute_quality_metrics(vtu_path: str) -> Dict[str, Any]:
    """Compute aspect ratio, minimum dihedral angle and skewness of a ``.vtu`` file.

    Parameters
    ----------
    vtu_path : str
        Path to the input ``.vtu`` file.

    Returns
    -------
    Dict[str, Any]
        Dictionary with the statistics for each metric. ``min``/``max``/``avg``
        are provided together with the list of per-cell values.
    """
    reader = XMLUnstructuredGridReader(FileName=[vtu_path])

    metrics = {}
    for name, key in [
        ("Aspect Ratio", "aspect_ratio"),
        ("Minimum Dihedral Angle", "minimum_dihedral_angle"),
        ("Skewness", "skewness"),
    ]:
        values = _compute_metric(reader, name)
        stats = {
            "values": values,
            "min": min(values) if values else None,
            "max": max(values) if values else None,
            "avg": sum(values) / len(values) if values else None,
        }
        metrics[key] = stats

    Delete(reader)
    return metrics


def _main() -> None:
    """Entry point for command line execution."""
    parser = argparse.ArgumentParser(
        description="Compute mesh quality metrics for a VTU file using ParaView"
    )
    parser.add_argument("mesh", help="Path to the input .vtu mesh file")
    args = parser.parse_args()

    metrics = compute_quality_metrics(args.mesh)
    print(json.dumps(metrics, indent=2))


__all__ = ["compute_quality_metrics"]

if __name__ == "__main__":  # pragma: no cover - CLI entry point
    _main()
