"""Utility to compute mesh quality metrics from Fluent ``.h5`` files.

This module replicates :mod:`paraview_mesh_metrics` but uses ParaView's
``OpenDataFile`` function to load Fluent ``.h5`` case files.  The metrics are computed using
ParaView's built-in ``MeshQuality`` filter in the same way as for ``.vtu``
meshes.
"""

from typing import Dict, Any, List
import argparse
import json

try:
    from paraview.simple import OpenDataFile, MeshQuality, Delete
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
        The dataset object returned by ``OpenDataFile`` for the input Fluent file.
    measure : str
        Name of the quality measure (e.g. ``'Aspect Ratio'``).

    Returns
    -------
    List[float]
        List containing the quality value for each cell in the mesh.
    """
    quality = MeshQuality(Input=reader)
    try:
        quality.TetQualityMeasure = measure
    except ValueError:
        # ParaView 5.13 renames "Skewness" to "Skew". Try this as fallback
        if measure == "Skewness":
            quality.TetQualityMeasure = "Skew"
        else:
            raise
    if hasattr(quality, "SaveCellQuality"):
        quality.SaveCellQuality = 1

    data = Fetch(quality)
    array = data.GetCellData().GetArray("Quality")
    values = [array.GetValue(i) for i in range(array.GetNumberOfTuples())]

    Delete(quality)
    return values


def compute_quality_metrics(h5_path: str) -> Dict[str, Any]:
    """Compute aspect ratio, minimum dihedral angle and skewness of a Fluent ``.h5`` file.

    Parameters
    ----------
    h5_path : str
        Path to the input Fluent ``.h5`` case or data file.

    Returns
    -------
    Dict[str, Any]
        Dictionary with the statistics for each metric. ``min``/``max``/``avg``
        are provided together with the list of per-cell values.
    """
    reader = OpenDataFile(h5_path)

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
        description="Compute mesh quality metrics for a Fluent .h5 file using ParaView"
    )
    parser.add_argument("mesh", help="Path to the input Fluent .h5 file")
    args = parser.parse_args()

    metrics = compute_quality_metrics(args.mesh)
    print(json.dumps(metrics, indent=2))


__all__ = ["compute_quality_metrics"]

if __name__ == "__main__":  # pragma: no cover - CLI entry point
    _main()
