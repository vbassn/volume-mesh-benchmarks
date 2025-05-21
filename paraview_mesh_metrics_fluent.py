"""Utility to compute mesh quality metrics from Fluent ``.h5`` files.

This module replicates :mod:`paraview_mesh_metrics` but uses ParaView's
``OpenDataFile`` function to load Fluent ``.h5`` case files.  The metrics are computed using
ParaView's built-in ``MeshQuality`` filter in the same way as for ``.vtu``
meshes.
"""

from typing import Dict, Any, List
import argparse
import json
import sys

try:
    from paraview.simple import OpenDataFile, MeshQuality, Delete, Sphere
    from paraview.servermanager import Fetch
except ModuleNotFoundError as exc:  # pragma: no cover - ParaView may not be installed
    raise ImportError(
        "ParaView is required to use this module. Make sure paraview.simple is available"
    ) from exc


def print_available_quality_measures(silent=False):
    """Print the available quality measures for the current ParaView version."""
    # Create a dummy reader and quality filter
    dummy = Sphere()
    quality = MeshQuality(Input=dummy)
    
    measures = []
    if hasattr(quality, "GetProperty") and hasattr(quality.GetProperty("TetQualityMeasure"), "GetAvailable"):
        measures = quality.GetProperty("TetQualityMeasure").GetAvailable()
        if not silent:
            print("Available TetQualityMeasure values:")
            for measure in measures:
                print(f"  - {measure}")
    else:
        if not silent:
            print("Could not determine available TetQualityMeasure values")
    
    Delete(quality)
    Delete(dummy)
    
    return measures


def _compute_metric(reader: Any, measure: str, silent=False) -> List[float]:
    """Internal helper that returns the raw quality values for *measure*.

    Parameters
    ----------
    reader : Any
        The dataset object returned by ``OpenDataFile`` for the input Fluent file.
    measure : str
        Name of the quality measure (e.g. ``'Aspect Ratio'``).
    silent : bool, optional
        If True, suppress verbose output, by default False

    Returns
    -------
    List[float]
        List containing the quality value for each cell in the mesh.
    """
    quality = MeshQuality(Input=reader)
    
    # Get the available quality measures
    try:
        # Try to set the requested measure
        quality.TetQualityMeasure = measure
    except ValueError as e:
        # If that fails, print available options and try alternative names
        if not silent:
            print(f"Error setting {measure} as TetQualityMeasure: {e}")
        
        # Get available options by inspecting the property
        available_measures = []
        if hasattr(quality, "GetProperty") and hasattr(quality.GetProperty("TetQualityMeasure"), "GetAvailable"):
            available_measures = quality.GetProperty("TetQualityMeasure").GetAvailable()
            if not silent:
                print(f"Available TetQualityMeasure options: {available_measures}")
        
        # Try common alternatives
        alternatives = {
            "Skewness": ["Skew", "EquiAngleSkew", "EquivolumeSkininess"], 
            "Skew": ["Skewness", "EquiAngleSkew", "EquivolumeSkininess"],
            "Aspect Ratio": ["AspectRatio", "Aspect", "AspectGamma"],
            "Minimum Dihedral Angle": ["MinDihedralAngle", "DihedralAngle", "Min Angle"]
        }
        
        found = False
        if measure in alternatives:
            for alt in alternatives[measure]:
                try:
                    quality.TetQualityMeasure = alt
                    if not silent:
                        print(f"Successfully used alternative: {alt}")
                    found = True
                    break
                except ValueError:
                    continue
        
        if not found:
            raise ValueError(f"Could not find a valid alternative for {measure}. Available options: {available_measures}")

    if hasattr(quality, "SaveCellQuality"):
        try:
            quality.SaveCellQuality = 1
        except AttributeError:
            pass

    data = Fetch(quality)

    def _extract(ds):
        if not ds or not hasattr(ds, "GetCellData"):
            return []
        cell_data = ds.GetCellData()
        if not cell_data:
            return []
        arr = cell_data.GetArray("Quality")
        if not arr:
            return []
        return [arr.GetValue(i) for i in range(arr.GetNumberOfTuples())]

    values = []
    if hasattr(data, "GetNumberOfBlocks"):
        for i in range(data.GetNumberOfBlocks()):
            block = data.GetBlock(i)
            if block is not None:
                values.extend(_extract(block))
    else:
        values = _extract(data)

    Delete(quality)
    return values


def compute_quality_metrics(h5_path: str, silent=False) -> Dict[str, Any]:
    """Compute aspect ratio, minimum dihedral angle and skewness of a Fluent ``.h5`` file.

    Parameters
    ----------
    h5_path : str
        Path to the input Fluent ``.h5`` case or data file.
    silent : bool, optional
        If True, suppress verbose output, by default False

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
        try:
            values = _compute_metric(reader, name, silent)
            stats = {
                "min": min(values) if values else None,
                "max": max(values) if values else None,
                "avg": sum(values) / len(values) if values else None,
            }
            # Only include raw values if not in silent mode
            if not silent:
                stats["values"] = values
            metrics[key] = stats
        except Exception as e:
            if not silent:
                print(f"Error computing {name}: {e}")
            metrics[key] = {
                "min": None,
                "max": None,
                "avg": None,
            }
            if not silent:
                metrics[key]["values"] = []
                metrics[key]["error"] = str(e)

    Delete(reader)
    return metrics


def _main() -> None:
    """Entry point for command line execution."""
    parser = argparse.ArgumentParser(
        description="Compute mesh quality metrics for a Fluent .h5 file using ParaView"
    )
    parser.add_argument("mesh", help="Path to the input Fluent .h5 file")
    parser.add_argument("--list-measures", action="store_true", 
                      help="List available quality measures and exit")
    parser.add_argument("--silent", "-s", action="store_true",
                      help="Suppress verbose output and only show min/max/avg stats")
    parser.add_argument("--no-values", action="store_true",
                      help="Don't include raw cell values in the output (smaller JSON)")
    parser.add_argument("--format", choices=["json", "simple"], default="json",
                      help="Output format: 'json' (default) or 'simple' for basic text")
    args = parser.parse_args()
    
    if args.list_measures:
        print_available_quality_measures(args.silent)
        return

    # Redirect stdout to /dev/null if in silent mode
    if args.silent:
        original_stdout = sys.stdout
        sys.stdout = open('/dev/null', 'w')
    
    try:
        metrics = compute_quality_metrics(args.mesh, args.silent or args.no_values)
        
        # Restore stdout if in silent mode
        if args.silent:
            sys.stdout.close()
            sys.stdout = original_stdout
        
        if args.format == "json":
            print(json.dumps(metrics, indent=2))
        else:  # simple format
            print("Mesh Quality Report for:", args.mesh)
            print("-" * 40)
            for metric_name, stats in metrics.items():
                print(f"{metric_name.replace('_', ' ').title()}:")
                if stats["min"] is not None:
                    print(f"  Min: {stats['min']:.6f}")
                    print(f"  Max: {stats['max']:.6f}")
                    print(f"  Avg: {stats['avg']:.6f}")
                else:
                    print("  Failed to compute this metric")
                print()
    finally:
        # Make sure stdout is restored even if an exception occurs
        if args.silent:
            try:
                sys.stdout.close()
            except:
                pass
            sys.stdout = original_stdout


__all__ = ["compute_quality_metrics"]

if __name__ == "__main__":  # pragma: no cover - CLI entry point
    _main()