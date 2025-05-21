# volume-mesh-benchmarks

## Python utilities

The repository provides a couple of helper scripts:

* `paraview_mesh_metrics.py` exposes a `compute_quality_metrics` function for
  reading a `.vtu` file and computing its aspect ratio, minimum dihedral
  angle and skewness using ParaView's `MeshQuality` filter. The script can
  also be run from the command line by passing the mesh file path:

  ```bash
  python paraview_mesh_metrics.py mesh.vtu
  ```

