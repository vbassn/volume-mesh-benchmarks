# amgcl-tests

## Python utilities

The repository provides a couple of helper scripts:

* `paraview_mesh_metrics.py` exposes a `compute_quality_metrics` function for
  reading a `.vtu` file and computing its aspect ratio, minimum dihedral
  angle and skewness using ParaView's `MeshQuality` filter.
* `h5_to_vtu.py` converts meshes stored in an HDF5 file into a `.vtu`
  representation. The script relies on the `meshio` package and preserves all
  point, cell and field data supported by `meshio`.
