# volume-mesh-benchmarks

## Gmsh vs dtcc

George write here

## Python utilities

The repository provides a couple of helper scripts:

* `scripts/paraview_mesh_metrics.py` calculates basic mesh metrics using Paraview's API. The script can be run from the command line by passing the mesh file path:

  ```bash
  python3 scripts/paraview_mesh_metrics.py mesh.vtu
  ```



* `dtcc/dtcc_volume_mesh_runs.py` This script performs a systematic parameter sweep of the DTCC volume-mesh generator and records detailed metrics for each run in the `benchmarks/dtcc_runs_results.csv`

  Run: 
  ```bash
  python dtcc/run_dtcc_benchmarks.py
  ```
  after modifying the parameter grid provided in `dtcc/dtcc_volume_mesh_runs.py`:

  ```python
  param_grid = {
    "xmin": [319891],
    "ymin": [6399790],
    "xmax": [319891 + 200.0 , 319891 + 500.0],
    "ymax": [6399790 + 500.0, 6399790 + 600.0],
    "max_mesh_size": [10],
    "min_mesh_angle": [30],
    "smoothing_relative_tolerance": [0.005],
    "debug_step": [7]
  }

  ```

Each call to the C++ Volume Mesh Builder creates a `mesh_report.txt` reporting the execution time, number of vertices and cells and min/max/median aspect ratio which is used to populate `benchmarks/dtcc_runs_results.csv`

* `benchmarks/benchmark_plots.ipynb` produces the performance plots using `benchmarks/dtcc_runs_results.csv`
