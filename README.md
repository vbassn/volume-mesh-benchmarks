# volume-mesh-benchmarks

## Gmsh vs dtcc

To evaluate the quality and performance of dtcc in generating LoD1 volume meshes, we replicate the workflow using Gmsh for comparison.

This section explains our approach using Gmsh to build a 3D volume mesh from a terrain raster and a set of building footprints, using Gmsh’s Python API.


1. **Create the Domain Box**  
  Compute the extent of the raster bounds (`xmin`, `ymin`, `xmax`, `ymax`) and set a base depth (`zmin - 1.0`). Based on those create a box volume spanning the terrain footprint and rising to `domain_height` above ground.

2. **Build the Terrain Surface**  
   - Sample the raster grid of size (height × width)  
   - For each raster cell `(i,j)` compute its world `(x,y,z)` coordinate  
   - Add Gmsh control points at these coordinates (`gmsh.model.occ.addPoint`)  
   - Create a B-spline surface through the points.

3. **Extrude the Ground Surface Downward**  
   - Extrude the 2D B-spline surface down by `2 * domain_height` to form a subsurface volume  
   - Extract the resulting volume entity from the extrusion result  

4. **Carve Out the “Upper” Domain**  
   - Perform a Boolean cut: subtract the extruded subsurface volume from the original domain box  
   - This yields the volume above the terrain surface but below the top of the domain box  

5. **Add Building Volumes**  
   For each building in your list:
   - **Footprint Loop**  
     - Extract the exterior boundary and any interior holes, ensuring proper (CCW/CW) vertex ordering  
     - Add Gmsh points along the footprint (`addPoint`) and connect them into line loops (`addLine` + `addCurveLoop`)  
     - Create a planar surface from the loop(s)  
   - **Extrusion**  
     - Extrude the 2D building footprint upward by `(ground_height + building_height)`  
     - Collect the resulting 3D volume tag(s)  
   - Register all building volumes as a physical group for post-processing  

6. **Subtract Buildings from the Domain**  
   - Boolean‐cut the domain volume by each building volume, removing the building solids from the meshable domain  
   - Synchronize one last time  

7. **Generate the Mesh**  
   - Configure mesh element sizes via  
     ```python
     gmsh.option.setNumber("Mesh.CharacteristicLengthMin", min_mesh_size)
     gmsh.option.setNumber("Mesh.CharacteristicLengthMax", max_mesh_size)
     ```  
   - Invoke the 3D mesh generator: `gmsh.model.mesh.generate(3)`

For a fair comparison of volume geometry processing and mesh generation, we strictly adhere to the dtcc workflow up to the preprocessing stage, including terrain (raster) and building footprint extraction. We then diverge only at the volume mesh generation step, using Gmsh instead. This ensures that both dtcc and Gmsh operate on identical input data.


**Usage Example**  
To generate a volume mesh using Gmsh:

1. run the demo on `./gmsh`
```bash
cd gmsh
python gmsh_demo.py
```

or incorporate the `GMSHVolumeMesh` class:

```python
from dtcc.model import Raster, Building
# load your raster + buildings...
mesh = GMSHVolumeMesh(raster, buildings,
                      domain_height=50.0,
                      mesh_size=0.5,
                      lod=GeometryType.LOD0)
mesh.generate_mesh(min_mesh_size=1.0, max_mesh_size=10.0)
mesh.write_mesh("city_volume.msh")
mesh.view_mesh()
```
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
