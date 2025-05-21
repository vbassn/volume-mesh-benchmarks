import gmsh
import sys
import math
from pathlib import Path
import numpy as np
import dtcc
import time
from src.gmshVolumeMesh import GMSHVolumeMesh


h = 500.0
parameters = {
    "xmin": 319891,
    "ymin": 6399790,
    "xmax": 319891 + h,
    "ymax": 6399790 + h,
    "domain_height": 50.0,
    "mesh_size": 0.5,
    "model_name": "Volume Mesh",
}


def main():
    
    # Define bounds (a residential area in Helsingborg)
    bounds = dtcc.Bounds(parameters["xmin"], 
                         parameters["ymin"], 
                         parameters["xmax"], 
                         parameters["ymax"])

    # Download pointcloud and building footprints
    pointcloud = dtcc.download_pointcloud(bounds=bounds)
    buildings = dtcc.download_footprints(bounds=bounds)

    # Remove global outliers
    pointcloud = pointcloud.remove_global_outliers(3.0)

    # Build terrain raster and mesh
    raster = dtcc.builder.build_terrain_raster(
        pointcloud, cell_size=5, ground_only=True)

    # Extract roof points and compute building heights
    buildings = dtcc.extract_roof_points(buildings, pointcloud)
    buildings = dtcc.compute_building_heights(buildings, raster, overwrite=True)


    # Simplify building footprints
    lod = dtcc.GeometryType.LOD0
    buildings = dtcc.merge_building_footprints(
            buildings, lod=lod, max_distance=0.5, min_area=10)
    buildings = dtcc.simplify_building_footprints(
            buildings, tolerance=0.25, lod=lod)
    buildings = dtcc.fix_building_footprint_clearance(buildings, 0.5)

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    gmsh_time_begin = time.time()

    _volume_mesh_builder = GMSHVolumeMesh(raster=raster, 
                                        buildings=buildings,
                                        domain_height=parameters["domain_height"],
                                        lod=lod,
                                        model_name=parameters["model_name"],
                                        mesh_size=parameters["mesh_size"],
                                        )
    
    gmsh_mesh_time_begin= time.time()
    _volume_mesh_builder.generate_mesh()

    gmsh_time_end = time.time()
    print("GMSH Mesh generation time: ", gmsh_time_end - gmsh_mesh_time_begin)
    print("GMSH Total processing time: ", gmsh_time_end - gmsh_time_begin)

    _volume_mesh_builder.view_mesh()

if __name__ == "__main__":
	main()