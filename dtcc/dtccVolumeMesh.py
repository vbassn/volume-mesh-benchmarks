#!/usr/bin/env python
import sys
from pathlib import Path
# from dtcc import *
import dtcc

# FIXME: Obscure imports
from dtcc_core.builder.model_conversion import (
    create_builder_polygon,
    create_builder_surface,
    builder_mesh_to_mesh,
    raster_to_builder_gridfield,
    builder_volume_mesh_to_volume_mesh,
)

# FIXME: Obscure imports
from dtcc_core.builder._dtcc_builder import build_ground_mesh, VolumeMeshBuilder, compute_boundary_mesh

# Set parameters
h = 500.0
_parameters = {}


_parameters["xmin"] = 319891
_parameters["ymin"] = 6399790
_parameters["xmax"] = 319891 + h
_parameters["ymax"] = 6399790 + h
_parameters["domain_height"] = 50.0
_parameters["max_mesh_size"] = 10
_parameters["min_mesh_angle"] = 30
_parameters["smoother_max_iterations"] = 5000
_parameters["smoothing_relative_tolerance"] = 0.0005
_parameters["debug_step"] = 7
_parameters["aspect_ratio_threshold"] = 10.0
_parameters["output_directory"] = Path(__file__).parent

# Define bounds (a residential area in Helsingborg)
bounds = dtcc.Bounds(_parameters["xmin"], 
                     _parameters["ymin"], 
                     _parameters["xmax"], 
                     _parameters["ymax"])

# Download pointcloud and building footprints
pointcloud = dtcc.download_pointcloud(bounds=bounds)
buildings = dtcc.download_footprints(bounds=bounds)

# FIXME: Are all operations on point clouds and footprints out-place?
# FIXME: Explicit parameter 3 for remove_global_outliers() is not clear.

# Remove global outliers
pointcloud = pointcloud.remove_global_outliers(3)



# Build terrain raster
terrain = dtcc.build_terrain_raster(pointcloud, cell_size=2, radius=3, ground_only=True)

# Extract roof points
footprints = dtcc.extract_roof_points(
    buildings, pointcloud, statistical_outlier_remover=True
)

# Compute building heights
footprints = dtcc.compute_building_heights(footprints, terrain, overwrite=True)

merge_footprints = dtcc.merge_building_footprints
simplify_footprints = dtcc.simplify_building_footprints
fix_footprint_clearance = dtcc.fix_building_footprint_clearance



# Merge and simplify building footprints
lod = dtcc.GeometryType.LOD0
footprints = merge_footprints(footprints, lod=lod, max_distance=0.5, min_area=10)


footprints = simplify_footprints(footprints, 0.25, lod=lod)
footprints = fix_footprint_clearance(footprints, 0.5)

# FIXME: Is this the ideal resolution? Handle it automatically?


# Set subdomain resolution to half the building height
subdomain_resolution = [
    min(building.height, _parameters["max_mesh_size"]) for building in footprints
]

# subdomain_resolution = []


# FIXME: Should not need to convert from Python to C++.

# Convert from Python to C++
_footprints = [
    create_builder_polygon(building.lod0.to_polygon())
    for building in footprints
    if building is not None
]

# FIXME: Pass bounds as argument (not xmin, ymin, xmax, ymax).

# Build ground mesh
_ground_mesh = build_ground_mesh(
    _footprints,
    subdomain_resolution,
    terrain.bounds.xmin,
    terrain.bounds.ymin,
    terrain.bounds.xmax,
    terrain.bounds.ymax,
    _parameters["max_mesh_size"],
    _parameters["min_mesh_angle"],
    True,
)

# FIXME: Should not need to convert from C++ to Python mesh.

# Convert from C++ to Python
ground_mesh = builder_mesh_to_mesh(_ground_mesh)

# Save ground mesh to file
ground_mesh.save(_parameters["output_directory"] / "ground_mesh.vtu")
ground_mesh.save("ground_mesh.pb")

_dem = raster_to_builder_gridfield(terrain)

# Convert from Python to C++
_surfaces = [
    create_builder_surface(building.lod0)
    for building in footprints
    if building is not None
]


# Create volume mesh builder
volume_mesh_builder = VolumeMeshBuilder(_surfaces, _dem, _ground_mesh, 100.0)

# Build volume mesh
_volume_mesh = volume_mesh_builder.build(
    _parameters["smoother_max_iterations"],
    _parameters["smoothing_relative_tolerance"],
    0.0,
    _parameters["aspect_ratio_threshold"],
    _parameters["debug_step"],
)

_boundary_mesh = compute_boundary_mesh(_volume_mesh)

# FIXME: Should not need to convert from C++ to Python

# Convert from C++ to Python
volume_mesh = builder_volume_mesh_to_volume_mesh(_volume_mesh)
boundary_mesh = builder_mesh_to_mesh(_boundary_mesh)


# Save volume mesh to file
volume_mesh.save(_parameters["output_directory"] / f"volume_mesh_{_parameters["debug_step"]}.vtu")
boundary_mesh.save(_parameters["output_directory"] / f"boundary_mesh_{_parameters["debug_step"]}.vtu")
