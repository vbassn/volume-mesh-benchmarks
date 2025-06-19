from typing import List

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
from dtcc_core.builder._dtcc_builder import (
    build_ground_mesh,
    VolumeMeshBuilder,
    compute_boundary_face_data,
)
import boundary_face_markers

def build_volume_mesh(
    pointcloud: dtcc.PointCloud,
    buildings: List[dtcc.Building],
    domain_height: float = 100.0,
    max_mesh_size: float = 10.0,
    compute_boundary_face_markers: bool = True,
) -> dtcc.VolumeMesh:

    # FIXME: Where do we set these parameters?
    min_mesh_angle = 30
    smoother_max_iterations = 5000
    smoothing_relative_tolerance = 0.005
    aspect_ratio_threshold = 10.0
    min_mesh_angle = 30
    debug_step = 7

    # Remove global outliers
    pointcloud = pointcloud.remove_global_outliers(3)

    # Build terrain raster
    terrain = dtcc.build_terrain_raster(
        pointcloud, cell_size=2, radius=3, ground_only=True
    )

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
        min(building.height, max_mesh_size) for building in footprints
    ]

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
        max_mesh_size,
        min_mesh_angle,
        True,
    )

    # FIXME: Should not need to convert from C++ to Python mesh.

    # Convert from C++ to Python
    ground_mesh = builder_mesh_to_mesh(_ground_mesh)
    _dem = raster_to_builder_gridfield(terrain)

    # Convert from Python to C++
    _surfaces = [
        create_builder_surface(building.lod0)
        for building in footprints
        if building is not None
    ]

    # Create volume mesh builder
    volume_mesh_builder = VolumeMeshBuilder(
        _surfaces, _dem, _ground_mesh, domain_height
    )

    # FIXME: How do we handle parameters?

    # Build volume mesh
    _volume_mesh = volume_mesh_builder.build(
        smoother_max_iterations,
        smoothing_relative_tolerance,
        0.0,
        aspect_ratio_threshold,
        debug_step,
    )
    volume_mesh = builder_volume_mesh_to_volume_mesh(_volume_mesh)


    if compute_boundary_face_markers:
        boundary_face_markers = compute_boundary_face_data(_volume_mesh)
        if boundary_face_markers is not None:
            volume_mesh.boundary_markers = boundary_face_markers
    

    return volume_mesh
