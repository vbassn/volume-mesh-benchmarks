from typing import List
import dtcc
import tetwrap
from pathlib import Path


def main():
    x0 = 319995.962899
    y0 = 6399009.716755
    L = 500.0
    H = 100.0

    # TetGen quality parameters (see TetGen manual and tetwrap/switches.py)
    max_edge_radius_ratio = 1.414
    min_dihedral_angle = 30.0
    max_tet_volume = 10.0

    # Location of the current file
    current_file = Path(__file__).resolve().parent

    # Make an "output" folder next to this file
    output_dir = current_file / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Define bounds
    bounds = dtcc.Bounds(x0 - 0.5 * L, y0 - 0.5 * L, x0 + 0.5 * L, y0 + 0.5 * L)

    # Create city and set bounds
    city = dtcc.City()
    city.bounds = bounds

    # Download pointcloud and building footprints
    city.download_pointcloud(bounds=bounds, filter_on_z_bounds=True)
    city.download_footprints(bounds=bounds)

    # Build city mesh
    city.building_heights_from_pointcloud()
    mesh = city.build_city_surface_mesh()

    mesh.view()

    # Build simple box mesh

    # Download pointcloud and building footprints
    # pointcloud = dtcc.download_pointcloud(bounds=bounds)
    # buildings = dtcc.download_footprints(bounds=bounds)

    # Remove global outliers
    # pointcloud = pointcloud.remove_global_outliers(3.0)

    # Build terrain raster
    # raster = dtcc.build_terrain_raster(
    #    pointcloud, cell_size=2, radius=3, ground_only=True
    # )

    # Extract roof points and compute building heights
    # buildings = dtcc.extract_roof_points(buildings, pointcloud)
    # buildings = dtcc.compute_building_heights(buildings, raster, overwrite=True)

    # Create city and add geometries
    # city = dtcc.City()
    # city.add_terrain(raster)
    # city.add_buildings(buildings, remove_outside_terrain=True)

    # Build surface mesh
    # mesh = dtcc.build_city_mesh(city, lod=dtcc.GeometryType.LOD1)

    mesh.offset((0.0 - (x0 - 0.5 * L), 0.0 - (y0 - 0.5 * L), 0))

    switches_params = {
        "plc": True,
        "quality": (
            max_edge_radius_ratio,
            min_dihedral_angle,
        ),  # ( max radius-edge ratio, min dihedral angle )
        "max_volume": max_tet_volume,
        "verbose": True,
    }

    volume_mesh = tetwrap.build_volume_mesh(
        mesh=mesh,
        build_top_sidewalls=True,
        top_height=H,
        switches_params=switches_params,
    )

    volume_mesh.save(output_dir / "tetgen_volume_mesh.vtu")


if __name__ == "__main__":
    main()
