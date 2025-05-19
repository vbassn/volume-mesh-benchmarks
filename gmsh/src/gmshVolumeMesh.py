# TO FIX: Handle buildings with holes as they break the boolean difference 
# and don't allow for proper volume mesh generation.

import gmsh
import dtcc
import numpy as np


class GMSHVolumeMesh:
  """
  Class to create a volume mesh using GMSH.
  """

  def __init__(self, 
              raster: dtcc.model.Raster,
              buildings: list[dtcc.model.Building],
              domain_height: float = 50.0,
              lod: dtcc.GeometryType = dtcc.GeometryType.LOD0,
              model_name: str = "Volume Mesh",
              mesh_size: float = 0.5,
              ):
    self.mesh_size = mesh_size
    self.model_name = model_name
    self.domain_height = domain_height
    self.lod = lod

    gmsh.initialize()
    gmsh.model.add(model_name)

    self.build_geometries(raster, buildings,domain_height)

  def __del__(self):
    """
    Destructor to finalize GMSH.
    """
    print("Finalizing GMSH...")
    gmsh.finalize()


  def _create_domain_box(self, bounds: dtcc.Bounds, domain_height: float):
    """
    Create a domain box.
    """
    dx, dy = bounds.xmax - bounds.xmin, bounds.ymax - bounds.ymin
    cut_box = gmsh.model.occ.addBox(bounds.xmin, bounds.ymin, bounds.zmin -1.0, dx, dy, domain_height)
    gmsh.model.occ.synchronize()
    return cut_box
  
  def _create_ground_surface(self, raster: dtcc.model.Raster):
    bounds = raster.bounds
    height = raster.data.shape[0]
    width = raster.data.shape[1]

    dx = (bounds.xmax - bounds.xmin)/(width - 1)
    dy = (bounds.ymax - bounds.ymin)/(height - 1)
    ctrl_pts = np.zeros((width, height), dtype=int)

    for i in range(height):
      for j in range(width):
        x = bounds.xmin + j * dx
        y = bounds.ymin + i * dy
        z = raster.data[i, j]
        ctrl_pts[i, j] = gmsh.model.occ.addPoint(x, y, z, 0.5)

    ground_surface = gmsh.model.occ.addBSplineSurface(
        ctrl_pts.flatten(), width)

    return ground_surface, ctrl_pts

  def _add_building(self, building: dtcc.model.Building):
    """
    Add a building to the GMSH model.
    """
    # Get the building footprint
    building_height = building.attributes["height"]
    ground_height = building.attributes["ground_height"]
    holes_points = []

    exterior_points = []
    exterior_curves = []
    for vertex in building.geometry[self.lod].vertices:
        x, y, z = vertex
        exterior_points.append(gmsh.model.occ.addPoint(x, y, 0.0, 0.5))
    for i in range(len(exterior_points)):
        exterior_curves.append(
            gmsh.model.occ.addLine(
                exterior_points[i], exterior_points[(i + 1) % len(exterior_points)])
        )
    exterior_curve_loop = gmsh.model.occ.addCurveLoop(exterior_curves)

    hole_curve_loops = []
    # for hole in building.geometry[self.lod].holes:
    #     hole_points = []
    #     for vertex in hole:
    #         x, y, z = vertex
    #         hole_points.append(gmsh.model.occ.addPoint(x, y, 0.0, 0.5))
    #     holes_points.append(hole_points)
    #     hole_curves = []
    #     for i in range(len(hole_points)):
    #         hole_curves.append(
    #             gmsh.model.occ.addLine(
    #                 hole_points[i], hole_points[(i + 1) % len(hole_points)])
    #         )
    #     hole_curve_loops.append(gmsh.model.occ.addCurveLoop(hole_curves))
    building_surface_tag = gmsh.model.occ.addPlaneSurface(
        [exterior_curve_loop] + hole_curve_loops)

    extr = gmsh.model.occ.extrude(
        [(2, building_surface_tag)],
        0, 0, ground_height + building_height
    )
    gmsh.model.occ.synchronize()
    vol_tags = [tag for (dim, tag) in extr if dim == 3]
    if not vol_tags:
        raise RuntimeError("Building extrusion failed to produce a volume.")
    return vol_tags
  
  def _add_buildings(self, buildings: list[dtcc.model.Building]):
    """
    Add buildings to the GMSH model.
    """
    building_tags = []
    for building in buildings:
        building_tag = self._add_building(building)
        building_tags += building_tag
    gmsh.model.addPhysicalGroup(3, building_tags, tag=2)
    gmsh.model.setPhysicalName(3, 1, "Buildings") 

    return building_tags
  

  def build_geometries(self, 
                  raster: dtcc.model.Raster,
                  buildings: list[dtcc.model.Building],
                  domain_height: float = 10.0):
    """
    Build the geometries for the GMSH model.
    """
    # Create the domain box
    domain_volume = self._create_domain_box(raster.bounds, domain_height)

    # Create the ground surface
    ground_surface_tag, _ = self._create_ground_surface(raster)

    extruded = gmsh.model.occ.extrude(
				[(2, ground_surface_tag)],
				0, 0, -2*domain_height
		)				
    extruded_vol = next(ent for ent in extruded if ent[0] == 3) 	

    gmsh.model.occ.synchronize() 

    print("Domain Volume Bounding Box:", gmsh.model.getBoundingBox(3, domain_volume))   
    print("Extruded surface Bounding Box:", gmsh.model.getBoundingBox(2, ground_surface_tag))      
    upper_parts, _ = gmsh.model.occ.cut(
      [(3, domain_volume)],           # object = the original box
      [extruded_vol],           # tool   = the volume under the surface
      removeObject=True,        # discard the old box
      removeTool=True           # discard the extruded “tool” volume
      )
    
    gmsh.model.occ.synchronize()       
    # find the new volume tag
    vol_tag = next(t for (dim,t) in gmsh.model.getEntities(3) if dim == 3)

    print("Extruded volume tag: ", vol_tag, domain_volume)

    # Add the buildings to the model
    building_tags = self._add_buildings(buildings)

    tools = [(3, btag) for btag in building_tags]   
    objRes, toolRes = gmsh.model.occ.cut(
          [(3, domain_volume)],
          tools
      )
    print("Cutting: ", objRes, toolRes)
    gmsh.model.occ.synchronize()


  def generate_mesh(self,min_mesh_size: float = 1.0, max_mesh_size: float = 10.0):
    """
    Generate the mesh for the GMSH model.
    """
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", min_mesh_size)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", max_mesh_size)
    gmsh.model.mesh.generate(3)
		

  def write_mesh(self, filename: str):
    """
    Write the mesh to a file.
    """
    gmsh.write(filename)

  def view_mesh(self):
    """
    Visualize the mesh using GMSH's built-in viewer.
    """
    gmsh.fltk.run()