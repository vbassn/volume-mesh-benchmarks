# Manually attach the data to a city model by reading from files and
# piecing together the data in a City object. Ideally, we should run
# the whole thing in a single script: download data, build meshes,
# run simulations and attach simulation results.

import h5py
from dtcc import *

# Load meshes
mesh = load_mesh("gbg_surface_mesh.pb")
volume_mesh = load_volume_mesh("gbg_volume_mesh.pb")


def load_h5(filename):
    "Helper function for reading simulation data from .h5."
    info(f"Reading H5 data from {filename}")
    with h5py.File(filename, mode="r") as h5:
        return h5["/Function/f/0"][...]


# Load simulation data
gbg_poisson_solution = load_h5("../fenics/gbg_poisson_output/solution.h5")
gbg_wave_solution = load_h5("../fenics/gbg_wave_output/final_solution.h5")
gbg_helmholtz_solution = load_h5("../fenics/gbg_helmholtz_output/solution.h5")
gbg_advdiff_solution = load_h5("../fenics/gbg_advdiff_output/final_solution.h5")
gbg_advdiff_velocity = load_h5("../fenics/gbg_advdiff_output/velocity.h5")

# Create fields
T = Field(
    name="Temperature",
    unit="C",
    description="Simulation in central Gothenburg by solving the Poisson equation",
    values=gbg_poisson_solution,
)
p = Field(
    name="Pressure",
    unit="N/m^2",
    description="Simulation in central Gothenburg by solving the wave equation",
    values=gbg_wave_solution,
)
u = Field(
    name="Complex pressure magnitude",
    unit="N/m^2",
    description="Simulation in central Gothenburg by solving the Helmholtz equation",
    values=gbg_wave_solution,
)
c = Field(
    name="Concentration",
    unit="M",
    description="Simulation in central Gothenburg by solving the advection-diffusion",
    values=gbg_advdiff_solution,
)
v = Field(
    name="Velocity",
    unit="m/s",
    description="Velocity field for the advection-diffusion equation",
    values=gbg_advdiff_velocity,
    dim=3,
)

# Create city model and attach data
city = City()
city.attributes["name"] = "Gothenburg"
city.attributes["created_date"] = "1621-06-04"
city.add_geometry(mesh)
city.add_geometry(volume_mesh)
city.add_field(T, VolumeMesh)
city.add_field(p, VolumeMesh)
city.add_field(u, VolumeMesh)
city.add_field(c, VolumeMesh)
city.add_field(v, VolumeMesh)


# Add some city objects
park_bench = CityObject()
lamp_post = CityObject()
city.add_child(park_bench)
city.add_child(lamp_post)

# Save to file
city.save("gbg_city.pb")

# Load city for inspection
_city = load_city("gbg_city.pb")
for f in _city.geometry[GeometryType.VOLUME_MESH].fields:
    print(f"{f.description}: {f.values.shape}")

print()
city.tree()
