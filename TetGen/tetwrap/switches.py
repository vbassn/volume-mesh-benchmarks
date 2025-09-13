from copy import deepcopy

"""Helpers to build TetGen command-line switches from descriptive kwargs.
tetgen [-pYrq_Aa_miO_S_T_XMwcdzfenvgkJBNEFICQVh] input_file
    -p  Tetrahedralizes a piecewise linear complex (PLC).
    -Y  Preserves the input surface mesh (does not modify it).
    -r  Reconstructs a previously generated mesh.
    -q  Refines mesh (to improve mesh quality).
    -R  Mesh coarsening (to reduce the mesh elements).
    -A  Assigns attributes to tetrahedra in different regions.
    -a  Applies a maximum tetrahedron volume constraint.
    -m  Applies a mesh sizing function.
    -i  Inserts a list of additional points.
    -O  Specifies the level of mesh optimization.
    -S  Specifies maximum number of added points.
    -T  Sets a tolerance for coplanar test (default 1e-8).
    -X  Suppresses use of exact arithmetic.
    -M  No merge of coplanar facets or very close vertices.
    -w  Generates weighted Delaunay (regular) triangulation.
    -c  Retains the convex hull of the PLC.
    -d  Detects self-intersections of facets of the PLC.
    -z  Numbers all output items starting from zero.
    -f  Outputs all faces to .face file.
    -e  Outputs all edges to .edge file.
    -n  Outputs tetrahedra neighbors to .neigh file.
    -v  Outputs Voronoi diagram to files.
    -g  Outputs mesh to .mesh file for viewing by Medit.
    -k  Outputs mesh to .vtk file for viewing by Paraview.
    -J  No jettison of unused vertices from output .node file.
    -B  Suppresses output of boundary information.
    -N  Suppresses output of .node file.
    -E  Suppresses output of .ele file.
    -F  Suppresses output of .face and .edge file.
    -I  Suppresses mesh iteration numbers.
    -C  Checks the consistency of the final mesh.
    -Q  Quiet:  No terminal output except errors.
    -V  Verbose:  Detailed information, more terminal output.
    -h  Help:  A brief instruction for using TetGen.

"""
DEFAULT_TETGEN_PARAMS = {
    # Core
    "plc": True,                    # -p : input is a PLC
    "preserve_surface": False,      # -Y : keep input surface unchanged
    "reconstruct": False,           # -r : reconstruct a previous mesh
    "coarsen": False,               # -R : coarsen mesh
    "assign_region_attributes": False,  # -A

    # Sizing / quality
    "quality": None,                # -q{val} or -q if True
    "max_volume": None,             # -a{val} or -a if True (per-region)
    "sizing_function": None,        # -m{token} or -m if True
    "insert_points": None,          # -i{token} or -i if True
    "optimize_level": None,         # -O{int}
    "max_added_points": None,       # -S{int}
    "coplanar_tolerance": None,     # -T{float}

    # Numerical / topology
    "no_exact_arithmetic": False,   # -X
    "no_merge_coplanar": False,     # -M
    "weighted_delaunay": False,     # -w
    "keep_convex_hull": False,      # -c
    "detect_self_intersections": False,  # -d

    # Numbering / output control
    "zero_numbering": False,        # -z (output files start from 0)
    "output_faces": False,          # -f
    "output_edges": False,          # -e
    "output_neighbors": False,      # -n
    "output_voronoi": False,        # -v
    "output_medit_mesh": False,     # -g
    "output_vtk": False,            # -k
    "no_jettison_unused_vertices": False,  # -J
    "suppress_boundary_output": False,     # -B
    "suppress_node_file": False,    # -N
    "suppress_ele_file": False,     # -E
    "suppress_face_edge_files": False,     # -F
    "suppress_iteration_numbers": False,   # -I
    "check_mesh": False,            # -C

    # Verbosity
    "quiet": False,                 # -Q
    "verbose": False,               # -V

    # Misc
    "help": False,                  # -h
    "extra": "",                    # anything to append verbatim
    # Alias: if you prefer `refine=True` to mean bare `-q`
    "refine": False,                # -> -q (only if quality is None)
}

def tetgen_defaults():
    """Return a fresh copy of default TetGen parameters you can tweak."""
    return deepcopy(DEFAULT_TETGEN_PARAMS)

def _fmt_num(x):
    if x is True:   # bare flag (no value)
        return ""
    if isinstance(x, float):
        return f"{x:g}"
    return str(x)

def _emit_q(cfg) -> str:
    # precedence: explicit radius_edge_ratio/min_dihedral_angle override 'quality'
    ratio = cfg.get("radius_edge_ratio")
    angle = cfg.get("min_dihedral_angle")
    q = cfg.get("quality")

    if ratio is None and angle is None and q is None:
        return ""

    # Parse compound forms
    if (ratio is None and angle is None) and q is not None:
        if q is True:
            return "q"
        if isinstance(q, (int, float)):
            ratio = q
        elif isinstance(q, (tuple, list)) and len(q) == 2:
            ratio, angle = q
        elif isinstance(q, dict):
            ratio = q.get("ratio", ratio)
            angle = q.get("min_dihedral", angle)
        else:
            raise ValueError("quality must be True | number | (ratio, angle) | {'ratio':..,'min_dihedral':..}")

    # Build q string
    s = "q"
    if ratio is not None:
        s += _fmt_num(ratio)
        if angle is not None:
            s += f"/{_fmt_num(angle)}"
    elif angle is not None:
        # angle without ratio: TetGen expects q[ratio]/angle — emit bare 'q' then '/angle'
        s += f"/{_fmt_num(angle)}"
    return s


def build_tetgen_switches(params=None, **overrides) -> str:
    """
    Build a TetGen switch string from descriptive kwargs.

    Example:
        build_tetgen_switches(plc=True, quality=1.414, max_volume=0.1) -> 'pq1.414a0.1'
    """
    cfg = tetgen_defaults()
    if params:
        cfg.update(params)
    if overrides:
        cfg.update(overrides)

    # sanity
    if cfg["quiet"] and cfg["verbose"]:
        raise ValueError("`quiet` (-Q) and `verbose` (-V) are mutually exclusive.")

    parts = []

    # Simple toggles (order chosen to be readable)
    toggles = [
        ("plc", "p"),
        ("preserve_surface", "Y"),
        ("reconstruct", "r"),
        ("coarsen", "R"),
        ("assign_region_attributes", "A"),
        ("no_exact_arithmetic", "X"),
        ("no_merge_coplanar", "M"),
        ("weighted_delaunay", "w"),
        ("keep_convex_hull", "c"),
        ("detect_self_intersections", "d"),
        ("zero_numbering", "z"),
        ("output_faces", "f"),
        ("output_edges", "e"),
        ("output_neighbors", "n"),
        ("output_voronoi", "v"),
        ("output_medit_mesh", "g"),
        ("output_vtk", "k"),
        ("no_jettison_unused_vertices", "J"),
        ("suppress_boundary_output", "B"),
        ("suppress_node_file", "N"),
        ("suppress_ele_file", "E"),
        ("suppress_face_edge_files", "F"),
        ("suppress_iteration_numbers", "I"),
        ("check_mesh", "C"),
        ("quiet", "Q"),
        ("verbose", "V"),
        ("help", "h"),
    ]
    for key, flag in toggles:
        if cfg.get(key):
            parts.append(flag)

    # Valued flags (value or bare if True)
    # quality: -q, -q{val}

    # quality
    q_str = _emit_q(cfg)
    if q_str: parts.append(q_str)

    # max volume: -a, -a{val}
    a = cfg.get("max_volume")
    if a is not None:
        parts.append("a" + _fmt_num(a))

    # sizing function: -m, -m{token}
    m = cfg.get("sizing_function")
    if m is not None:
        parts.append("m" + ("" if m is True else str(m)))

    # insert points: -i, -i{token}
    i = cfg.get("insert_points")
    if i is not None:
        parts.append("i" + ("" if i is True else str(i)))

    # optimization level: -O{int}
    O = cfg.get("optimize_level")
    if O is not None:
        parts.append("O" + _fmt_num(O))

    # max added points: -S{int}
    S = cfg.get("max_added_points")
    if S is not None:
        parts.append("S" + _fmt_num(S))

    # coplanar tolerance: -T{float}
    T = cfg.get("coplanar_tolerance")
    if T is not None:
        parts.append("T" + _fmt_num(T))

    if cfg.get("extra"):
        parts.append(cfg["extra"])

    return "".join(parts)
