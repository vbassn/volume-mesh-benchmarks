import meshio
import os
import sys
from pathlib import Path

def load_tetgen(tetgen_base: str) -> meshio.Mesh:
    """
    Load TetGen output files (.ele, .node, .face) into a meshio Mesh object.

    Parameters
    ----------
    tetgen_base : str
        The base name of the TetGen files (without extensions).
        For example, if the files are 'mesh.node', 'mesh.ele', and 'mesh.face',
        then tetgen_base should be 'mesh'.

    Returns
    -------
    meshio.Mesh
        A meshio Mesh object containing the data from the TetGen files.

    Notes
    -----
    - This function reads the TetGen files and combines them into a single meshio Mesh object.
    - Ensure that the TetGen files exist before calling this function.
    """

    tetgen_base = Path(tetgen_base)
    if tetgen_base.suffix in {".node", ".ele", ".face", ".edge"}:
        tetgen_base = tetgen_base.with_suffix("").absolute()
        print(f"Warning: Removed file extension from tetgen_base. New base: {tetgen_base}")

    node_path = tetgen_base.with_suffix(".node")
    ele_path = tetgen_base.with_suffix(".ele")
    face_path = tetgen_base.with_suffix(".face")
    edge_path = tetgen_base.with_suffix(".edge")
    print(f"Loading TetGen files: {node_path}, {ele_path}, {face_path}, {edge_path}")

    if not (os.path.exists(node_path) and os.path.exists(ele_path) and os.path.exists(face_path) and os.path.exists(edge_path)):
        raise FileNotFoundError("One or more TetGen files are missing.")

    # Read TetGen files using meshio
    mesh = meshio.read(node_path)
    # mesh.cells.extend(meshio.read(ele_path).cells)
    # mesh.cell_data.update(meshio.read(ele_path).cell_data)
    # mesh.cells.extend(meshio.read(face_path).cells)
    # mesh.cell_data.update(meshio.read(face_path).cell_data)

    return mesh


def tetgen_to_vtu(tetgen_base: str, vtu_path: str) -> None:
    """
    Convert TetGen output files (.ele, .node, .face) to a single VTU file.

    Parameters
    ----------
    tetgen_base : str
        The base name of the TetGen files (without extensions).
        For example, if the files are 'mesh.node', 'mesh.ele', and 'mesh.face',
        then tetgen_base should be 'mesh'.
    vtu_path : str
        The output path for the VTU file.

    Notes
    -----
    - This function reads the TetGen files and combines them into a single VTU file.
    - Ensure that the TetGen files exist before calling this function.
    """
    
    mesh = load_tetgen(tetgen_base)
    meshio.write(vtu_path, mesh)

    
if __name__ == "__main__": 
    if len(sys.argv) != 3:
        print("Usage: python tetgen_to_vtu.py <tetgen_base> <output_vtu>")
        sys.exit(1)

    tetgen_base = sys.argv[1]
    vtu_path = sys.argv[2]

    tetgen_to_vtu(tetgen_base, vtu_path)
    print(f"Converted TetGen files with base '{tetgen_base}' to VTU file '{vtu_path}'")