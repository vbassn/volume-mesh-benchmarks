#!/usr/bin/env python3

import sys
import os
import vtk
import numpy as np
from math import acos, degrees
from functools import partial
from multiprocessing import Pool

# Hardcode the tetra cell type ID as 10 (VTK_TET).
VTK_TET = 10

def get_unstructured_grid_from_multiblock(mb_data):
    """
    Recursively search through a vtkMultiBlockDataSet (mb_data)
    and return the first vtkUnstructuredGrid we find.
    If none is found, return None.
    """
    if not isinstance(mb_data, vtk.vtkMultiBlockDataSet):
        if isinstance(mb_data, vtk.vtkUnstructuredGrid):
            return mb_data
        return None

    num_blocks = mb_data.GetNumberOfBlocks()
    for i in range(num_blocks):
        block = mb_data.GetBlock(i)
        if block is None:
            continue

        if isinstance(block, vtk.vtkMultiBlockDataSet):
            found = get_unstructured_grid_from_multiblock(block)
            if found is not None:
                return found
        elif isinstance(block, vtk.vtkUnstructuredGrid):
            return block
    return None

def compute_metric(ugrid, metric_function):
    """
    Applies vtkMeshQuality to 'ugrid' using a function like:
        lambda mq: mq.SetTetQualityMeasureToAspectRatio()
    Returns a NumPy array of results or None if something fails.
    """
    mesh_quality = vtk.vtkMeshQuality()
    mesh_quality.SetInputData(ugrid)

    # Call the passed-in function to set the appropriate measure
    metric_function(mesh_quality)

    mesh_quality.Update()
    out_ugrid = mesh_quality.GetOutput()
    if out_ugrid is None:
        return None

    quality_array = out_ugrid.GetCellData().GetArray("Quality")
    if quality_array is None:
        return None

    num_cells = quality_array.GetNumberOfTuples()
    if num_cells == 0:
        return None

    # Convert VTK array to NumPy
    return np.array([quality_array.GetValue(i) for i in range(num_cells)])


def _compute_one_tet_ortho(args, points):
    """
    Worker function for one tetrahedron. 
    'args' is a tuple (cell_id, p0, p1, p2, p3) of point indices.
    'points' is an Nx3 NumPy array of all point coordinates.
    
    Returns: (cell_id, measure)
    """
    cell_id, pid0, pid1, pid2, pid3 = args

    # Extract the tetra's 4 vertices
    pts = points[[pid0, pid1, pid2, pid3]]  # shape: (4,3)

    # Cell centroid
    cell_centroid = np.mean(pts, axis=0)

    # Each tetra has 4 triangular faces:
    # We'll define them as sets of 3 out of the 4 points.
    face_ids_list = [
        [1, 2, 3],
        [0, 2, 3],
        [0, 1, 3],
        [0, 1, 2]
    ]
    
    face_qualities = []
    for face_ids in face_ids_list:
        fpts = pts[face_ids]  # shape (3,3)
        face_centroid = np.mean(fpts, axis=0)
        
        # Face normal = cross( (p1 - p0), (p2 - p0) )
        v1 = fpts[1] - fpts[0]
        v2 = fpts[2] - fpts[0]
        normal = np.cross(v1, v2)
        norm_len = np.linalg.norm(normal)
        
        # Vector from cell centroid to face centroid
        cF = face_centroid - cell_centroid
        cF_len = np.linalg.norm(cF)
        
        if norm_len < 1e-20 or cF_len < 1e-20:
            face_orth = 0.0
        else:
            # angle in degrees between normal and cF
            cos_val = np.dot(normal, cF) / (norm_len * cF_len)
            cos_val = max(-1.0, min(1.0, cos_val))  # clamp
            angle_deg = degrees(acos(cos_val))
            
            # measure = 1 at 90°, 0 at 0° or 180°
            face_orth = 1.0 - (abs(90.0 - angle_deg) / 90.0)
            if face_orth < 0.0:
                face_orth = 0.0
        
        face_qualities.append(face_orth)
    
    # The tetra's measure is the minimum face measure (i.e., the worst face)
    return (cell_id, min(face_qualities))


def compute_orthogonality_python_parallel(ugrid, nprocs=None):
    """
    Parallel computation of a simple orthogonality measure for tetrahedral cells.
    
    - We first copy relevant data from 'ugrid' into NumPy arrays (picklable).
    - Then we parallel-map each tetra using _compute_one_tet_ortho(...).
    
    Returns a NumPy array of length num_cells, with orthogonality measures
    (NaN for non-tet cells).
    """
    points_vtk = ugrid.GetPoints()
    if points_vtk is None:
        return None

    num_points = points_vtk.GetNumberOfPoints()
    num_cells = ugrid.GetNumberOfCells()

    # Convert points to a NumPy array of shape (num_points, 3)
    points = np.array([points_vtk.GetPoint(i) for i in range(num_points)], dtype=float)

    # Collect all TET cells in a list of (cell_id, p0, p1, p2, p3)
    tetra_list = []
    for cell_id in range(num_cells):
        print(f"Processing cell {cell_id} of {num_cells}, i.e. percent {cell_id / num_cells:.2%}", end="\r")
        cell = ugrid.GetCell(cell_id)
        if cell.GetCellType() == VTK_TET:
            pt_ids = cell.GetPointIds()
            # We'll store the 4 point indices
            p0 = pt_ids.GetId(0)
            p1 = pt_ids.GetId(1)
            p2 = pt_ids.GetId(2)
            p3 = pt_ids.GetId(3)
            tetra_list.append((cell_id, p0, p1, p2, p3))

    # Prepare the output array (default to NaN)
    orth_values = np.full(num_cells, np.nan, dtype=float)

    # We'll do a parallel map using a Pool
    # partial() is used to freeze the 'points' argument in place
    worker = partial(_compute_one_tet_ortho, points=points)

    with Pool(processes=nprocs) as pool:
        # results is a list of (cell_id, measure)
        results = pool.map(worker, tetra_list)

    # Place each measure in the orth_values array at the appropriate index
    for (cid, measure) in results:
        orth_values[cid] = measure

    return orth_values


def print_stats(data, label):
    """Helper to print min, average, and max for a data array."""
    if data is None or len(data) == 0 or np.all(np.isnan(data)):
        print(f"No data to compute '{label}'.")
        return

    valid_data = data[~np.isnan(data)]
    if len(valid_data) == 0:
        print(f"No valid tetra cells to compute '{label}'.")
        return

    avg_val = np.mean(valid_data)
    min_val = np.min(valid_data)
    max_val = np.max(valid_data)
    print(f"{label}:")
    print(f"  Average: {avg_val:.4f}")
    print(f"  Minimum: {min_val:.4f}")
    print(f"  Maximum: {max_val:.4f}")

def main(filename):
    # Decide which reader to use
    if filename.lower().endswith(".cas.h5"):
        print("Using vtkFLUENTCFFReader to read Fluent case file.")
        reader = vtk.vtkFLUENTCFFReader()
        reader.SetFileName(filename)
        reader.Update()
        output_data = reader.GetOutputDataObject(0)
        ugrid = get_unstructured_grid_from_multiblock(output_data)

    elif filename.lower().endswith(".vtu"):
        print("Using vtkXMLUnstructuredGridReader to read .vtu file.")
        reader = vtk.vtkXMLUnstructuredGridReader()
        reader.SetFileName(filename)
        reader.Update()
        ugrid = reader.GetOutput()
    else:
        print("Unrecognized file extension. Only .cas.h5 or .vtu are supported.")
        sys.exit(1)

    if not ugrid:
        print("No vtkUnstructuredGrid found in the file.")
        return

    # 1) Tet Aspect Ratio (VTK)
    data_ar = compute_metric(ugrid, lambda mq: mq.SetTetQualityMeasureToAspectRatio())
    print_stats(data_ar, "Tet Aspect Ratio (VTK)")

    # 2) Equi-Angle Skew (VTK)
    data_eas = compute_metric(ugrid, lambda mq: mq.SetTetQualityMeasureToEquiangleSkew())
    print_stats(data_eas, "Tet Equi-Angle Skew (VTK)")

    # 3) Minimum Dihedral Angle (VTK)
    data_dihedral = compute_metric(ugrid, lambda mq: mq.SetTetQualityMeasureToMinAngle())
    print_stats(data_dihedral, "Tet Minimum Dihedral Angle (VTK)")

    # 4) Orthogonality in Parallel (Pure Python)
    data_py_orth = compute_orthogonality_python_parallel(ugrid, nprocs=None)
    print_stats(data_py_orth, "Tet Orthogonality (Python, Parallel)")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <filename.cas.h5 or filename.vtu>")
        sys.exit(1)

    main(sys.argv[1])
