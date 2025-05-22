#!/usr/bin/env python3

import sys
import os
import vtk
import numpy as np

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
    Applies vtkMeshQuality to the given ugrid with a function like:
        lambda mq: mq.SetTetQualityMeasureToAspectRatio()
    Returns a numpy array of the results or None if the metric isn't available.
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

def print_stats(data, label):
    """Helper to print min, average, and max for a data array."""
    if data is None or len(data) == 0:
        print(f"No data to compute '{label}'.")
        return

    avg_val = np.mean(data)
    min_val = np.min(data)
    max_val = np.max(data)
    print(f"{label}:")
    print(f"  Average: {avg_val:.4f}")
    print(f"  Minimum: {min_val:.4f}")
    print(f"  Maximum: {max_val:.4f}")

def main(filename):
    _, ext = os.path.splitext(filename.lower())
    
    # In case of .h5, we should check if it's specifically .cas.h5
    # We'll just do a simple check for '.cas.h5'
    if filename.lower().endswith(".cas.h5"):
        print("Using vtkFLUENTCFFReader to read Fluent case file.")
        reader = vtk.vtkFLUENTCFFReader()
        reader.SetFileName(filename)
        reader.Update()
        output_data = reader.GetOutputDataObject(0)
        
        # We may need to extract the unstructured grid from a multi-block
        ugrid = get_unstructured_grid_from_multiblock(output_data)

    elif ext == ".vtu":
        print("Using vtkXMLUnstructuredGridReader to read .vtu file.")
        reader = vtk.vtkXMLUnstructuredGridReader()
        reader.SetFileName(filename)
        reader.Update()
        # If .vtu, the output should already be a single vtkUnstructuredGrid
        ugrid = reader.GetOutput()

    else:
        print("Unrecognized file extension. Only .cas.h5 or .vtu are supported.")
        sys.exit(1)

    if ugrid is None:
        print("No vtkUnstructuredGrid found in the file.")
        return

    # 1. Aspect Ratio
    data_ar = compute_metric(ugrid, 
        lambda mq: mq.SetTetQualityMeasureToAspectRatio()
    )
    print_stats(data_ar, "Tet Aspect Ratio")

    # 2. “Equi-Angle Skew” (aka skewness for tets in many VTK builds)
    data_eas = compute_metric(ugrid, 
        lambda mq: mq.SetTetQualityMeasureToEquiangleSkew())
                      
    print_stats(data_eas, "Tet Equi-Angle Skew (Skewness)")

    # 3. Minimum Dihedral Angle
    data_dihedral = compute_metric(ugrid,
        lambda mq: mq.SetTetQualityMeasureToMinAngle()
    )
    print_stats(data_dihedral, "Tet Minimum Dihedral Angle")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <fluent_mesh_file.cas.h5 or unstructured_mesh.vtu>")
        sys.exit(1)

    main(sys.argv[1])
