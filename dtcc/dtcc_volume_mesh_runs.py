#!/usr/bin/env python

import itertools
from pathlib import Path
import pandas as pd
import csv
from pathlib import Path
import io
# Import the build_volume_mesh function from your existing module
from build_volume_mesh import build_volume_mesh
from utils import parse_mesh_report_txt


# Prepare an output directory to hold results of each run
benchmarks_dir = Path(__file__).parent / "../benchmarks/"
base_output_dir = benchmarks_dir / "dtcc_output_runs"
base_output_dir.mkdir(exist_ok=True)

mesh_report_txt_path = Path(__file__).parent / "mesh_report.txt"

dtcc_runs_results_csv_path = benchmarks_dir / "dtcc_runs_results.csv"


# Define a grid of parameter values you want to sweep over
param_grid = {
    "xmin": [319891],
    "ymin": [6399790],
    "xmax": [319891 + 200.0 , 319891 + 500.0],
    "ymax": [6399790 + 500.0, 6399790 + 600.0],
    "max_mesh_size": [10],
    "min_mesh_angle": [30],
    "smoothing_relative_tolerance": [0.005],
    "debug_step": [7]
}

cols = ['xmin', 'ymin', 'xmax', 'ymax', 'max_mesh_size', 'min_mesh_angle', 'smoothing_relative_tolerance', 'debug_step', 'num_buildings', 'vertices_2', 'cells_2', 'time_s_2', 'min_ar_2', 'median_ar_2', 'max_ar_2', 'vertices_3', 'cells_3',
        'time_s_3', 'min_ar_3', 'median_ar_3', 'max_ar_3', 'vertices_4', 'cells_4', 'time_s_4', 'min_ar_4', 'median_ar_4', 'max_ar_4', 'vertices_5', 'cells_5', 'time_s_5', 'min_ar_5', 'median_ar_5', 'max_ar_5', 'vertices_6', 'cells_6', 'time_s_6', 'min_ar_6', 'median_ar_6', 'max_ar_6', 'vertices_7', 'cells_7', 'time_s_7', 'min_ar_7', 'median_ar_7', 'max_ar_7']
run_results_df = pd.read_csv(dtcc_runs_results_csv_path, index_col=0) if dtcc_runs_results_csv_path.exists(
) else pd.DataFrame(columns=cols)




# Extract keys and the list of values for itertools.product
keys = list(param_grid.keys())
values = list(param_grid.values())

# Loop over every combination of parameters
for idx, combo in enumerate(itertools.product(*values), start=1):
    # Build the parameter dict for this run
    params = dict(zip(keys, combo))

    # Create a subdirectory for this run to avoid overwriting meshes
    run_dir = base_output_dir / f"run_{idx}"
    run_dir.mkdir(exist_ok=True)
    params["output_directory"] = run_dir

    # Execute the mesh build with these parameters
    vertices, cells, num_buildings, elapsed = build_volume_mesh(parameters=params, save_meshes=True)
    params.pop("output_directory", None)  # Remove output_directory from params

    run_report = parse_mesh_report_txt(mesh_report_txt_path)
 
    row: dict[str, float | int | None] = {k: params[k] for k in keys}
    row["num_buildings"] = num_buildings
    row.update(run_report)

    run_results_df.loc[len(run_results_df)] = row

    # Log the results
    print(f"Run {idx}: {params}")
    print(f"  → Vertices: {vertices}, Cells: {cells}, Time: {elapsed:.2f}s\n")


run_results_df.to_csv(dtcc_runs_results_csv_path, index=False)
print("All runs completed.")
