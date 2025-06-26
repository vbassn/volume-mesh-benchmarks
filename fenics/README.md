# FEniCS demos

## Installation

Install Conda:

    https://www.anaconda.com/docs/getting-started/miniconda/install#mac-os

Install FEniCSx:

    https://fenicsproject.org/download/

Activate Conda:

    (source ~/miniconda3/bin/activate)

    conda activate fenicsx-env

Installing FEniCSx with complex number support (didn't work):

conda create -n fenicsx-cmplx \
             -c conda-forge \
             python=3.11 \
             fenics-dolfinx \
             "petsc=*=*complex*" \
             "petsc4py=*=*complex*" \
             "slepc=*=*complex*" \
             mpich \
             pyvista
