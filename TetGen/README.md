# TetGen Build Instructions

This directory provides two build targets:

1. **TetGen CLI** – the standalone command-line tool  
2. **TetGen Python Binding** – a shared library (`.so`) that exposes TetGen through `pybind11`

---

## 1. Build TetGen CLI

```bash
cd tetgen
mkdir build && cd build
cmake ..
make -j
```

This produces the `tetgen` executable inside `tetgen/build/`.

---

## 2. Build TetGen Python Binding (tetwrap)

```bash
cd tetwrap
mkdir build && cd build
cmake ..
make -j
```

This produces a dynamic library (e.g. _tetwrap.so) that can be imported from Python.

---

After building the Python binding, you can run the demo script:

```bash
python tetgen_volume_mesh_demo.py
```