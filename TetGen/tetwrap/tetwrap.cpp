#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include <cmath>
#include <iostream>
#include <map>
#include <stack>
#include <tuple>
#include <vector>

#include "tetgen.h"

namespace py = pybind11;

// Helper: convert tetgenio 'out' into (vertices, tets)
static std::pair<py::array_t<double>, py::array_t<int>>
tetgen_to_numpy(const tetgenio &out)
{
    // 1) Vertices
    const int N = out.numberofpoints;
    py::array_t<double> V({N, 3});
    {
        auto v = V.mutable_unchecked<2>();
        for (int i = 0; i < N; ++i)
        {
            v(i, 0) = out.pointlist[3 * i + 0];
            v(i, 1) = out.pointlist[3 * i + 1];
            v(i, 2) = out.pointlist[3 * i + 2];
        }
    }

    // 2) Tetrahedra
    const int K = out.numberoftetrahedra;
    const int corners = out.numberofcorners; // usually 4 for tets
    py::array_t<int> T({K, corners});
    {
        auto t = T.mutable_unchecked<2>();
        for (int i = 0; i < K; ++i)
        {
            for (int j = 0; j < corners; ++j)
            {
                t(i, j) = out.tetrahedronlist[i * corners + j];
            }
        }
    }

    return {V, T};
}

void check_volume_mesh(py::array_t<double, py::array::c_style | py::array::forcecast> vertices,
                       py::array_t<int, py::array::c_style | py::array::forcecast> facets)
{
    // Check shapes
    if (vertices.ndim() != 2 || vertices.shape(1) != 3)
        throw std::runtime_error("vertices must have shape (N,3)");
    if (facets.ndim() != 2 || facets.shape(1) != 3)
        throw std::runtime_error("facets must have shape (M,3)");

    size_t n_vertices = vertices.shape(0);
    size_t n_facets = facets.shape(0);

    std::cout << "Received " << n_vertices << " vertices." << std::endl;
    std::cout << "Received " << n_facets << " triangular facets." << std::endl;

    auto V = vertices.unchecked<2>();
    auto F = facets.unchecked<2>();

    // Print a few vertices
    std::cout << "First 3 vertices:" << std::endl;
    for (size_t i = 0; i < std::min<size_t>(3, n_vertices); ++i)
    {
        std::cout << "  (" << V(i, 0) << ", " << V(i, 1) << ", " << V(i, 2) << ")" << std::endl;
    }

    // Print first few facets
    std::cout << "First 3 facets:" << std::endl;
    for (size_t i = 0; i < std::min<size_t>(3, n_facets); ++i)
    {
        std::cout << "  [" << F(i, 0) << ", " << F(i, 1) << ", " << F(i, 2) << "]" << std::endl;
    }

    tetgenio in, out;

    in.initialize();
    in.deinitialize();
}

static std::vector<char> make_mutable_switches(py::object obj)
{
    // Accept str or bytes
    if (py::isinstance<py::str>(obj) || py::isinstance<py::bytes>(obj))
    {
        std::string s = py::cast<std::string>(obj);
        std::vector<char> buf(s.begin(), s.end());
        buf.push_back('\0');
        return buf;
    }
    // Accept a 1D NumPy array of dtype=|S1 or int8/uint8 (char-ish)
    if (py::isinstance<py::array>(obj))
    {
        auto arr = py::cast<py::array>(obj);
        if (arr.ndim() != 1)
            throw std::runtime_error("tetgen_switches array must be 1D");
        // Force a contiguous view of bytes
        py::array_t<uint8_t, py::array::c_style | py::array::forcecast> a = arr;
        auto r = a.unchecked<1>();
        std::vector<char> buf(r.shape(0) + 1);
        for (ssize_t i = 0; i < r.shape(0); ++i)
            buf[i] = static_cast<char>(r(i));
        buf.back() = '\0';
        return buf;
    }
    throw std::runtime_error("tetgen_switches must be str, bytes, or 1D char/byte array");
}

std::pair<py::array_t<double>, py::array_t<int>>
build_volume_mesh(py::array_t<double, py::array::c_style | py::array::forcecast> vertices,
                  py::array_t<int, py::array::c_style | py::array::forcecast> mesh_facets,
                  const std::vector<std::vector<int>> &boundary_facets,
                  py::object tetgen_switches)
{
    if (vertices.ndim() != 2 || vertices.shape(1) != 3)
        throw std::runtime_error("vertices must have shape (N,3)");
    if (mesh_facets.ndim() != 2 || mesh_facets.shape(1) != 3)
        throw std::runtime_error("mesh facets must have shape (M,3)");

    if (boundary_facets.size() < 5)
    {
        throw std::runtime_error(
            "boundary_facets must contain at least 5 polygons "
            "(top, east, south, north, west).");
    }

    // each facet must have at least 3 indices
    for (size_t i = 0; i < boundary_facets.size(); ++i)
    {
        if (boundary_facets[i].size() < 3)
        {
            throw std::runtime_error("facet " + std::to_string(i) +
                                     " has fewer than 3 vertices.");
        }
    }

    auto V = vertices.unchecked<2>();    // (N,3)
    auto F = mesh_facets.unchecked<2>(); // (M,3)
    const int N = static_cast<int>(V.shape(0));
    const int M = static_cast<int>(F.shape(0));

    // index-range check for mesh_facets
    for (int i = 0; i < M; ++i)
    {
        for (int k = 0; k < 3; ++k)
        {
            int vid = F(i, k);
            if (vid < 0 || vid >= N)
                throw std::runtime_error("mesh_facets index out of range at row " + std::to_string(i));
        }
    }
    // index-range check for boundary_facets
    for (size_t bi = 0; bi < boundary_facets.size(); ++bi)
    {
        for (int vid : boundary_facets[bi])
        {
            if (vid < 0 || vid >= N)
                throw std::runtime_error("boundary_facets index out of range at polygon " + std::to_string(bi));
        }
    }

    tetgenio in, out;
    in.initialize();
    out.initialize();

    in.firstnumber = 0;
    in.numberofpoints = static_cast<int>(N);
    in.pointlist = new REAL[in.numberofpoints * 3];

    std::cout << "Number of input points: " << N << std::endl;
    std::cout << "Number of tetgen points: " << in.numberofpoints << std::endl;

    REAL *guard = &in.pointlist[3 * N];
    guard[0] = guard[1] = guard[2] = guard[3] = 1234567.0;

    for (int i = 0; i < N; ++i)
    {
        // Bounds-safe: last index written is 3*(N-1)+2
        in.pointlist[3 * i + 0] = static_cast<REAL>(V(i, 0));
        in.pointlist[3 * i + 1] = static_cast<REAL>(V(i, 1));
        in.pointlist[3 * i + 2] = static_cast<REAL>(V(i, 2));
    }

    // Verify no overwrite happened
    if (guard[0] != 1234567.0 || guard[1] != 1234567.0 ||
        guard[2] != 1234567.0 || guard[3] != 1234567.0)
    {
        throw std::runtime_error("pointlist overflow detected");
    }
    // facets = mesh triangles (M) + boundary polygons (B)
    const int B = static_cast<int>(boundary_facets.size());
    const int T = M + B;

    if (N <= 0)
        throw std::runtime_error("vertices: N <= 0");
    if (M < 0)
        throw std::runtime_error("mesh_facets: M < 0");
    if (B < 5)
        throw std::runtime_error("boundary_facets must have >= 5");
    if (T <= 0)
        throw std::runtime_error("total facets (M+B) <= 0");

    std::cout << "N=" << N << " M=" << M << " B=" << B << " T=" << T << std::endl;
    in.numberoffacets = T;

    in.facetlist = new tetgenio::facet[in.numberoffacets]();

    // mesh triangles in [0..M-1]
    for (int fi = 0; fi < M; ++fi)
    {
        tetgenio::facet &fac = in.facetlist[fi];
        fac.numberofholes = 0;
        fac.holelist = nullptr;
        fac.numberofpolygons = 1;
        fac.polygonlist = new tetgenio::polygon[1];
        tetgenio::polygon &poly = fac.polygonlist[0];
        poly.numberofvertices = 3;
        poly.vertexlist = new int[3];
        poly.vertexlist[0] = F(fi, 0);
        poly.vertexlist[1] = F(fi, 1);
        poly.vertexlist[2] = F(fi, 2);
    }
    // boundary polys in [M..M+B-1]
    for (int bi = 0; bi < B; ++bi)
    {
        tetgenio::facet &fac = in.facetlist[M + bi];
        fac.numberofholes = 0;
        fac.holelist = nullptr;
        fac.numberofpolygons = 1;
        fac.polygonlist = new tetgenio::polygon[1];
        tetgenio::polygon &poly = fac.polygonlist[0];
        const auto &loop = boundary_facets[bi];
        poly.numberofvertices = (int)loop.size();
        poly.vertexlist = new int[poly.numberofvertices];
        for (int j = 0; j < poly.numberofvertices; ++j)
            poly.vertexlist[j] = loop[j];
    }

    std::vector<char> sw = make_mutable_switches(tetgen_switches);

    tetrahedralize(sw.data(), &in, &out);

    // Convert TetGen output to NumPy
    std::cout << "Converting TetGen output to NumPy\n";
    auto result = tetgen_to_numpy(out);

    // in.deinitialize();
    // out.deinitialize();
    std::cout << "Tetwrap Meshing Done.\n";
    return result;
}

PYBIND11_MODULE(_tetwrap, m)
{
    m.def("check_volume_mesh", &check_volume_mesh,
          py::arg("facets"), py::arg("vertices"),
          R"pbdoc(
              Check volume mesh (Sanity check).

              facets: List[List[int]] – list of facets, each facet is a list of vertex indices
              vertices: List[float] – flat list of vertex coordinates [x0,y0,z0,x1,y1,z1,...]
          )pbdoc");

    m.def("build_volume_mesh", &build_volume_mesh,
          py::arg("vertices"),
          py::arg("mesh_facets"),
          py::arg("boundary_facets"),
          py::arg("tetgen_switches"),
          R"pbdoc(
              Build volume mesh (demo function).

              mesh_facets: List[List[int]] – list of facets, each facet is a list of vertex indices
              boundary_facets: List[List[int]] – list of facets, each facet is a list of vertex indices for boundary surfaces.
              vertices: List[float] – flat list of vertex coordinates [x0,y0,z0,x1,y1,z1,...]
          )pbdoc");
}