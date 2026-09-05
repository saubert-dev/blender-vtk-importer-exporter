# Unit tests of mesh.get_mesh_data_from_vtk()

import pytest

from utilities import *


m_mesh = import_submodule("mesh")


# Parametrization of the characteristics of a test case
#   name:       Name of the fixture manufacturing the dataset
#               see conftest_fixtures.py
#   n_vertices: Expected number of points
#   n_edges:    Expected number of edges
#   n_faces:    Expected number of faces
#               Can be different according to the type of PyVista DataSet
#               Set to -1 for non-testable types of cell (e.g. PIXEL not supported by PolyData)
@pytest.mark.parametrize(
    "name, n_vertices, n_edges, n_faces",
    [
        pytest.param(
            "PolyData_one_vertex", 1, 0, 0, id="one_vertex"
        ),
        pytest.param(
            "PolyData_one_polyvertex", 2, 0, 0, id="one_polyvertex"
        ),
        pytest.param(
            "PolyData_one_line", 2, 1, 0, id="one_line"
        ),
        pytest.param(
            "PolyData_one_polyline", 3, 2, 0, id="one_polyline"
        ),
        pytest.param(
            "PolyData_one_triangle", 3, 0, 1, id="one_triangle"
        ),
        pytest.param(
            "PolyData_one_quad", 4, 0, {"PolyData": 2, "UnstructuredGrid": 1}, id="one_quad"
        ),
        pytest.param(
            "PolyData_one_polygon", 6, 0, {"PolyData": 4, "UnstructuredGrid": 1}, id="one_polygon"
        ),
        pytest.param(
            "PolyData_one_strip", 4, 0, 2, id="one_strip"
        ),
        pytest.param(
            "PolyData_one_merged", 25, 3, {"PolyData": 9, "UnstructuredGrid": 5}, id="one_merged"
        ),
        pytest.param(
            "UnstructuredGrid_one_pixel", 4, 0, {"PolyData": -1, "UnstructuredGrid": 1}, id="one_pixel"
        ),
        pytest.param(
            "UnstructuredGrid_one_tetrahedron", 4, 0, {"PolyData": -1, "UnstructuredGrid": 0}, id="one_tetrahedron"
        ),
        pytest.param(
            "UnstructuredGrid_one_shuffled", 37, 7, {"PolyData": -1, "UnstructuredGrid": 10}, id="one_shuffled"
        ),
        pytest.param(
            "PolyData_two_vertexes", 2, 0, 0, id="two_vertexes"
        ),
        pytest.param(
            "PolyData_two_polyvertexes", 5, 0, 0, id="two_polyvertexes"
        ),
        pytest.param(
            "PolyData_two_lines", 3, 2, 0, id="two_lines"
        ),
        pytest.param(
            "PolyData_two_polylines", 6, 5, 0, id="two_polylines"
        ),
        pytest.param(
            "PolyData_two_triangles", 4, 0, 2, id="two_triangles"
        ),
        pytest.param(
            "PolyData_two_quads", 6, 0, {"PolyData": 4, "UnstructuredGrid": 2}, id="two_quads"
        ),
        pytest.param(
            "PolyData_two_polygons", 9, 0, {"PolyData": 7, "UnstructuredGrid": 2}, id="two_polygons"
        ),
        pytest.param(
            "PolyData_two_strips", 7, 0, 5, id="two_strips"
        ),
        pytest.param(
            "PolyData_two_merged", 31, 7, {"PolyData": 18, "UnstructuredGrid": 8}, id="two_merged"
        ),
        pytest.param(
            "UnstructuredGrid_two_pixels", 6, 0, {"PolyData": -1, "UnstructuredGrid": 2}, id="two_pixels"
        ),
        pytest.param(
            "UnstructuredGrid_two_tetrahedrons", 6, 0, {"PolyData": -1, "UnstructuredGrid": 0}, id="two_tetrahedrons"
        ),
        pytest.param(
            "UnstructuredGrid_two_shuffled", 37, 7, {"PolyData": -1, "UnstructuredGrid": 10}, id="two_shuffled"
        ),
    ],
)

# Parametrization of the type of PyVista DataSet
@pytest.mark.parametrize("dataset_type", ["PolyData", "UnstructuredGrid"])

class TestClass:
    
    def test_counts(
        self,
        dataset_type,
        name, n_vertices, n_edges, n_faces,
        request
    ):
        if isinstance(n_faces, dict):
            if n_faces[dataset_type] < 0:
                pytest.skip("Non-testable type(s) of cell")
                
        dataset = request.getfixturevalue(name)
        match dataset_type:
            case "PolyData":
                vtk_data = dataset
            case "UnstructuredGrid":
                vtk_data = dataset.cast_to_unstructured_grid()
            case _:
                msg = f"Unsupported PyVista DataSet type: {dataset_type}."
                raise ValueError(msg)
                
        vertices, edges, faces = m_mesh.get_mesh_data_from_vtk(vtk_data)
        
        assert len(vertices) == n_vertices
        assert len(edges)    == n_edges
        if isinstance(n_faces, dict):
            assert len(faces) == n_faces[dataset_type]
        else:
            assert len(faces) == n_faces
        
