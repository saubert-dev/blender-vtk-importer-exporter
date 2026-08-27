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
@pytest.mark.parametrize(
    "name, n_vertices, n_edges, n_faces",
    [
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
        

class TestClass_UnstructuredGrid:
    
    def test_three_segments(self, pvUG_three_segments):
        vertices, edges, faces = m_mesh.get_mesh_data_from_vtk(
            pvUG_three_segments
        )
        assert len(vertices) == 3
        assert len(edges)    == 3
        assert len(faces)    == 0
        

    def test_one_triangle(self, pvUG_one_triangle):
        vertices, edges, faces = m_mesh.get_mesh_data_from_vtk(
            pvUG_one_triangle
        )
        assert len(vertices) == 3
        assert len(edges)    == 0
        assert len(faces)    == 1
        
