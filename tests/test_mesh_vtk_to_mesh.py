# Unit tests of mesh.vtk_to_mesh()

import bpy

from utilities import *


m_mesh = import_submodule("mesh")


class TestClass:

    def test_bpy_data_meshes_update(self, PolyData_one_vertex):
        mesh_name = unique_mesh_name()
        mesh = m_mesh.vtk_to_mesh(
            PolyData_one_vertex,
            mesh_name
        )
        assert bpy.data.meshes.find(mesh_name) != -1 # "find != -1" means "found"
        

    def test_one_triangle(self, PolyData_one_triangle):
        mesh = m_mesh.vtk_to_mesh(
            PolyData_one_triangle,
            unique_mesh_name()
        )
        assert len(mesh.vertices) == 3
        assert len(mesh.edges)    == 3
        assert len(mesh.polygons) == 1
        
