# Unit tests of mesh.separate_polydata_homogeneous()

import pyvista as pv

import pytest

from utilities import *


m_mesh = import_submodule("mesh")


# TODO Add comments
@pytest.mark.parametrize(
    "name, cell_type, container",
    [
        pytest.param(
            "PolyData_two_vertexes", pv.CellType.VERTEX, "verts", id="vertex"
        ),
        pytest.param(
            "PolyData_two_polyvertexes", pv.CellType.POLY_VERTEX, "verts", id="polyvertex"
        ),
        pytest.param(
            "PolyData_two_lines", pv.CellType.LINE, "lines", id="line"
        ),
        pytest.param(
            "PolyData_two_polylines", pv.CellType.POLY_LINE, "lines", id="polyline"
        ),
        pytest.param(
            "PolyData_two_triangles", pv.CellType.TRIANGLE, "faces", id="triangle"
        ),
        pytest.param(
            "PolyData_two_quads", pv.CellType.QUAD, "faces", id="quad"
        ),
        pytest.param(
            "PolyData_two_polygons", pv.CellType.POLYGON, "faces", id="polygon"
        ),
        pytest.param(
            "PolyData_two_strips", pv.CellType.TRIANGLE_STRIP, "strips", id="strip"
        ),
    ],
)
def test(
    name, cell_type, container,
    request
):
    # Deep copy of the input fixture to protect it from any subsequent modification
    dataset = request.getfixturevalue(name)
    vtk_data = type(dataset)(dataset, deep=True)
    
    subsets = m_mesh.separate_polydata_homogeneous(
        vtk_data,
        m_mesh.initialize_subsets(vtk_data)
    )
    
    # Check connectivity
    subset = subsets[cell_type] # Only one type of cell expected
    for part in ("verts", "lines", "faces", "strips"):
        if part == container:
            assert getattr(subset, part) == pytest.approx(getattr(vtk_data, part))
        else:
            assert len(getattr(subset, part)) == 0
    
    # Check attributes
    assert len(subset.cell_data) == len(vtk_data.cell_data)
    for attr_name in vtk_data.cell_data.keys():
        assert subset.cell_data[attr_name] == pytest.approx(vtk_data.cell_data[attr_name])
    
    # Arbitrary in-place modifications of the cloned fixture
    # NB: Connectivity array cannot be modified in-place
    for values in vtk_data.cell_data.values():
        values -= 200
    
    # Check there was no deep copy of attributes
    for attr_name in vtk_data.cell_data.keys():
        assert subset.cell_data[attr_name] == pytest.approx(vtk_data.cell_data[attr_name])
    
