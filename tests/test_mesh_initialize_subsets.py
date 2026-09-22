# Unit tests of mesh.initialize_subsets()

import pyvista as pv

import pytest

from utilities import *


m_mesh = import_submodule("mesh")


# TODO Add comments
@pytest.mark.parametrize(
    "name, cell_type",
    [
        pytest.param(
            "PolyData_two_vertexes", pv.CellType.VERTEX, id="vertex"
        ),
        pytest.param(
            "PolyData_two_polyvertexes", pv.CellType.POLY_VERTEX, id="polyvertex"
        ),
        pytest.param(
            "PolyData_two_lines", pv.CellType.LINE, id="line"
        ),
        pytest.param(
            "PolyData_two_polylines", pv.CellType.POLY_LINE, id="polyline"
        ),
        pytest.param(
            "PolyData_two_triangles", pv.CellType.TRIANGLE, id="triangle"
        ),
        pytest.param(
            "PolyData_two_quads", pv.CellType.QUAD, id="quad"
        ),
        pytest.param(
            "PolyData_two_polygons", pv.CellType.POLYGON, id="polygon"
        ),
        pytest.param(
            "PolyData_two_strips", pv.CellType.TRIANGLE_STRIP, id="strip"
        ),
        pytest.param(
            "UnstructuredGrid_two_pixels", pv.CellType.PIXEL, id="pixel"
        ),
    ],
)
# Parametrization of the type of PyVista DataSet
@pytest.mark.parametrize("dataset_type", ["PolyData", "UnstructuredGrid"])
def test_homogeneous(
    name, cell_type, dataset_type,
    request
):
    dataset = request.getfixturevalue(name)
    match dataset_type:
        case "PolyData":
            if isinstance(dataset, pv.PolyData):
                vtk_data = dataset
            else:
                pytest.skip("Non-testable type(s) of cell")
        case "UnstructuredGrid":
            vtk_data = dataset.cast_to_unstructured_grid()
        case _:
            msg = f"Unsupported PyVista DataSet type: {dataset_type}."
            raise ValueError(msg)
    
    subsets = m_mesh.initialize_subsets(vtk_data)
    
    assert len(subsets) == 2 # Only a single cell type and points are expected
    assert cell_type in subsets
    assert pv.CellType.EMPTY_CELL in subsets # Holding the points
    
    for subset in subsets.values():
        assert subset.n_cells == 0 # Only points are initialized
        assert len(subset.cell_data) == 0
        assert subset.points == pytest.approx(vtk_data.points)
        
    subset = subsets[cell_type]
    assert len(subset.point_data) == 0
    
    subset = subsets[pv.CellType.EMPTY_CELL]
    assert len(subset.point_data) == len(vtk_data.point_data)
    for attr_name in vtk_data.point_data.keys():
        assert subset.point_data[attr_name] == pytest.approx(vtk_data.point_data[attr_name])
    

# TODO Add comments
@pytest.mark.parametrize(
    "name",
    [
        pytest.param(
            "UnstructuredGrid_two_tetrahedrons", id="tetrahedron"
        ),
    ],
)
def test_not_imported(
    name,
    request
):
    vtk_data = request.getfixturevalue(name)
    
    subsets = m_mesh.initialize_subsets(vtk_data)
    
    assert len(subsets) == 1 # Only points are expected
    assert pv.CellType.EMPTY_CELL in subsets # Holding the points
    
    subset = subsets[pv.CellType.EMPTY_CELL]
    assert subset.n_cells == 0 # Only points are initialized
    assert len(subset.cell_data) == 0
    assert subset.points == pytest.approx(vtk_data.points)
    assert len(subset.point_data) == len(vtk_data.point_data)
    for attr_name in vtk_data.point_data.keys():
        assert subset.point_data[attr_name] == pytest.approx(vtk_data.point_data[attr_name])
    

# TODO Add comments
@pytest.mark.parametrize(
    "name, n_cell_types",
    [
        pytest.param(
            "PolyData_two_merged", 8, id="PolyData"
        ),
        pytest.param(
            "UnstructuredGrid_two_shuffled", 9, id="UnstructuredGrid"
        ),
    ],
)
def test_heterogeneous(
    name, n_cell_types,
    request
):
    vtk_data = request.getfixturevalue(name)
    
    subsets = m_mesh.initialize_subsets(vtk_data)
    
    cell_types = list(subsets)
    cell_types.remove(pv.CellType.EMPTY_CELL)
    assert len(cell_types) == n_cell_types
    
    cell_type_is_used = {cell_type: False for cell_type in cell_types}
    for cell in vtk_data.cell:
        cell_type_is_used[cell.type] = True
    assert all(cell_type_is_used.values())
    
    for cell_type, subset in subsets.items():
        assert subset.n_cells == 0 # Only points are initialized
        assert len(subset.cell_data) == 0
        assert subset.points == pytest.approx(vtk_data.points)
        if cell_type == pv.CellType.EMPTY_CELL:
            assert len(subset.point_data) == len(vtk_data.point_data)
            for attr_name in vtk_data.point_data.keys():
                assert subset.point_data[attr_name] == pytest.approx(vtk_data.point_data[attr_name])
        else:
            assert len(subset.point_data) == 0
    

# TODO Add comments
@pytest.mark.parametrize(
    "name",
    [
        pytest.param(
            "PolyData_two_merged", id="PolyData"
        ),
        pytest.param(
            "UnstructuredGrid_two_shuffled", id="UnstructuredGrid"
        ),
    ],
)
def test_no_deep_copy(
    name,
    request
):
    # Deep copy of the input fixture to protect it from any subsequent modification
    dataset = request.getfixturevalue(name)
    vtk_data = type(dataset)(dataset, deep=True)
    
    subsets = m_mesh.initialize_subsets(vtk_data)
    
    # Arbitrary in-place modifications of the cloned fixture
    vtk_data.points += 100
    for values in vtk_data.point_data.values():
        values += 200
    
    for subset in subsets.values():
        assert subset.points == pytest.approx(vtk_data.points)
    
    subset = subsets[pv.CellType.EMPTY_CELL]
    for attr_name in vtk_data.point_data.keys():
        assert subset.point_data[attr_name] == pytest.approx(vtk_data.point_data[attr_name])
    
