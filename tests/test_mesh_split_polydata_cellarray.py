# Unit tests of mesh.split_polydata_cellarray()

import numpy as np

import pytest

from utilities import *


m_mesh = import_submodule("mesh")


# A cellarray is a padded VTK connectivity array, structured as:
#   [n0, p0_0, p0_1, ..., p0_n, n1, p1_0, p1_1, ..., p1_n, ...]
#   where nX is the number of points in cell X and pX_Y is point Y in cell X
# See  https://docs.pyvista.org/api/core/_autosummary/pyvista.polydata.faces


def test_one_cell():
    cellarray = np.hstack([
        [1, 11], # [n0, p0_0]
    ])
    expected_cellarray = cellarray # Identical because there is a single type of cell
    expected_indexset  = np.array([0])
    
    n_poly = 1 # No cell with variable size
    parts = m_mesh.split_polydata_cellarray(cellarray, n_poly)
    
    assert len(parts) == 1 # Single type of cell
    n_points = 1 # Cells made of a single point
    assert parts[n_points][0] == pytest.approx(expected_cellarray)
    assert parts[n_points][1] == pytest.approx(expected_indexset)
   

def test_two_cells_same_size():
    cellarray = np.hstack([
        [1, 11], # [n0, p0_0]
        [1, 12], # [n1, p1_0]
    ])
    expected_cellarray = cellarray # Identical because there is a single type of cell
    expected_indexset  = np.array([0, 1])
    
    n_poly = 1 # No cell with variable size
    parts = m_mesh.split_polydata_cellarray(cellarray, n_poly)
    
    assert len(parts) == 1 # Single type of cell
    n_points = 1 # Cells made of a single point
    assert parts[n_points][0] == pytest.approx(expected_cellarray)
    assert parts[n_points][1] == pytest.approx(expected_indexset)
    

def test_two_cells_any_size():
    cellarray = np.hstack([
        [2, 21, 22], # [n0, p0_0, p0_1]
        [1, 11],     # [n1, p1_0]
    ])
    expected_cellarray = [
        None,                  # Placeholder
        np.array([1, 11]),     # Cells made of a single point
        np.array([2, 21, 22]), # Cells made of two points
    ]
    expected_indexset = [
        None,          # Placeholder
        np.array([1]), # Rank of cells made of a single point
        np.array([0]), # Rank of cells made of two points
    ]
    
    n_poly = 2 # No cell with variable size
    parts = m_mesh.split_polydata_cellarray(cellarray, n_poly)
    
    assert len(parts) == 2 # Two types of cell
    for n_points in range(n_poly+1):
        if expected_cellarray[n_points] is not None:
            assert parts[n_points][0] == pytest.approx(expected_cellarray[n_points])
            assert parts[n_points][1] == pytest.approx(expected_indexset[n_points])
    

def test_one_polycell():
    cellarray = np.hstack([
        [3, 11, 12, 13], # [n0, p0_0, p0_1, p0_2]
    ])
    expected_cellarray = cellarray # Identical because there is a single type of cell
    expected_indexset  = np.array([0])
    
    n_poly = 2 # It is assumed that the number of points of POLY_TYPE >= 2
    parts = m_mesh.split_polydata_cellarray(cellarray, n_poly)
    
    assert len(parts) == 1 # Single type of cell
    n_points = n_poly # Cells made of n_poly points or more
    assert parts[n_points][0] == pytest.approx(expected_cellarray)
    assert parts[n_points][1] == pytest.approx(expected_indexset)
   

def test_two_polycells():
    cellarray = np.hstack([
        [3, 11, 12, 13], # [n0, p0_0, p0_1, p0_2]
        [2, 21, 22],     # [n1, p1_0, p1_1]
    ])
    expected_cellarray = cellarray # Identical because there is a single type of cell
    expected_indexset  = np.array([0, 1])
    
    n_poly = 2 # It is assumed that the number of points of POLY_TYPE >= 2
    parts = m_mesh.split_polydata_cellarray(cellarray, n_poly)
    
    assert len(parts) == 1 # Single type of cell
    n_points = n_poly # Cells made of n_poly points or more
    assert parts[n_points][0] == pytest.approx(expected_cellarray)
    assert parts[n_points][1] == pytest.approx(expected_indexset)
   

def test_chunks_mid():
    cellarray = np.hstack([
        [2, 11, 12],         # [n0, p0_0, p0_1]
        [1, 21],             # [n1, p1_0] - First chunk
        [3, 31, 32, 33],     # [n2, p2_0, p2_1, p2_2]
        [1, 41],             # [n3, p3_0] - Second chunk
        [4, 51, 52, 53, 54], # [n4, p4_0, p4_1, p4_2, p4_3]
    ])
    expected_cellarray = [
        None,                          # Placeholder
        np.array([1, 21, 1, 41]),      # Cells made of a single point
        np.array([2, 11, 12]),         # Cells made of two points
        np.array([3, 31, 32, 33]),     # Cells made of three points
        np.array([4, 51, 52, 53, 54]), # Cells made of four points
    ]
    expected_indexset = [
        None,             # Placeholder
        np.array([1, 3]), # Rank of cells made of a single point
        np.array([0]),    # Rank of cells made of two points
        np.array([2]),    # Rank of cells made of three points
        np.array([4]),    # Rank of cells made of four points
    ]
    
    n_poly = 4 # No cell with variable size
    parts = m_mesh.split_polydata_cellarray(cellarray, n_poly)
    
    assert len(parts) == 4 # Four types of cell
    for n_points in range(n_poly+1):
        if expected_cellarray[n_points] is not None:
            assert parts[n_points][0] == pytest.approx(expected_cellarray[n_points])
            assert parts[n_points][1] == pytest.approx(expected_indexset[n_points])
    

def test_chunks_beg():
    cellarray = np.hstack([
        [1, 11],         # [n0, p0_0] - First chunk
        [2, 21, 22],     # [n1, p1_0, p1_1]
        [1, 31],         # [n2, p2_0] - Second chunk
        [3, 41, 42, 43], # [n3, p3_0, p3_1, p3_2]
    ])
    expected_cellarray = [
        None,                      # Placeholder
        np.array([1, 11, 1, 31]),  # Cells made of a single point
        np.array([2, 21, 22]),     # Cells made of two points
        np.array([3, 41, 42, 43]), # Cells made of three points
    ]
    expected_indexset = [
        None,             # Placeholder
        np.array([0, 2]), # Rank of cells made of a single point
        np.array([1]),    # Rank of cells made of two points
        np.array([3]),    # Rank of cells made of three points
    ]
    
    n_poly = 3 # No cell with variable size
    parts = m_mesh.split_polydata_cellarray(cellarray, n_poly)
    
    assert len(parts) == 3 # Three types of cell
    for n_points in range(n_poly+1):
        if expected_cellarray[n_points] is not None:
            assert parts[n_points][0] == pytest.approx(expected_cellarray[n_points])
            assert parts[n_points][1] == pytest.approx(expected_indexset[n_points])
    

def test_chunks_end():
    cellarray = np.hstack([
        [2, 11, 12],     # [n0, p0_0, p0_1]
        [1, 21],         # [n1, p1_0] - First chunk
        [3, 31, 32, 33], # [n2, p2_0, p2_1, p2_2]
        [1, 41],         # [n3, p3_0] - Second chunk
    ])
    expected_cellarray = [
        None,                      # Placeholder
        np.array([1, 21, 1, 41]),  # Cells made of a single point
        np.array([2, 11, 12]),     # Cells made of two points
        np.array([3, 31, 32, 33]), # Cells made of three points
    ]
    expected_indexset = [
        None,             # Placeholder
        np.array([1, 3]), # Rank of cells made of a single point
        np.array([0]),    # Rank of cells made of two points
        np.array([2]),    # Rank of cells made of three points
    ]
    
    n_poly = 3 # No cell with variable size
    parts = m_mesh.split_polydata_cellarray(cellarray, n_poly)
    
    assert len(parts) == 3 # Three types of cell
    for n_points in range(n_poly+1):
        if expected_cellarray[n_points] is not None:
            assert parts[n_points][0] == pytest.approx(expected_cellarray[n_points])
            assert parts[n_points][1] == pytest.approx(expected_indexset[n_points])
    

def test_mixed_cells():
    cellarray = np.hstack([
        [4, 11, 12, 13, 14], # [n0, p0_0, p0_1, p0_2, p0_3]
        [1, 21],             # [n1, p1_0]
        [3, 31, 32, 33],     # [n2, p2_0, p2_1, p2_2]
        [1, 41],             # [n3, p3_0]
    ])
    expected_cellarray = [
        None,                                         # Placeholder
        np.array([1, 21, 1, 41]),                     # Cells made of a single point
        None,                                         # Placeholder
        np.array([4, 11, 12, 13, 14, 3, 31, 32, 33]), # Cells made of three points or more
    ]
    expected_indexset = [
        None,             # Placeholder
        np.array([1, 3]), # Rank of cells made of a single point
        None,             # Placeholder
        np.array([0, 2]), # Rank of cells made of three points
    ]
    
    n_poly = 3 # It is assumed that the number of points of POLY_TYPE >= 3
    parts = m_mesh.split_polydata_cellarray(cellarray, n_poly)
    
    assert len(parts) == 2 # Two types of cell
    for n_points in range(n_poly+1):
        if expected_cellarray[n_points] is not None:
            assert parts[n_points][0] == pytest.approx(expected_cellarray[n_points])
            assert parts[n_points][1] == pytest.approx(expected_indexset[n_points])
    
