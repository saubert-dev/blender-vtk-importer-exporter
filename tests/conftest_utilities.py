# Functions shared by fixtures

import os

import numpy as np
import pyvista as pv
from vtk import vtkDataSetAttributes as vtkAttributeTypes


# Add attributes to a dataset
#   See https://docs.pyvista.org/api/core/_autosummary/pyvista.dataset
#       https://docs.pyvista.org/api/core/_autosummary/pyvista.datasetattributes
def set_attributes(dataset, fields):
    for data, suffix, scale in (
        (dataset.cell_data,  "_cell", -1),  # Cell values are set negative
        (dataset.point_data, "_point", 1)): # Point values are set positive
        
        n = data.valid_array_len
        if n > len(fields[0][2]):
            msg = (f"The size of fields must be at least equal to {n}. "
                   "See manufactured_fields() in conftest_fixtures.py.")
            raise ValueError(msg)
            
        for f_name, f_type, f_array in fields:
            f_data = f_array[:n] * scale
            match f_type:
                case vtkAttributeTypes.NORMALS:
                    data.active_normals = f_data
                case vtkAttributeTypes.TCOORDS:
                    data.active_texture_coordinates = f_data
                case _:
                    f_name += suffix
                    data.set_array(f_data, f_name)
                    match f_type:
                        case vtkAttributeTypes.SCALARS:
                            dataset.active_scalars_name = f_name
                        case vtkAttributeTypes.VECTORS:
                            dataset.active_vectors_name = f_name
                        case vtkAttributeTypes.TENSORS:
                            dataset.active_tensors_name = f_name
                        case _:
                            msg = f"Unsupported vtkDataSetAttributes type: {f_type}."
                            raise ValueError(msg)

    return
    

# Name of the directory to dump the manufactured datasets,
#   relative to the directory "tests"
dumpdir_name="data"


# Dump to disk a dataset
def dump_dataset(dataset, request):
    if request.config.getoption("dump"):
        filename = os.path.join(
            dumpdir_name,
            request.fixturename + ".vtk"
        )
        dataset.save(filename)
    return
    

# Name of the attribute indicating the cells to be deleted,
#   to manufacture the datasets with a single cell
to_remove_key = ".to_remove" # Hidden from Blender UI


# Manufacture a PolyData dataset
def make_PolyData(
    request,
    points, fields, 
    verts=None, lines=None, faces=None, strips=None
):
    dataset = pv.PolyData(
        points,
        verts=verts, lines=lines, faces=faces, strips=strips
    )
    
    set_attributes(dataset, fields)
    to_remove = np.full(dataset.cell_data.valid_array_len, True)
    to_remove[0] = False # To keep only the first cell
    dataset.cell_data[to_remove_key] = to_remove
    
    dump_dataset(dataset, request)
    
    return dataset
    

# Manufacture a dataset by removing cells
def remove_cells(full_dataset, request):
    strip_dataset = full_dataset.remove_cells(
        full_dataset.cell_data[to_remove_key].astype(np.bool_)
    )
    dump_dataset(strip_dataset, request)
    return strip_dataset
    
