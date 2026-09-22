import warnings
from typing import Union

import bpy
import numpy as np
import pyvista as pv

from .attributes import initialize_material_attributes, update_attributes_from_vtk
from .material_panel import update_attributes_enum
from .nodes import convert_mesh_to_pointcloud, create_attribute_material_nodes

VTK_data = Union[pv.PolyData, pv.UnstructuredGrid]


def initialize_subsets(vtk_data: VTK_data):
    # Initialization of one empty dataset per cell type used
    subsets = dict()
    supported_cell_types = [
        pv.CellType.EMPTY_CELL,
        pv.CellType.VERTEX,
        pv.CellType.POLY_VERTEX,
        pv.CellType.LINE,
        pv.CellType.POLY_LINE,
        pv.CellType.TRIANGLE,
        pv.CellType.QUAD,
        pv.CellType.POLYGON,
        pv.CellType.TRIANGLE_STRIP,
        pv.CellType.PIXEL,
    ]
    cell_types = vtk_data.distinct_cell_types
    cell_types.add(pv.CellType.EMPTY_CELL) # To hold only the points
    for cell_type in cell_types:
        if cell_type in supported_cell_types:
            subsets[cell_type] = pv.PolyData()
            subsets[cell_type].points = vtk_data.points # No VERTEX cells created
        else:
            msg = f"<{cell_type.name}> type cells are not imported."
            warnings.warn(msg, stacklevel=2)
            
    # Setup of the dataset holding the points position and attributes
    subset = subsets[pv.CellType.EMPTY_CELL]
    for attr_name, values in vtk_data.point_data.items():
        subset.point_data.set_array(values, attr_name, deep_copy=False)
    
    return subsets
    

def separate_polydata_homogeneous(vtk_data: pv.PolyData, subsets):
    cell_types = list(subsets)
    cell_types.remove(pv.CellType.EMPTY_CELL) # To filter out the points
    cell_type = cell_types[0]
    subset = subsets[cell_type]
        
    match cell_type: # NB: Following assignments are deep copies
        case ( pv.CellType.VERTEX
             | pv.CellType.POLY_VERTEX ):
            subset.verts = vtk_data.verts
        case ( pv.CellType.LINE
             | pv.CellType.POLY_LINE ):
            subset.lines = vtk_data.lines
        case ( pv.CellType.TRIANGLE
             | pv.CellType.QUAD
             | pv.CellType.POLYGON ):
            subset.faces = vtk_data.faces
        case ( pv.CellType.TRIANGLE_STRIP ):
            subset.strips = vtk_data.strips
            
    for attr_name, values in vtk_data.cell_data.items():
        subset.cell_data.set_array(values, attr_name, deep_copy=False)
    
    return subsets
    

def split_polydata_cellarray(cellarray, n_poly):
    cellarrays = [
        list() for n_points in range(n_poly+1)
    ] # Array of empty lists indexed by the capped number of points in a cell
    
    i_last = 0 # Index of the number of points of current cell
    c_last = 0 # Rank of current cell
    
    i_first = i_last # Index of the number of points of first cell
    c_first = c_last # Rank of first cell
    n_first = min(cellarray[i_first], n_poly) # Capped number of points of first cell

    while i_last < len(cellarray):
        n_last = min(cellarray[i_last], n_poly) # Capped number of points of current cell
        if n_first != n_last: # Types of first and current cell are different
            cellarrays[n_first].append((
                cellarray[i_first:i_last], # Connectivity
                np.arange(c_first, c_last) # Cells index
            )) # Store the current chunk of cells with same type
            i_first = i_last # Starting index of next chunk
            c_first = c_last
            n_first = n_last
        i_last += cellarray[i_last] + 1 # Jump to next cell
        c_last += 1
        
    cellarrays[n_first].append((
        cellarray[i_first:i_last], # Connectivity
        np.arange(c_first, c_last) # Cells index
    )) # Store the last chunk of cells with same type
    
    parts = dict() # Dictionary keyed by the capped number of points in a cell
    for n_points, part in enumerate(cellarrays):
        if len(part) > 0:
            # Join the chunks of cells
            parts[n_points] = (
                np.concatenate([chunk[0] for chunk in part]), # Connectivity
                np.concatenate([chunk[1] for chunk in part])  # IndexSet
            )
    
    return parts
    

def separate_polydata_heterogeneous(vtk_data: pv.PolyData, subsets):
    print("More than one type of cell")
    if vtk_data.n_verts > 0:
        print("il y a des points")
        print(vtk_data.verts)
        n_poly = 2 # Number of points of POLY_VERTEX >= 2
        parts = split_polydata_cellarray(vtk_data.verts, n_poly)
        if pv.CellType.VERTEX in subsets:
            subsets[pv.CellType.VERTEX].verts = parts[1][0]
        if pv.CellType.POLY_VERTEX in subsets:
            subsets[pv.CellType.POLY_VERTEX].verts = parts[n_poly][0]
        
    if vtk_data.n_lines > 0:
        print("il y a des lines")
        print(vtk_data.lines)
        n_poly = 3 # Number of points of POLY_LINE >= 3
        parts = split_polydata_cellarray(vtk_data.lines, n_poly)
        if pv.CellType.LINE in subsets:
            subsets[pv.CellType.LINE].lines = parts[2][0]
        if pv.CellType.POLY_LINE in subsets:
            subsets[pv.CellType.POLY_LINE].lines = parts[n_poly][0]
        
    if vtk_data.n_faces > 0:
        print("il y a des faces")
        print(vtk_data.faces)
        n_poly = 5 # Number of points of POLYGON >= 5
        parts = split_polydata_cellarray(vtk_data.faces, n_poly)
        if pv.CellType.TRIANGLE in subsets:
            subsets[pv.CellType.TRIANGLE].faces = parts[3][0]
        if pv.CellType.QUAD in subsets:
            subsets[pv.CellType.QUAD].faces = parts[4][0]
        if pv.CellType.POLYGON in subsets:
            subsets[pv.CellType.POLYGON].faces = parts[n_poly][0]
        
    if vtk_data.n_strips > 0:
        print("il y a des strips")
        print(vtk_data.strips)
        subsets[pv.CellType.TRIANGLE_STRIP].strips = vtk_data.strips
    
    return subsets
    

def separate_vtk_by_cell_types(vtk_data: VTK_data):
    subsets = initialize_subsets(vtk_data)
    if isinstance(vtk_data, pv.PolyData):
        if len(subsets) == 2:
            return separate_polydata_homogeneous(vtk_data, subsets)
        else:
            return separate_polydata_heterogeneous(vtk_data, subsets)
    # TODO separate_unstructuredgrid
    # TODO else unsupported dataset type
    return subsets
    

def separate_vtk_by_domains(vtk_data: VTK_data):
    datasets = {
        "POINT": pv.PointSet(),
        "EDGE":  pv.PolyData(),
        "FACE":  pv.UnstructuredGrid(),
    }
    return datasets
    

def update_mesh(scene):
    if (
        "vtk_files" not in bpy.context.scene
        and "vtk_directory" not in bpy.context.scene
    ):
        return

    frame = scene.frame_current

    files = bpy.context.scene["vtk_files"]
    directory = bpy.context.scene["vtk_directory"]
    frame_sep = bpy.context.scene["frame_sep"]

    for file in files:
        mesh_name = file[0].split(".")[0].split(frame_sep)[0]

        if len(file) > 1:
            polydata = pv.read(f"{directory}/{file[frame]}")
            mesh: bpy.types.Mesh = bpy.data.meshes[mesh_name]

            if (polydata.n_points, polydata.n_cells) == (
                len(mesh.vertices),
                len(mesh.polygons),
            ):
                update_attributes_from_vtk(polydata, mesh_name)
            else:  # mesh has changed
                update_mesh_from_vtk(mesh, polydata, update_attributes=True)


def get_mesh_data_from_vtk(vtk_data: VTK_data):
    edges = []
    faces = []

    if isinstance(vtk_data, pv.PolyData):
        if not vtk_data.is_all_triangles:
            vtk_data = vtk_data.triangulate()
        faces = np.reshape(vtk_data.faces, (vtk_data.n_cells, 4))[:, 1:]

    if isinstance(vtk_data, pv.UnstructuredGrid):
        for cell_type, cells in vtk_data.cells_dict.items():
            if cell_type == pv.CellType.LINE.value:
                edges += cells.tolist()
            elif pv.CellType.TRIANGLE.value <= cell_type <= pv.CellType.QUAD.value:
                faces += cells.tolist()
            else:
                warnings.warn(f"Unsupported cell type: {cell_type} yet.", stacklevel=2)

    return vtk_data.points, edges, faces


def update_mesh_from_vtk(
    mesh: bpy.types.Mesh, vtk_data: VTK_data, update_attributes: bool = True
):
    vertices, edges, faces = get_mesh_data_from_vtk(vtk_data)

    mesh.clear_geometry()
    mesh.from_pydata(vertices=vertices, edges=edges, faces=faces)

    if update_attributes:
        set_mesh_attributes(mesh, vtk_data)


def vtk_to_mesh(vtk_data, mesh_name):
    vertices, edges, faces = get_mesh_data_from_vtk(vtk_data)

    mesh = bpy.data.meshes.new(mesh_name)
    mesh.from_pydata(vertices=vertices, edges=edges, faces=faces)
    mesh.update()

    return mesh


def set_mesh_attributes(mesh, vtk_data):
    for attr_name, values in vtk_data.point_data.items():
        initialize_material_attributes(
            attr_name, values, mesh, mesh.materials[0], "POINT"
        )

    for attr_name, values in vtk_data.cell_data.items():
        initialize_material_attributes(
            attr_name, values, mesh, mesh.materials[0], "FACE"
        )


def create_object(context, vtk_data, mesh_name) -> bpy.types.Object:
    # convert vtk mesh to blender mesh
    # if attributes exist
    # - create material for attributes
    # - set vtk data into mesh attributes
    # create blender object
    # convert mesh to point cloud if it is point cloud
    mesh = vtk_to_mesh(vtk_data, mesh_name)

    mesh_name = mesh.name
    obj = bpy.data.objects.new(mesh_name, mesh)
    # obj.location = vtk_data.center
    obj["data_range"] = {}
    bpy.context.scene.collection.objects.link(obj)
    bpy.context.view_layer.objects.active = obj

    if vtk_data.point_data or vtk_data.cell_data:
        mat = bpy.data.materials.new(name=f"{mesh_name}_attributes")
        mat["attributes"] = {}

        for attr_name, values in vtk_data.point_data.items():
            initialize_material_attributes(attr_name, values, mesh, mat, "POINT")

        for attr_name, values in vtk_data.cell_data.items():
            initialize_material_attributes(attr_name, values, mesh, mat, "FACE")

        create_attribute_material_nodes(mesh_name)
        update_attributes_enum(mat, context)

    is_point_cloud = (
        len(mesh.polygons) + len(mesh.edges) == 0 and "rad" in vtk_data.point_data
    )
    if is_point_cloud:
        convert_mesh_to_pointcloud(mesh_name)

    return obj
