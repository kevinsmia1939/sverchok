# This file is part of project Sverchok. It's copyrighted by the contributors
# recorded in the version control history of the file, available from
# its original location https://github.com/nortikin/sverchok/commit/master
#
# SPDX-License-Identifier: GPL3
# License-Filename: LICENSE

import os
import site
import sys

import bpy
import numpy as np
from bpy.props import BoolProperty, EnumProperty, FloatProperty, IntProperty

from sverchok.data_structure import list_match_func, list_match_modes, updateNode
from sverchok.node_tree import SverchCustomTreeNode


SURFACE_SIDE_ITEMS = [
    ("SIDE_A", "Side A", "Fill one side of the input surface"),
    ("SIDE_B", "Side B", "Fill the opposite side of the input surface"),
]


def _ensure_pyvista_on_path():
    paths = [site.getusersitepackages()]
    for path in paths:
        if os.path.isdir(path) and path not in sys.path:
            sys.path.insert(0, path)


def _polydata_from_sverchok_mesh(vertices, faces):
    face_data = []
    for face in faces:
        if len(face) >= 3:
            face_data.extend([len(face), *face])

    if not vertices or not face_data:
        return None

    import pyvista as pv

    polydata = pv.PolyData(
        np.asarray(vertices, dtype=float),
        np.asarray(face_data, dtype=np.int64),
    )
    return (
        polydata.clean()
        .triangulate()
        .compute_normals(
            consistent_normals=True,
            auto_orient_normals=True,
            inplace=False,
        )
    )


def _bounds_from_vertices(vertices):
    points = np.asarray(vertices, dtype=float)
    bounds_min = points.min(axis=0)
    bounds_max = points.max(axis=0)
    return bounds_min, bounds_max


def _make_grid(bounds_min, bounds_max, samples_x, samples_y, samples_z):
    import pyvista as pv

    dimensions = (
        max(int(samples_x), 2),
        max(int(samples_y), 2),
        max(int(samples_z), 2),
    )
    extents = bounds_max - bounds_min
    extents = np.where(extents == 0.0, 1.0, extents)
    spacing = tuple(extents / (np.asarray(dimensions) - 1))
    return pv.ImageData(
        dimensions=dimensions,
        spacing=spacing,
        origin=tuple(bounds_min),
    )


def _polydata_to_sverchok_mesh(polydata):
    if not polydata.is_all_triangles:
        polydata = polydata.triangulate()

    polydata = polydata.flip_faces()

    faces = []
    offset = 0
    raw_faces = polydata.faces
    while offset < len(raw_faces):
        face_size = int(raw_faces[offset])
        start = offset + 1
        end = start + face_size
        faces.append([int(index) for index in raw_faces[start:end]])
        offset = end

    edges = sorted(
        {
            tuple(sorted((face[index], face[(index + 1) % len(face)])))
            for face in faces
            for index in range(len(face))
        }
    )

    return polydata.points.tolist(), [list(edge) for edge in edges], faces


def fill_surface_side(
    vertices,
    faces,
    bounds_vertices,
    side,
    samples_x,
    samples_y,
    samples_z,
    padding,
    flip_side,
):
    _ensure_pyvista_on_path()

    surface = _polydata_from_sverchok_mesh(vertices, faces)
    if surface is None:
        return [], [], []

    if bounds_vertices:
        bounds_min, bounds_max = _bounds_from_vertices(bounds_vertices)
    else:
        bounds_min, bounds_max = _bounds_from_vertices(vertices)

    padding = max(float(padding), 0.0)
    if padding:
        span = bounds_max - bounds_min
        bounds_min = bounds_min - span * padding
        bounds_max = bounds_max + span * padding

    grid = _make_grid(bounds_min, bounds_max, samples_x, samples_y, samples_z)

    invert = side == "SIDE_A"
    if flip_side:
        invert = not invert

    clipped = grid.clip_surface(surface, invert=invert)
    polydata = (
        clipped.extract_surface(algorithm="dataset_surface")
        .clean()
        .triangulate()
    )

    return _polydata_to_sverchok_mesh(polydata)


class SvSurfaceSideFillNode(SverchCustomTreeNode, bpy.types.Node):
    """
    Triggers: surface side fill, half-space fill, marching cubes fill
    Tooltip: Fill one side of an incoming surface mesh
    """

    bl_idname = "SvSurfaceSideFillNode"
    bl_label = "Surface Side Fill"
    bl_icon = "MESH_GRID"

    side: EnumProperty(
        name="Side",
        description="Which side of the input surface to fill",
        items=SURFACE_SIDE_ITEMS,
        default="SIDE_A",
        update=updateNode,
    )

    samples_x: IntProperty(
        name="Samples X",
        description="Fill grid samples on X",
        default=30,
        min=2,
        soft_max=100,
        update=updateNode,
    )

    samples_y: IntProperty(
        name="Samples Y",
        description="Fill grid samples on Y",
        default=30,
        min=2,
        soft_max=100,
        update=updateNode,
    )

    samples_z: IntProperty(
        name="Samples Z",
        description="Fill grid samples on Z",
        default=30,
        min=2,
        soft_max=100,
        update=updateNode,
    )

    padding: FloatProperty(
        name="Padding",
        description="Relative padding to add when Bounds is not connected",
        default=0.0,
        min=0.0,
        soft_max=0.25,
        update=updateNode,
    )

    flip_side: BoolProperty(
        name="Flip Side",
        description="Reverse which side of the surface is filled",
        default=False,
        update=updateNode,
    )

    list_match: EnumProperty(
        name="List Match",
        description="Behavior on different list lengths, object level",
        items=list_match_modes,
        default="REPEAT",
        update=updateNode,
    )

    def sv_init(self, context):
        self.inputs.new("SvVerticesSocket", "Vertices")
        self.inputs.new("SvStringsSocket", "Faces")
        self.inputs.new("SvVerticesSocket", "Bounds")
        self.inputs.new("SvStringsSocket", "Samples X").prop_name = "samples_x"
        self.inputs.new("SvStringsSocket", "Samples Y").prop_name = "samples_y"
        self.inputs.new("SvStringsSocket", "Samples Z").prop_name = "samples_z"
        self.inputs.new("SvStringsSocket", "Padding").prop_name = "padding"

        self.outputs.new("SvVerticesSocket", "Vertices")
        self.outputs.new("SvStringsSocket", "Edges")
        self.outputs.new("SvStringsSocket", "Polygons")

    def draw_buttons(self, context, layout):
        row = layout.row(align=True)
        row.prop(self, "side", expand=True)
        layout.prop(self, "flip_side", toggle=True)

    def draw_buttons_ext(self, context, layout):
        layout.prop(self, "side")
        layout.prop(self, "flip_side")
        layout.prop(self, "list_match")

    def process(self):
        if not any(output.is_linked for output in self.outputs):
            return

        vertices_s = self.inputs["Vertices"].sv_get(default=[[]])
        faces_s = self.inputs["Faces"].sv_get(default=[[]])
        if self.inputs["Bounds"].is_linked:
            bounds_s = self.inputs["Bounds"].sv_get(default=[[]])
        else:
            bounds_s = [None]

        samples_x_s = [
            max(int(value), 2)
            for value in self.inputs["Samples X"].sv_get()[0]
        ]
        samples_y_s = [
            max(int(value), 2)
            for value in self.inputs["Samples Y"].sv_get()[0]
        ]
        samples_z_s = [
            max(int(value), 2)
            for value in self.inputs["Samples Z"].sv_get()[0]
        ]
        padding_s = [
            max(float(value), 0.0)
            for value in self.inputs["Padding"].sv_get()[0]
        ]

        params = list_match_func[self.list_match](
            [vertices_s, faces_s, bounds_s, samples_x_s, samples_y_s, samples_z_s, padding_s]
        )

        verts_out = []
        edges_out = []
        faces_out = []

        for vertices, faces, bounds, samples_x, samples_y, samples_z, padding in zip(*params):
            verts, edges, polygons = fill_surface_side(
                vertices,
                faces,
                bounds,
                self.side,
                samples_x,
                samples_y,
                samples_z,
                padding,
                self.flip_side,
            )
            verts_out.append(verts)
            edges_out.append(edges)
            faces_out.append(polygons)

        if self.outputs["Vertices"].is_linked:
            self.outputs["Vertices"].sv_set(verts_out)

        if self.outputs["Edges"].is_linked:
            self.outputs["Edges"].sv_set(edges_out)

        if self.outputs["Polygons"].is_linked:
            self.outputs["Polygons"].sv_set(faces_out)


def register():
    bpy.utils.register_class(SvSurfaceSideFillNode)


def unregister():
    bpy.utils.unregister_class(SvSurfaceSideFillNode)
