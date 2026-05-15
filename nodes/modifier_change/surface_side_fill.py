# This file is part of project Sverchok. It's copyrighted by the contributors
# recorded in the version control history of the file, available from
# its original location https://github.com/nortikin/sverchok/commit/master
#
# SPDX-License-Identifier: GPL3
# License-Filename: LICENSE

import bpy
import numpy as np
from bpy.props import BoolProperty, EnumProperty, FloatProperty, IntProperty
from mathutils import Vector
from mathutils.bvhtree import BVHTree

from sverchok.data_structure import list_match_func, list_match_modes, updateNode
from sverchok.dependencies import mcubes
from sverchok.node_tree import SverchCustomTreeNode
from sverchok.utils.marching_cubes import isosurface_np


SURFACE_SIDE_ITEMS = [
    ("INSIDE", "Inside", "Fill the inside of the input surface"),
    ("OUTSIDE", "Outside", "Fill the outside of the input surface"),
]


def _has_vertices(values):
    if values is None:
        return False
    try:
        return len(values) > 0
    except TypeError:
        return False


def _poly_faces(vertices, faces):
    if vertices is None or faces is None:
        return None, None

    vertices = np.asarray(vertices, dtype=float)
    if len(vertices) == 0:
        return None, None

    poly_faces = [tuple(int(index) for index in face) for face in faces if len(face) >= 3]
    if not poly_faces:
        return None, None

    return vertices, poly_faces


def _bounds_from_vertices(vertices):
    points = np.asarray(vertices, dtype=float)
    bounds_min = points.min(axis=0)
    bounds_max = points.max(axis=0)
    return bounds_min, bounds_max


def _make_grid(bounds_min, bounds_max, samples_x, samples_y, samples_z):
    dimensions = np.array(
        [
            max(int(samples_x), 2),
            max(int(samples_y), 2),
            max(int(samples_z), 2),
        ],
        dtype=int,
    )
    extents = np.asarray(bounds_max - bounds_min, dtype=float)
    extents = np.where(extents == 0.0, 1.0, extents)
    spacing = extents / (dimensions - 1)
    origin = np.asarray(bounds_min, dtype=float) - spacing
    full_dimensions = dimensions + 2
    x_coords = origin[0] + spacing[0] * np.arange(full_dimensions[0], dtype=float)
    y_coords = origin[1] + spacing[1] * np.arange(full_dimensions[1], dtype=float)
    z_coords = origin[2] + spacing[2] * np.arange(full_dimensions[2], dtype=float)
    return origin, spacing, x_coords, y_coords, z_coords, full_dimensions


def _collect_scanline_hits(bvh, y, z, x_start, x_end, epsilon):
    axis = Vector((1.0, 0.0, 0.0))
    origin = Vector((x_start, y, z))
    remaining = x_end - x_start
    hits = []

    while remaining > epsilon:
        location, normal, index, distance = bvh.ray_cast(origin, axis, remaining)
        if index is None or location is None:
            break

        hit_x = float(location.x)
        if not hits or abs(hit_x - hits[-1]) > epsilon:
            hits.append(hit_x)

        origin = location + axis * epsilon
        remaining = x_end - origin.x

    return np.asarray(hits, dtype=float)


def _rasterize_volume(
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
    verts, poly_faces = _poly_faces(vertices, faces)
    if verts is None or poly_faces is None:
        return None

    bvh = BVHTree.FromPolygons(verts.tolist(), poly_faces, all_triangles=False, epsilon=0.0)

    if _has_vertices(bounds_vertices):
        bounds_min, bounds_max = _bounds_from_vertices(bounds_vertices)
    else:
        bounds_min, bounds_max = _bounds_from_vertices(verts)

    padding = max(float(padding), 0.0)
    if padding:
        span = bounds_max - bounds_min
        span = np.where(span == 0.0, 1.0, span)
        bounds_min = bounds_min - span * padding
        bounds_max = bounds_max + span * padding

    origin, spacing, x_coords, y_coords, z_coords, full_dimensions = _make_grid(
        bounds_min,
        bounds_max,
        samples_x,
        samples_y,
        samples_z,
    )

    epsilon = max(float(np.max(spacing)) * 1e-6, 1e-9)
    x_start = float(x_coords[0] - spacing[0])
    x_end = float(x_coords[-1] + spacing[0])
    inner_x = x_coords[1:-1]
    fill_inside = side == "INSIDE"
    if flip_side:
        fill_inside = not fill_inside

    volume = np.zeros(tuple(full_dimensions.tolist()), dtype=np.float32)
    for y_index, y in enumerate(y_coords[1:-1], start=1):
        for z_index, z in enumerate(z_coords[1:-1], start=1):
            hits = _collect_scanline_hits(bvh, float(y), float(z), x_start, x_end, epsilon)
            if hits.size == 0:
                selected = np.zeros_like(inner_x, dtype=bool)
                if not fill_inside:
                    selected = np.ones_like(inner_x, dtype=bool)
            else:
                inside = (np.searchsorted(hits, inner_x, side="right") % 2) == 1
                selected = inside if fill_inside else ~inside
            volume[1:-1, y_index, z_index] = selected.astype(np.float32)

    return volume, origin, spacing


def _scale_vertices(vertices, origin, spacing):
    verts = np.asarray(vertices, dtype=float)
    if len(verts) == 0:
        return verts

    verts = verts.copy()
    verts[:, 0] = origin[0] + verts[:, 0] * spacing[0]
    verts[:, 1] = origin[1] + verts[:, 1] * spacing[1]
    verts[:, 2] = origin[2] + verts[:, 2] * spacing[2]
    return verts


def _faces_to_mesh(vertices, faces):
    if len(vertices) == 0 or len(faces) == 0:
        return [], [], []

    face_list = [list(map(int, face)) for face in faces]
    edges = sorted(
        {
            tuple(sorted((face[index], face[(index + 1) % len(face)])))
            for face in face_list
            for index in range(len(face))
        }
    )
    return vertices.tolist(), [list(edge) for edge in edges], face_list


def _extract_surface_mesh(volume, origin, spacing):
    if mcubes is not None:
        vertices, faces = mcubes.marching_cubes(volume, 0.5)
    else:
        vertices, faces = isosurface_np(volume, 0.5)

    vertices = _scale_vertices(vertices, origin, spacing)
    return _faces_to_mesh(vertices, faces)


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
    rasterized = _rasterize_volume(
        vertices,
        faces,
        bounds_vertices,
        side,
        samples_x,
        samples_y,
        samples_z,
        padding,
        flip_side,
    )
    if rasterized is None:
        return [], [], []

    volume, origin, spacing = rasterized
    return _extract_surface_mesh(volume, origin, spacing)


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
        default="INSIDE",
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
