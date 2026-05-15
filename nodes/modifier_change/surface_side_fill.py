# This file is part of project Sverchok. It's copyrighted by the contributors
# recorded in the version control history of the file, available from
# its original location https://github.com/nortikin/sverchok/commit/master
#
# SPDX-License-Identifier: GPL3
# License-Filename: LICENSE

import bpy
import numpy as np
from bpy.props import BoolProperty, EnumProperty, FloatProperty, IntProperty
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

    poly_faces = []
    for face in faces:
        if len(face) < 3:
            continue
        indices = [int(index) for index in face]
        if len(indices) == 3:
            poly_faces.append(tuple(indices))
        else:
            root = indices[0]
            for i in range(1, len(indices) - 1):
                poly_faces.append((root, indices[i], indices[i + 1]))
    if not poly_faces:
        return None, None

    return vertices, poly_faces


def _mesh_topology(vertices, faces):
    face_normals = []
    edge_faces = {}
    vert_faces = {}

    for face_index, face in enumerate(faces):
        if len(face) != 3:
            continue

        a, b, c = face
        tri = vertices[np.array((a, b, c), dtype=int)]
        normal = np.cross(tri[1] - tri[0], tri[2] - tri[0])
        length = np.linalg.norm(normal)
        if length > 0.0:
            normal = normal / length
        face_normals.append(normal)

        for vertex_index in face:
            vert_faces.setdefault(vertex_index, []).append(face_index)

        for edge in ((a, b), (b, c), (c, a)):
            key = tuple(sorted(edge))
            edge_faces.setdefault(key, []).append(face_index)

    return np.asarray(face_normals, dtype=float), edge_faces, vert_faces


def _triangle_barycentric(point, tri):
    a, b, c = tri
    v0 = b - a
    v1 = c - a
    v2 = point - a
    d00 = np.dot(v0, v0)
    d01 = np.dot(v0, v1)
    d11 = np.dot(v1, v1)
    d20 = np.dot(v2, v0)
    d21 = np.dot(v2, v1)
    denom = d00 * d11 - d01 * d01
    if abs(denom) < 1e-20:
        return np.array((1.0, 0.0, 0.0), dtype=float)
    v = (d11 * d20 - d01 * d21) / denom
    w = (d00 * d21 - d01 * d20) / denom
    u = 1.0 - v - w
    return np.array((u, v, w), dtype=float)


def _bounds_from_vertices(vertices):
    points = np.asarray(vertices, dtype=float)
    bounds_min = points.min(axis=0)
    bounds_max = points.max(axis=0)
    return bounds_min, bounds_max


def _make_grid(bounds_min, bounds_max, samples_x, samples_y, samples_z):
    cell_counts = np.array(
        [
            max(int(samples_x), 1),
            max(int(samples_y), 1),
            max(int(samples_z), 1),
        ],
        dtype=int,
    )
    extents = np.asarray(bounds_max - bounds_min, dtype=float)
    extents = np.where(extents == 0.0, 1.0, extents)
    spacing = extents / cell_counts
    origin = np.asarray(bounds_min, dtype=float) - spacing
    point_dimensions = cell_counts + 3
    x_coords = origin[0] + spacing[0] * np.arange(point_dimensions[0], dtype=float)
    y_coords = origin[1] + spacing[1] * np.arange(point_dimensions[1], dtype=float)
    z_coords = origin[2] + spacing[2] * np.arange(point_dimensions[2], dtype=float)
    return origin, spacing, x_coords, y_coords, z_coords, point_dimensions


def _feature_normal(vertices, faces, face_normals, edge_faces, vert_faces, face_index, location, epsilon):
    tri_indices = faces[face_index]
    tri = vertices[np.asarray(tri_indices, dtype=int)]
    weights = _triangle_barycentric(location, tri)

    zero_mask = np.abs(weights) <= epsilon
    if zero_mask.sum() == 0:
        return face_normals[face_index]

    if zero_mask.sum() == 1:
        # Edge case
        edge_positions = np.flatnonzero(~zero_mask)
        edge_key = tuple(sorted((tri_indices[edge_positions[0]], tri_indices[edge_positions[1]])))
        adj_faces = edge_faces.get(edge_key, [])
        normals = face_normals[adj_faces] if adj_faces else face_normals[face_index:face_index + 1]
        normal = normals.sum(axis=0)
        length = np.linalg.norm(normal)
        if length > 0.0:
            normal = normal / length
        return normal

    # Vertex case
    vertex_pos = int(np.flatnonzero(~zero_mask)[0])
    vertex_index = tri_indices[vertex_pos]
    adj_faces = vert_faces.get(vertex_index, [])
    if not adj_faces:
        return face_normals[face_index]

    point_vec = vertices[vertex_index]
    normal = np.zeros(3, dtype=float)
    for adj_face in adj_faces:
        adj_tri_indices = faces[adj_face]
        adj_tri = vertices[np.asarray(adj_tri_indices, dtype=int)]
        adj_normal = face_normals[adj_face]
        other = [idx for idx in adj_tri_indices if idx != vertex_index]
        if len(other) != 2:
            continue
        v1 = vertices[other[0]] - point_vec
        v2 = vertices[other[1]] - point_vec
        v1_len = np.linalg.norm(v1)
        v2_len = np.linalg.norm(v2)
        if v1_len == 0.0 or v2_len == 0.0:
            continue
        v1 /= v1_len
        v2 /= v2_len
        alpha = np.arccos(np.clip(np.dot(v1, v2), -1.0, 1.0))
        normal += alpha * adj_normal

    length = np.linalg.norm(normal)
    if length > 0.0:
        normal = normal / length
    else:
        normal = face_normals[face_index]
    return normal


def _signed_distance(bvh, point, vertices, faces, face_normals, edge_faces, vert_faces, epsilon):
    location, normal, index, distance = bvh.find_nearest(point)
    if location is None or normal is None or index is None:
        return None

    point_vec = np.asarray(point, dtype=float)
    location_vec = np.asarray(location, dtype=float)
    delta = point_vec - location_vec
    dist = np.linalg.norm(delta)
    if dist <= epsilon:
        return 0.0

    feature_normal = _feature_normal(
        vertices,
        faces,
        face_normals,
        edge_faces,
        vert_faces,
        index,
        location_vec,
        epsilon,
    )

    # Match vtkImplicitPolyDataDistance sign convention:
    # negative inside, positive outside.
    signed = dist if np.dot(delta, feature_normal) > 0.0 else -dist
    return signed


def _sample_scalar_field(
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
    face_normals, edge_faces, vert_faces = _mesh_topology(verts, poly_faces)

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
    fill_inside = side == "INSIDE"
    if flip_side:
        fill_inside = not fill_inside

    field = np.empty(tuple(full_dimensions.tolist()), dtype=np.float32)
    for x_index, x in enumerate(x_coords):
        for y_index, y in enumerate(y_coords):
            column = field[x_index, y_index]
            for z_index, z in enumerate(z_coords):
                signed = _signed_distance(
                    bvh,
                    (float(x), float(y), float(z)),
                    verts,
                    poly_faces,
                    face_normals,
                    edge_faces,
                    vert_faces,
                    epsilon,
                )
                if signed is None:
                    continue
                surface_value = signed if fill_inside else -signed
                box_value = max(
                    bounds_min[0] - x,
                    x - bounds_max[0],
                    bounds_min[1] - y,
                    y - bounds_max[1],
                    bounds_min[2] - z,
                    z - bounds_max[2],
                )
                column[z_index] = max(surface_value, box_value)

    return field, origin, spacing


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


def _extract_surface_mesh(field, origin, spacing):
    if mcubes is not None:
        vertices, faces = mcubes.marching_cubes(field, 0.0)
    else:
        vertices, faces = isosurface_np(field, 0.0)

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
    rasterized = _sample_scalar_field(
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

    field, origin, spacing = rasterized
    return _extract_surface_mesh(field, origin, spacing)


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
