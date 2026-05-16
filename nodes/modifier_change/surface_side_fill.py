# This file is part of project Sverchok. It's copyrighted by the contributors
# recorded in the version control history of the file, available from
# its original location https://github.com/nortikin/sverchok/commit/master
#
# SPDX-License-Identifier: GPL3
# License-Filename: LICENSE

import bpy
import bmesh
from bpy.props import BoolProperty, EnumProperty
from mathutils import Vector

from sverchok.data_structure import dataCorrect, updateNode, zip_long_repeat
from sverchok.node_tree import SverchCustomTreeNode
from sverchok.utils.sv_bmesh_utils import bmesh_from_pydata, pydata_from_bmesh


def _bounds_from_vertices(vertices):
    points = [Vector(v) for v in vertices or []]
    if not points:
        return None, None

    bounds_min = Vector((
        min(point.x for point in points),
        min(point.y for point in points),
        min(point.z for point in points),
    ))
    bounds_max = Vector((
        max(point.x for point in points),
        max(point.y for point in points),
        max(point.z for point in points),
    ))
    return bounds_min, bounds_max


def _boundary_components(bm):
    boundary_edges = [edge for edge in bm.edges if len(edge.link_faces) == 1]
    if not boundary_edges:
        return []

    vert_edges = {}
    for edge in boundary_edges:
        for vert in edge.verts:
            vert_edges.setdefault(vert, []).append(edge)

    components = []
    visited = set()
    for start_edge in boundary_edges:
        if start_edge in visited:
            continue

        stack = [start_edge]
        component_edges = []
        component_verts = set()
        visited.add(start_edge)

        while stack:
            edge = stack.pop()
            component_edges.append(edge)
            for vert in edge.verts:
                component_verts.add(vert)
                for next_edge in vert_edges.get(vert, []):
                    if next_edge not in visited:
                        visited.add(next_edge)
                        stack.append(next_edge)

        components.append((component_verts, component_edges))

    return components


def _component_endpoints(component_verts, component_edges):
    degree = {vert: 0 for vert in component_verts}
    for edge in component_edges:
        degree[edge.verts[0]] += 1
        degree[edge.verts[1]] += 1
    return [vert for vert, count in degree.items() if count == 1]


def _component_plane(component_verts, bounds_min, bounds_max):
    coords = [vert.co for vert in component_verts]
    spans = [
        max(coord[index] for coord in coords) - min(coord[index] for coord in coords)
        for index in range(3)
    ]
    axis = min(range(3), key=lambda index: spans[index])
    avg = sum(coord[axis] for coord in coords) / len(coords)
    if abs(avg - bounds_min[axis]) <= abs(avg - bounds_max[axis]):
        plane_value = bounds_min[axis]
    else:
        plane_value = bounds_max[axis]
    uv_axes = tuple(index for index in range(3) if index != axis)
    return axis, plane_value, uv_axes


def _choose_corner(endpoints, bounds_min, bounds_max, axis, plane_value, uv_axes):
    if len(endpoints) != 2:
        return None

    u_axis, v_axis = uv_axes
    uv_min = Vector((bounds_min[u_axis], bounds_min[v_axis]))
    uv_max = Vector((bounds_max[u_axis], bounds_max[v_axis]))
    corners_2d = [
        Vector((uv_min.x, uv_min.y)),
        Vector((uv_max.x, uv_min.y)),
        Vector((uv_max.x, uv_max.y)),
        Vector((uv_min.x, uv_max.y)),
    ]

    endpoint_uv = [Vector((endpoint.co[u_axis], endpoint.co[v_axis])) for endpoint in endpoints]
    corner_index = min(
        range(4),
        key=lambda index: sum((point - corners_2d[index]).length_squared for point in endpoint_uv),
    )

    corner = Vector((0.0, 0.0, 0.0))
    corner[axis] = plane_value
    corner[u_axis] = corners_2d[corner_index].x
    corner[v_axis] = corners_2d[corner_index].y
    return corner


def _find_or_create_vert(bm, component_verts, co, epsilon):
    for vert in component_verts:
        if (vert.co - co).length <= epsilon:
            return vert
    return bm.verts.new(co)


def _close_open_components(bm, bounds_min, bounds_max):
    components = _boundary_components(bm)
    if not components:
        return

    epsilon = max((bounds_max - bounds_min).length * 1e-6, 1e-9)

    for component_verts, component_edges in components:
        endpoints = _component_endpoints(component_verts, component_edges)
        if len(endpoints) != 2:
            continue

        axis, plane_value, uv_axes = _component_plane(component_verts, bounds_min, bounds_max)
        corner_co = _choose_corner(endpoints, bounds_min, bounds_max, axis, plane_value, uv_axes)
        if corner_co is None:
            continue

        corner_vert = _find_or_create_vert(bm, component_verts, corner_co, epsilon)
        for endpoint in endpoints:
            try:
                bm.edges.new((endpoint, corner_vert))
            except ValueError:
                pass


def _cap_boundary(vertices, faces, bounds_vertices, correct_normals, invert_cap, fill_side):
    if not vertices or not faces:
        return [], [], []

    bm = bmesh_from_pydata(vertices, faces=faces, normal_update=False)

    if bounds_vertices:
        bounds_min, bounds_max = _bounds_from_vertices(bounds_vertices)
    else:
        bounds_min, bounds_max = _bounds_from_vertices(vertices)

    if bounds_min is not None and bounds_max is not None:
        _close_open_components(bm, bounds_min, bounds_max)

    boundary_edges = [edge for edge in bm.edges if len(edge.link_faces) == 1]

    if not boundary_edges:
        if correct_normals:
            bm.normal_update()
            bmesh.ops.recalc_face_normals(bm, faces=bm.faces[:])
        verts, edges, polys = pydata_from_bmesh(bm)
        bm.clear()
        bm.free()
        return verts, edges, polys

    fill_result = bmesh.ops.holes_fill(
        bm,
        edges=boundary_edges,
        sides=max(len(boundary_edges), 3),
    )
    cap_faces = fill_result.get("faces", [])

    if (invert_cap or fill_side == "OUTER") and cap_faces:
        bmesh.ops.reverse_faces(bm, faces=cap_faces)

    if correct_normals:
        bm.normal_update()
        bmesh.ops.recalc_face_normals(bm, faces=bm.faces[:])

    verts, edges, polys = pydata_from_bmesh(bm)
    bm.clear()
    bm.free()
    return verts, edges, polys


class SvSurfaceSideFillNode(SverchCustomTreeNode, bpy.types.Node):
    """
    Triggers: surface side fill, boundary cap, hole cap
    Tooltip: Cap the boundary of an open mesh while preserving its vertices
    """

    bl_idname = "SvSurfaceSideFillNode"
    bl_label = "Surface Side Fill"
    bl_icon = "MESH_GRID"

    fill_side: EnumProperty(
        name="Fill Side",
        description="Choose which side of the cap should face outward",
        items=[
            ("INNER", "Inner", "Keep the cap orientation as generated"),
            ("OUTER", "Outer", "Flip the cap orientation"),
        ],
        default="INNER",
        update=updateNode,
    )

    correct_normals: BoolProperty(
        name="Correct normals",
        description="Recalculate normals after filling the boundary",
        default=True,
        update=updateNode,
    )

    invert_cap: BoolProperty(
        name="Invert cap",
        description="Legacy cap flip toggle. Kept for older saved node trees.",
        default=False,
        update=updateNode,
    )

    def sv_init(self, context):
        self.inputs.new("SvVerticesSocket", "Vertices")
        self.inputs.new("SvStringsSocket", "Faces")
        self.inputs.new("SvVerticesSocket", "Bounds")

        self.outputs.new("SvVerticesSocket", "Vertices")
        self.outputs.new("SvStringsSocket", "Edges")
        self.outputs.new("SvStringsSocket", "Polygons")

    def draw_buttons(self, context, layout):
        box = layout.box()
        box.prop(self, "fill_side")
        box.prop(self, "correct_normals")
        box.prop(self, "invert_cap")

    def draw_buttons_ext(self, context, layout):
        layout.prop(self, "fill_side")
        layout.prop(self, "correct_normals")
        layout.prop(self, "invert_cap")

    def process(self):
        if not any(output.is_linked for output in self.outputs):
            return

        if not (self.inputs["Vertices"].is_linked and self.inputs["Faces"].is_linked):
            return

        vertices_s = dataCorrect(self.inputs["Vertices"].sv_get(default=[[]], deepcopy=False))
        faces_s = dataCorrect(self.inputs["Faces"].sv_get(default=[[]], deepcopy=False))
        if self.inputs["Bounds"].is_linked:
            bounds_s = dataCorrect(self.inputs["Bounds"].sv_get(default=[None], deepcopy=False))
        else:
            bounds_s = [None]

        verts_out = []
        edges_out = []
        faces_out = []

        for vertices, faces, bounds in zip_long_repeat(vertices_s, faces_s, bounds_s):
            invert_cap = self.invert_cap or self.fill_side == "OUTER"
            verts, edges, polys = _cap_boundary(
                vertices,
                faces,
                bounds,
                self.correct_normals,
                invert_cap,
                self.fill_side,
            )
            verts_out.append(verts)
            edges_out.append(edges)
            faces_out.append(polys)

        self.outputs["Vertices"].sv_set(verts_out)
        self.outputs["Edges"].sv_set(edges_out)
        self.outputs["Polygons"].sv_set(faces_out)


def register():
    bpy.utils.register_class(SvSurfaceSideFillNode)


def unregister():
    bpy.utils.unregister_class(SvSurfaceSideFillNode)
