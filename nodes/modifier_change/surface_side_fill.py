# This file is part of project Sverchok. It's copyrighted by the contributors
# recorded in the version control history of the file, available from
# its original location https://github.com/nortikin/sverchok/commit/master
#
# SPDX-License-Identifier: GPL3
# License-Filename: LICENSE

import bpy
import bmesh
from bpy.props import BoolProperty

from sverchok.data_structure import dataCorrect, updateNode, zip_long_repeat
from sverchok.node_tree import SverchCustomTreeNode
from sverchok.utils.sv_bmesh_utils import bmesh_from_pydata, pydata_from_bmesh


def _cap_boundary(vertices, faces, correct_normals, invert_cap):
    if not vertices or not faces:
        return [], [], []

    bm = bmesh_from_pydata(vertices, faces=faces, normal_update=False)
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

    if correct_normals:
        bm.normal_update()
        bmesh.ops.recalc_face_normals(bm, faces=bm.faces[:])

    if invert_cap and cap_faces:
        bmesh.ops.reverse_faces(bm, faces=cap_faces)

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

    correct_normals: BoolProperty(
        name="Correct normals",
        description="Recalculate normals after filling the boundary",
        default=True,
        update=updateNode,
    )

    invert_cap: BoolProperty(
        name="Invert cap",
        description="Flip the newly created cap faces",
        default=False,
        update=updateNode,
    )

    def sv_init(self, context):
        self.inputs.new("SvVerticesSocket", "Vertices")
        self.inputs.new("SvStringsSocket", "Faces")

        self.outputs.new("SvVerticesSocket", "Vertices")
        self.outputs.new("SvStringsSocket", "Edges")
        self.outputs.new("SvStringsSocket", "Polygons")

    def draw_buttons(self, context, layout):
        box = layout.box()
        box.prop(self, "correct_normals")
        box.prop(self, "invert_cap")

    def draw_buttons_ext(self, context, layout):
        layout.prop(self, "correct_normals")
        layout.prop(self, "invert_cap")

    def process(self):
        if not any(output.is_linked for output in self.outputs):
            return

        if not (self.inputs["Vertices"].is_linked and self.inputs["Faces"].is_linked):
            return

        vertices_s = dataCorrect(self.inputs["Vertices"].sv_get(default=[[]], deepcopy=False))
        faces_s = dataCorrect(self.inputs["Faces"].sv_get(default=[[]], deepcopy=False))

        verts_out = []
        edges_out = []
        faces_out = []

        for vertices, faces in zip_long_repeat(vertices_s, faces_s):
            verts, edges, polys = _cap_boundary(
                vertices,
                faces,
                self.correct_normals,
                self.invert_cap,
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
