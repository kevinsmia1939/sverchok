Surface Side Fill
=================

Functionality
-------------

This node caps the boundary of an open mesh without moving the original
vertices. It is intended for meshes produced by :doc:`Marching Cubes
</nodes/surface/marching_cubes>` or similar tools where the surface geometry is
already correct, but the boundary still needs to be closed to form a manifold
mesh.

The node builds a temporary BMesh from the incoming vertices and faces, finds
the boundary edges, then uses Blender's ``bmesh.ops.holes_fill`` operator to
create new faces over the open boundary. This preserves the incoming surface
geometry and only adds the missing cap faces.

Dependencies
------------

This node uses Blender's built-in BMesh operators only. No external surface
extraction library is required.

Inputs
------

- **Vertices**. Vertices of the input mesh.
- **Faces**. Faces of the input mesh.

Parameters
----------

- **Fill Side**. Choose whether the cap is kept as generated or flipped to the
  opposite side.
- **Correct normals**. Recalculate face normals after the boundary is filled.
- **Invert cap**. Legacy flip toggle kept for older saved node trees.

Outputs
-------

- **Vertices**. The original vertices plus any new cap faces using the same
  vertex positions.
- **Edges**. Edges of the resulting manifold mesh.
- **Polygons**. Faces of the resulting manifold mesh.

Usage Notes
-----------

Connect the output of :doc:`Marching Cubes </nodes/surface/marching_cubes>`
directly into this node. The input mesh geometry is preserved; only boundary
loops are closed.

``Correct normals`` is usually enough for clean output. Use ``Fill Side =
Outer`` when the marching-cubes output has opposite winding, for example after
using a negative iso value. ``Invert cap`` remains available for older saved
node trees, but ``Fill Side`` is the preferred control.

Examples of Usage
-----------------

Use this node when Marching Cubes produces an open surface that needs a cap at
its boundary. This is faster than remeshing because it does not resample the
surface and does not change vertex positions.
