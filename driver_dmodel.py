bl_info = {
    "name": "Reflections Interactive Driver DMODEL Import",
    "author": "chasseyblue.com",
    "version": (1, 0, 0),
    "blender": (3, 0, 0),
    "location": "File > Import > Driver DMODEL (.dmodel)",
    "description": "Import Reflections Interactive Driver (.dmodel) vehicle files with UVs",
    "category": "Import-Export",
}

import struct
from pathlib import Path

import bpy
from bpy.types import Operator
from bpy_extras.io_utils import ImportHelper
from bpy.props import StringProperty, BoolProperty
from typing import List, Tuple


# ---------- low-level DMODEL decoding ----------

def _read_u16(data: bytes, off: int) -> int:
    return struct.unpack_from("<H", data, off)[0]


def _read_u32(data: bytes, off: int) -> int:
    return struct.unpack_from("<I", data, off)[0]


def _decode_header(data: bytes):
    """
    Extract counts and offsets from the DMODEL header.
    Matches Driver_BuildPolys in the MilkShape importer.
    """
    if len(data) < 0x30:
        raise ValueError("File too small to be a valid DMODEL")

    vertex_count      = _read_u16(data, 0x14)
    poly_cmd_count    = _read_u16(data, 0x16)
    mesh_count        = _read_u16(data, 0x1A)  # may be 0
    vert_offset       = _read_u32(data, 0x20)
    plane_offset      = _read_u32(data, 0x24)  # not used here
    normal_offset     = _read_u32(data, 0x28)  # not used here
    cmd_offset        = _read_u32(data, 0x2C)

    return {
        "vertex_count":   vertex_count,
        "poly_cmd_count": poly_cmd_count,
        "mesh_count":     mesh_count,
        "vert_offset":    vert_offset,
        "plane_offset":   plane_offset,
        "normal_offset":  normal_offset,
        "cmd_offset":     cmd_offset,
    }


def _read_vertices(data: bytes, vert_offset: int, vertex_count: int) -> List[Tuple[float, float, float]]:
    """
    Read float3 vertex positions from the DMODEL and apply the same
    transform as the original importer: (-x/10, -y/10, z/10).
    """
    verts: List[Tuple[float, float, float]] = []
    base = vert_offset
    for i in range(vertex_count):
        x, y, z = struct.unpack_from("<fff", data, base + i * 12)
        verts.append((-x / 10.0, -y / 10.0, z / 10.0))
    return verts


def _parse_poly_commands(
    data: bytes,
    cmd_offset: int,
    poly_cmd_count: int,
):
    """
    Walk the polygon command stream exactly like Driver_BuildPolys.

    Returns:
        tris          : list of (i0, i1, i2) vertex indices (0-based)
        uvs_per_tri   : list of 3 (u, v) pairs per triangle
        mat_ids       : meshId byte for each triangle (0..255)
    """
    tris: List[Tuple[int, int, int]] = []
    uvs_per_tri: List[List[Tuple[float, float]]] = []
    mat_ids: List[int] = []

    p = cmd_offset
    n = len(data)

    for cmd_idx in range(poly_cmd_count):
        if p >= n:
            raise ValueError(f"Command buffer overrun at command {cmd_idx}, offset 0x{p:x}")

        op = data[p]
        mesh_id = data[p + 1]

        if op == 0x10:
            # simple triangle, indices only
            i0, i1, i2 = struct.unpack_from("<HHH", data, p + 2)
            tris.append((i0, i1, i2))
            uvs_per_tri.append([(0.0, 0.0)] * 3)
            mat_ids.append(mesh_id)
            p += 0x10

        elif op == 0x12:
            # triangle with reordered indices, no UVs
            a = _read_u16(data, p + 2)
            b = _read_u16(data, p + 4)
            c = _read_u16(data, p + 6)
            tris.append((c, b, a))  # matches importer behaviour
            uvs_per_tri.append([(0.0, 0.0)] * 3)
            mat_ids.append(mesh_id)
            p += 0x14

        elif op == 0x13:
            # skipped (no geometry)
            p += 0x16

        elif op == 0x14:
            # single triangle with UVs
            a = _read_u16(data, p + 2)
            b = _read_u16(data, p + 4)
            c = _read_u16(data, p + 6)

            i0, i1, i2 = c, b, a

            u2 = data[p + 0x0A] / 256.0
            v2 = data[p + 0x0B] / 256.0
            u1 = data[p + 0x0C] / 256.0
            v1 = data[p + 0x0D] / 256.0
            u0 = data[p + 0x0E] / 256.0
            v0 = data[p + 0x0F] / 256.0

            tris.append((i0, i1, i2))
            uvs_per_tri.append([(u0, v0), (u1, v1), (u2, v2)])
            mat_ids.append(mesh_id)

            p += 0x14

        elif op == 0x15:
            # quad → two triangles with UVs
            a = _read_u16(data, p + 2)
            b = _read_u16(data, p + 4)
            c = _read_u16(data, p + 6)
            d = _read_u16(data, p + 8)

            # tri0 = (C, B, A)
            i0_0, i1_0, i2_0 = c, b, a
            # tri1 = (D, C, A)
            i0_1, i1_1, i2_1 = d, c, a

            uC = data[p + 0x0C] / 256.0
            vC = data[p + 0x0D] / 256.0
            uB = data[p + 0x0E] / 256.0
            vB = data[p + 0x0F] / 256.0
            uA = data[p + 0x10] / 256.0
            vA = data[p + 0x11] / 256.0
            uD = data[p + 0x12] / 256.0
            vD = data[p + 0x13] / 256.0

            tris.append((i0_0, i1_0, i2_0))
            uvs_per_tri.append([(uA, vA), (uB, vB), (uC, vC)])
            mat_ids.append(mesh_id)

            tris.append((i0_1, i1_1, i2_1))
            uvs_per_tri.append([(uD, vD), (uA, vA), (uC, vC)])
            mat_ids.append(mesh_id)

            p += 0x18

        elif op == 0x16:
            # single triangle with UVs, extended record
            a = _read_u16(data, p + 2)
            b = _read_u16(data, p + 4)
            c = _read_u16(data, p + 6)
            i0, i1, i2 = c, b, a

            u2 = data[p + 0x0E] / 256.0
            v2 = data[p + 0x0F] / 256.0
            u1 = data[p + 0x10] / 256.0
            v1 = data[p + 0x11] / 256.0
            u0 = data[p + 0x12] / 256.0
            v0 = data[p + 0x13] / 256.0

            tris.append((i0, i1, i2))
            uvs_per_tri.append([(u0, v0), (u1, v1), (u2, v2)])
            mat_ids.append(mesh_id)

            p += 0x18

        elif op == 0x17:
            # quad to two triangles with UVs, extended record
            a = _read_u16(data, p + 2)
            b = _read_u16(data, p + 4)
            c = _read_u16(data, p + 6)
            d = _read_u16(data, p + 8)

            i0_0, i1_0, i2_0 = c, b, a
            i0_1, i1_1, i2_1 = d, c, a

            uC = data[p + 0x14] / 256.0
            vC = data[p + 0x15] / 256.0
            uB = data[p + 0x16] / 256.0
            vB = data[p + 0x17] / 256.0
            uA = data[p + 0x18] / 256.0
            vA = data[p + 0x19] / 256.0
            uD = data[p + 0x1A] / 256.0
            vD = data[p + 0x1B] / 256.0

            tris.append((i0_0, i1_0, i2_0))
            uvs_per_tri.append([(uA, vA), (uB, vB), (uC, vC)])
            mat_ids.append(mesh_id)

            tris.append((i0_1, i1_1, i2_1))
            uvs_per_tri.append([(uD, vD), (uA, vA), (uC, vC)])
            mat_ids.append(mesh_id)

            p += 0x20

        else:
            raise ValueError(f"Unknown opcode 0x{op:02X} at offset 0x{p:X}")

    return tris, uvs_per_tri, mat_ids


# ---------- Blender integration ----------

def _build_mesh_object(
    name: str,
    verts: List[Tuple[float, float, float]],
    tris: List[Tuple[int, int, int]],
    uvs_per_tri: List[List[Tuple[float, float]]],
    mat_ids: List[int],
    create_materials: bool = True,
):
    """
    Create a Blender mesh object from decoded data.
    """
    mesh = bpy.data.meshes.new(name)
    obj = bpy.data.objects.new(name, mesh)

    # Link to current collection
    collection = bpy.context.collection
    collection.objects.link(obj)

    # Geometry
    mesh.from_pydata(verts, [], tris)
    mesh.update(calc_edges=True)

    # UVs: one loop per triangle corner, same order as tris
    uv_layer = mesh.uv_layers.new(name="UVMap")
    loop_data = uv_layer.data

    loop_index = 0
    for tri_uvs in uvs_per_tri:
        for u, v in tri_uvs:
            # Flip V for Blender if needed
            loop_data[loop_index].uv = (u, 1.0 - v)
            loop_index += 1

    # Materials per meshId (optional; can be used for texture slots or later mapping)
    if create_materials:
        unique_ids = sorted(set(mat_ids))
        mat_index_for_id = {}

        for mid in unique_ids:
            mat = bpy.data.materials.new(name=f"Mesh_{mid:02d}")
            mesh.materials.append(mat)
            mat_index_for_id[mid] = len(mesh.materials) - 1

        for poly, mid in zip(mesh.polygons, mat_ids):
            poly.material_index = mat_index_for_id.get(mid, 0)

    # Optionally set shading smooth etc. here if desired
    return obj


class IMPORT_SCENE_OT_driver_dmodel(Operator, ImportHelper):
    """Import a Driver .dmodel file"""
    bl_idname = "import_scene.driver_dmodel"
    bl_label = "Import Driver DMODEL"
    bl_options = {'UNDO'}

    filename_ext = ".dmodel"
    filter_glob: StringProperty(
        default="*.dmodel",
        options={'HIDDEN'},
    )

    create_materials: BoolProperty(
        name="Create materials per meshId",
        description="Create and assign basic materials grouped by meshId",
        default=True,
    )

    def execute(self, context):
        path = Path(self.filepath)
        try:
            data = path.read_bytes()
        except Exception as e:
            self.report({'ERROR'}, f"Failed to read file: {e}")
            return {'CANCELLED'}

        try:
            hdr = _decode_header(data)
            verts = _read_vertices(data, hdr["vert_offset"], hdr["vertex_count"])
            tris, uvs_per_tri, mat_ids = _parse_poly_commands(
                data,
                hdr["cmd_offset"],
                hdr["poly_cmd_count"],
            )
        except Exception as e:
            self.report({'ERROR'}, f"Failed to decode DMODEL: {e}")
            return {'CANCELLED'}

        obj_name = path.stem
        _build_mesh_object(
            obj_name,
            verts,
            tris,
            uvs_per_tri,
            mat_ids,
            create_materials=self.create_materials,
        )

        self.report(
            {'INFO'},
            f"Imported {obj_name}: {len(verts)} verts, {len(tris)} tris",
        )
        return {'FINISHED'}


def menu_func_import(self, context):
    self.layout.operator(
        IMPORT_SCENE_OT_driver_dmodel.bl_idname,
        text="Driver DMODEL (.dmodel)",
    )


def register():
    bpy.utils.register_class(IMPORT_SCENE_OT_driver_dmodel)
    bpy.types.TOPBAR_MT_file_import.append(menu_func_import)


def unregister():
    bpy.types.TOPBAR_MT_file_import.remove(menu_func_import)
    bpy.utils.unregister_class(IMPORT_SCENE_OT_driver_dmodel)


if __name__ == "__main__":
    register()
