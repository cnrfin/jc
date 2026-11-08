#!/usr/bin/env python3
"""Two-beast merge family on a remesh of beast A (Blender / bpy 5.2).

usage: bl_pair_remesh.py <A.bin> <B.bin> <out.npz> [--voxel 0.02] [--tris 12000] [--res 1024]

1. A's own model is voxel-remeshed into one watertight surface and decimated to ~--tris
   triangles: at w = 0 the family member IS A (shape within a voxel of the original).
2. Smart UV Project gives the shared UV layout.
3. A copy is fitted onto B: a scale/translate match of the bounding boxes, then
   Shrinkwrap (nearest surface point) + Corrective Smooth, twice, finishing with a
   tight shrinkwrap so the surface sits on B.
4. Cycles bakes A's texture onto shape A and B's texture onto shape B, both on the shared UVs.
Output npz: PA, PB (vertex positions, JC2 axes), T, cuv (per-corner UVs, top-down V), texA, texB.
"""
import sys, os, argparse, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import bpy, bmesh
import jc2_gltf as J, template_fit as TF

def to_bl(P): return np.c_[P[:, 0], P[:, 2], -P[:, 1]]
def from_bl(W): return np.c_[W[:, 0], -W[:, 2], W[:, 1]]

def jc2_object(name, path):
    m = J.parse_model(open(path, 'rb').read()); P, UV, T = TF.mesh(m)
    me = bpy.data.meshes.new(name); me.from_pydata([tuple(p) for p in to_bl(P)], [], [tuple(int(i) for i in t) for t in T]); me.update()
    lay = me.uv_layers.new(name='UVMap')
    lay.data.foreach_set('uv', np.array([(UV[v][0], 1 - UV[v][1]) for t in T for v in t], np.float32).ravel())
    ob = bpy.data.objects.new(name, me); bpy.context.scene.collection.objects.link(ob)
    img = bpy.data.images.new(name + '_tex', m['tex'].shape[1], m['tex'].shape[0], alpha=True)
    img.pixels.foreach_set((m['tex'][::-1].astype(np.float32) / 255).ravel())
    mat = bpy.data.materials.new(name + '_mat'); nt = mat.node_tree
    b = next(n for n in nt.nodes if n.type == 'BSDF_PRINCIPLED'); tn = nt.nodes.new('ShaderNodeTexImage'); tn.image = img
    nt.links.new(tn.outputs['Color'], b.inputs['Base Color']); me.materials.append(mat)
    return ob

def apply_mods(ob):
    dg = bpy.context.evaluated_depsgraph_get(); ev = ob.evaluated_get(dg)
    me = bpy.data.meshes.new_from_object(ev); old = ob.data; ob.modifiers.clear(); ob.data = me; bpy.data.meshes.remove(old)

def coords(ob):
    W = np.zeros(len(ob.data.vertices) * 3, np.float32); ob.data.vertices.foreach_get('co', W); return W.reshape(-1, 3)

def set_coords(ob, W): ob.data.vertices.foreach_set('co', np.asarray(W, np.float32).ravel()); ob.data.update()

def select_only(*obs, active=None):
    for o in bpy.context.scene.objects: o.select_set(False)
    for o in obs: o.select_set(True)
    bpy.context.view_layer.objects.active = active or obs[-1]

def bake(target, source, res):
    img = bpy.data.images.new('bake_' + source.name, res, res, alpha=True)
    if not target.data.materials:
        m = bpy.data.materials.new('target_mat'); target.data.materials.append(m)
    nt = target.data.materials[0].node_tree
    for n in [n for n in nt.nodes if n.type == 'TEX_IMAGE']: nt.nodes.remove(n)
    tn = nt.nodes.new('ShaderNodeTexImage'); tn.image = img; nt.nodes.active = tn
    sc = bpy.context.scene; bk = sc.render.bake
    bk.use_selected_to_active = True; bk.margin = 16; bk.use_pass_direct = False; bk.use_pass_indirect = False; bk.use_pass_color = True
    out = None
    for ext, ray in ((0.03, 0.08), (0.15, 0.6)):                  # tight rays first, long rays fill the misses
        img.pixels.foreach_set(np.zeros(res * res * 4, np.float32))
        bk.cage_extrusion = ext; bk.max_ray_distance = ray
        select_only(source, target, active=target); bpy.ops.object.bake(type='DIFFUSE', pass_filter={'COLOR'})
        px = np.zeros(res * res * 4, np.float32); img.pixels.foreach_get(px); px = px.reshape(-1, 4)
        if out is None: out = px.copy()
        else: miss = out[:, 3] < 0.5; out[miss] = px[miss]
    return (np.clip(out.reshape(res, res, 4)[::-1], 0, 1) * 255).astype(np.uint8)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('A'); ap.add_argument('B'); ap.add_argument('out')
    ap.add_argument('--voxel', type=float, default=0.02); ap.add_argument('--tris', type=int, default=12000); ap.add_argument('--res', type=int, default=1024)
    ap.add_argument('--lm-a'); ap.add_argument('--lm-b')
    a = ap.parse_args()
    bpy.ops.wm.read_factory_settings(use_empty=True)
    sc = bpy.context.scene; sc.render.engine = 'CYCLES'; sc.cycles.samples = 1; sc.cycles.device = 'CPU'
    A = jc2_object('A', a.A); B = jc2_object('B', a.B)
    # 1. remesh a copy of A
    S = bpy.data.objects.new('shared', A.data.copy()); sc.collection.objects.link(S); S.data.materials.clear()
    rm = S.modifiers.new('remesh', 'REMESH'); rm.mode = 'VOXEL'; rm.voxel_size = a.voxel; rm.adaptivity = 0
    apply_mods(S)
    nf = sum(len(p.vertices) - 2 for p in S.data.polygons)
    if nf > a.tris:
        dc = S.modifiers.new('dec', 'DECIMATE'); dc.ratio = a.tris / nf; apply_mods(S)
    bm = bmesh.new(); bm.from_mesh(S.data); bmesh.ops.triangulate(bm, faces=bm.faces); bm.to_mesh(S.data); bm.free(); S.data.update()
    # 2. shared UVs
    for l in list(S.data.uv_layers): S.data.uv_layers.remove(l)
    S.data.uv_layers.new(name='UVMap'); select_only(S)
    bpy.ops.object.mode_set(mode='EDIT'); bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.uv.smart_project(angle_limit=1.2, island_margin=0.004); bpy.ops.object.mode_set(mode='OBJECT')
    PA = coords(S)
    texA = bake(S, A, a.res)
    # 3. fit onto B
    WA, WB = coords(A), coords(B)
    lo_a, hi_a, lo_b, hi_b = WA.min(0), WA.max(0), WB.min(0), WB.max(0)
    if a.lm_a and a.lm_b:                                   # landmark warp: legs to legs, tail to tail, body to body
        import json
        from scipy.interpolate import RBFInterpolator
        def load(p):
            out = {}
            for k, v in json.load(open(p)).items():
                if k.startswith('_'): continue
                x, up, z = v; out[k] = (x, z, up)                # (x, up, z) -> Blender (x, z, up)
                if k.startswith('L_'): out['R_' + k[2:]] = (-x, z, up)
            return out
        la, lb = load(a.lm_a), load(a.lm_b); keys = [k for k in la if k in lb]
        LA, LB = np.array([la[k] for k in keys]), np.array([lb[k] for k in keys])
        warp = RBFInterpolator(LA, LB - LA, kernel='thin_plate_spline', smoothing=1e-4)
        set_coords(S, PA + warp(PA))
    else:
        scale = (hi_b - lo_b) / np.maximum(hi_a - lo_a, 1e-6)
        set_coords(S, lo_b + (PA - lo_a) * scale)
    for k, (smooth, off) in enumerate(((0.6, 0.0), (0.3, 0.0), (0.0, 0.002))):
        sw = S.modifiers.new('wrap', 'SHRINKWRAP'); sw.target = B; sw.wrap_method = 'NEAREST_SURFACEPOINT'; sw.offset = off
        if smooth:
            cs = S.modifiers.new('relax', 'CORRECTIVE_SMOOTH'); cs.factor = smooth; cs.iterations = 6; cs.use_only_smooth = True
        apply_mods(S)
    PB = coords(S)
    texB = bake(S, B, a.res)
    T = np.array([[v for v in p.vertices] for p in S.data.polygons])
    luv = np.zeros(len(S.data.loops) * 2, np.float32); S.data.uv_layers.active.data.foreach_get('uv', luv)
    luv = luv.reshape(-1, 2); luv[:, 1] = 1 - luv[:, 1]
    np.savez(a.out, PA=from_bl(PA), PB=from_bl(PB), T=T, cuv=luv.reshape(len(T), 3, 2), texA=texA, texB=texB)
    print('family', a.out, 'verts', len(PA), 'tris', len(T))

if __name__ == '__main__':
    main()
