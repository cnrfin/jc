#!/usr/bin/env python3
"""Fit the shared creature mesh onto a JC2 beast and bake its texture, in Blender (bpy 5.2).

usage (with the bpy venv):  bl_wrap_bake.py <model.bin> <params.json> <out.npz> [--uv shared_uv.npy] [--res 1024]

1. build the shared creature mesh from creature_gen with the species' fitted parameters
2. shared UV layout: if --uv exists it is reused (same topology -> same UVs for every species),
   otherwise it is made once here with Blender's Smart UV Project and saved there
3. Shrinkwrap (nearest surface point) the body parts onto the JC2 model; horns, ears, spikes,
   eyes, claws and the mace keep their fitted shape (they ride along with the smoothing only)
4. Cycles bake (Selected to Active, diffuse colour) of the JC2 model's own texture onto the
   shared UVs
Output npz: P (wrapped positions, creature_gen order), T, uv (per-vertex), tex (baked RGBA), part.
"""
import sys, os, json, argparse, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import bpy, bmesh
import jc2_gltf as J, template_fit as TF, creature_gen as G

BODY = {'torso', 'head', 'fleg', 'bleg', 'tail'}

def to_bl(P):   # JC2 (x, y=-up, z) -> Blender (x, z, up) ; any consistent rigid map works for baking
    return np.c_[P[:, 0], P[:, 2], -P[:, 1]]

def make_mesh(name, P, T, uv=None, loop_uv=None):
    me = bpy.data.meshes.new(name); me.from_pydata([tuple(p) for p in P], [], [tuple(int(i) for i in t) for t in T]); me.update()
    if uv is not None or loop_uv is not None:
        lay = me.uv_layers.new(name='UVMap')
        if loop_uv is None:
            loop_uv = np.array([uv[li.vertex_index] for li in me.loops])
        lay.data.foreach_set('uv', np.asarray(loop_uv, np.float32).ravel())
    ob = bpy.data.objects.new(name, me); bpy.context.scene.collection.objects.link(ob); return ob

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('model'); ap.add_argument('params'); ap.add_argument('out')
    ap.add_argument('--uv', default=os.path.join(HERE, '..', 'species', 'shared_uv.npy')); ap.add_argument('--res', type=int, default=1024)
    a = ap.parse_args()
    bpy.ops.wm.read_factory_settings(use_empty=True)
    params = dict(G.CREATURES['Tuskrunner']); params.update({k: v for k, v in json.load(open(a.params)).items() if not k.startswith('_')})
    Q, _, QT, part = G.build(params)
    # JC2 beast with its own texture
    m = J.parse_model(open(a.model, 'rb').read()); P, UV, T = TF.mesh(m)
    src = make_mesh('jc2', to_bl(P), T, loop_uv=np.array([(UV[v][0], 1 - UV[v][1]) for t in T for v in t]))   # Blender V is bottom-up
    img = bpy.data.images.new('jc2tex', m['tex'].shape[1], m['tex'].shape[0], alpha=True)
    img.pixels.foreach_set((m['tex'][::-1].astype(np.float32) / 255).ravel())
    mat = bpy.data.materials.new('jc2mat'); nt = mat.node_tree
    bsdf = next(n for n in nt.nodes if n.type == 'BSDF_PRINCIPLED'); tn = nt.nodes.new('ShaderNodeTexImage'); tn.image = img; tn.interpolation = 'Closest'
    nt.links.new(tn.outputs['Color'], bsdf.inputs['Base Color']); src.data.materials.append(mat)
    # shared creature mesh + shared UV layout
    cre = make_mesh('creature', to_bl(Q), QT)
    bm = bmesh.new(); bm.from_mesh(cre.data); bmesh.ops.recalc_face_normals(bm, faces=bm.faces); bm.to_mesh(cre.data); bm.free()
    QT = np.array([[v for v in p.vertices] for p in cre.data.polygons])      # winding after the normal fix
    bpy.context.view_layer.objects.active = cre; cre.select_set(True)
    if os.path.exists(a.uv):
        luv = np.load(a.uv); lay = cre.data.uv_layers.new(name='UVMap'); lay.data.foreach_set('uv', luv.astype(np.float32).ravel())
    else:
        bpy.ops.object.mode_set(mode='EDIT'); bpy.ops.mesh.select_all(action='SELECT')
        bpy.ops.uv.smart_project(angle_limit=1.15, island_margin=0.01); bpy.ops.object.mode_set(mode='OBJECT')
        luv = np.zeros(len(cre.data.loops) * 2, np.float32); cre.data.uv_layers.active.data.foreach_get('uv', luv)
        np.save(a.uv, luv.reshape(-1, 2)); print('made shared UV layout ->', a.uv)
    # body parts wrap onto the JC2 surface; detail parts only follow the smoothing
    vg = cre.vertex_groups.new(name='body'); vg.add([i for i, p in enumerate(part) if p in BODY], 1.0, 'REPLACE')
    sw = cre.modifiers.new('wrap', 'SHRINKWRAP'); sw.target = src; sw.wrap_method = 'NEAREST_SURFACEPOINT'; sw.vertex_group = 'body'; sw.offset = 0.004
    cs = cre.modifiers.new('relax', 'CORRECTIVE_SMOOTH'); cs.factor = 0.5; cs.iterations = 4; cs.vertex_group = 'body'
    dg = bpy.context.evaluated_depsgraph_get(); ev = cre.evaluated_get(dg)
    W = np.zeros(len(ev.data.vertices) * 3, np.float32); ev.data.vertices.foreach_get('co', W); W = W.reshape(-1, 3)
    for md in list(cre.modifiers): cre.modifiers.remove(md)
    cre.data.vertices.foreach_set('co', W.ravel()); cre.data.update()
    # bake the JC2 texture onto the shared UVs
    out = bpy.data.images.new('baked', a.res, a.res, alpha=True)
    cm = bpy.data.materials.new('cmat'); cnt = cm.node_tree; ctn = cnt.nodes.new('ShaderNodeTexImage'); ctn.image = out; cnt.nodes.active = ctn
    cre.data.materials.append(cm)
    sc = bpy.context.scene; sc.render.engine = 'CYCLES'; sc.cycles.samples = 1; sc.cycles.device = 'CPU'
    bk = sc.render.bake; bk.use_selected_to_active = True; bk.cage_extrusion = 0.2; bk.max_ray_distance = 0.6; bk.margin = 8
    bk.use_pass_direct = False; bk.use_pass_indirect = False; bk.use_pass_color = True
    for o in sc.objects: o.select_set(False)
    src.select_set(True); cre.select_set(True); bpy.context.view_layer.objects.active = cre
    bpy.ops.object.bake(type='DIFFUSE', pass_filter={'COLOR'})
    px = np.zeros(a.res * a.res * 4, np.float32); out.pixels.foreach_get(px)
    # second pass with long rays fills texels whose short rays missed (detail parts far from the surface)
    near = px.reshape(-1, 4).copy()
    out.pixels.foreach_set(np.zeros_like(px)); bk.cage_extrusion = 0.6; bk.max_ray_distance = 2.5
    bpy.ops.object.bake(type='DIFFUSE', pass_filter={'COLOR'})
    far = np.zeros_like(px); out.pixels.foreach_get(far); far = far.reshape(-1, 4)
    miss = near[:, 3] < 0.5; near[miss] = far[miss]; px = near.ravel()
    tex = (np.clip(px.reshape(a.res, a.res, 4)[::-1], 0, 1) * 255).astype(np.uint8)
    Pw = np.c_[W[:, 0], -W[:, 2], W[:, 1]]                     # back to JC2 axes
    vuv = np.zeros((len(Pw), 2)); 
    for li, lp in zip(range(len(cre.data.loops)), cre.data.loops): vuv[lp.vertex_index] = luv.reshape(-1, 2)[li]
    luv2 = luv.reshape(-1, 2).copy(); luv2[:, 1] = 1 - luv2[:, 1]; cuv = luv2.reshape(len(QT), 3, 2)   # top-down V for the flipped image
    np.savez(a.out, P=Pw, T=QT, cuv=cuv, tex=tex, part=part)
    print('wrapped + baked', a.out, 'verts', len(Pw), 'tris', len(QT))

if __name__ == '__main__':
    main()
