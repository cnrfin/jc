#!/usr/bin/env python3
"""Write a rebuilt beast (jc1_to_jc2 npz) as a static textured GLB for hand-finishing in Blender.
usage: rebuilt_to_glb.py <in.npz> <out.glb>   (hidden JC1 part groups go in a separate 'hidden_parts' mesh)"""
import sys, os, io, numpy as np
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import jc2_gltf as J
Z = np.load(sys.argv[1]); G = J.GLB(); j = G.j
vis = np.all(np.isin(Z['part'][Z['tris']], Z['visibleParts']), 1)
buf = io.BytesIO(); Image.fromarray(Z['tex']).save(buf, 'PNG')
j['images'] = [dict(bufferView=G.view(buf.getvalue()), mimeType='image/png')]
j['samplers'] = [dict(magFilter=9728, minFilter=9728)]; j['textures'] = [dict(source=0, sampler=0)]
j['materials'] = [dict(name='shared_uv', doubleSided=True, pbrMetallicRoughness=dict(baseColorTexture=dict(index=0), metallicFactor=0, roughnessFactor=1))]
pos = G.acc(Z['P'].astype(np.float32), 5126, 'VEC3', 34962, True); uv = G.acc(Z['UV'].astype(np.float32), 5126, 'VEC2', 34962)
meshes, nodes = [], []
for name, sel in (('beast', vis), ('hidden_parts', ~vis)):
    if not sel.any(): continue
    idx = G.acc(Z['tris'][sel].astype(np.uint32).reshape(-1), 5125, 'SCALAR', 34963)
    meshes.append(dict(name=name, primitives=[dict(attributes=dict(POSITION=pos, TEXCOORD_0=uv), indices=idx, material=0)]))
    nodes.append(dict(name=name, mesh=len(meshes) - 1))
j['meshes'] = meshes
nodes.append(dict(name='root', rotation=[1, 0, 0, 0], children=list(range(len(nodes)))))
j['nodes'] = nodes; j['scenes'] = [dict(nodes=[len(nodes) - 1])]; j['scene'] = 0
G.write(sys.argv[2]); print('wrote', sys.argv[2])
