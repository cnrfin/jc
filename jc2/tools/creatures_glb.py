#!/usr/bin/env python3
"""creatures_glb.py <out.glb> : Tuskrunner with Macebulk as a shape key (morph target),
so the JC1-style merge can be previewed with a slider in Blender."""
import sys, os, io, numpy as np
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import creature_gen as G, jc2_gltf as J
A, B = G.CREATURES['Tuskrunner'], G.CREATURES['Macebulk']
PA, UV, T, _ = G.build(A); PB, _, _, _ = G.build(B)
flip = np.array([1, -1, -1], np.float32)                              # JC2 axes -> glTF Y-up
pa, pb = (PA * flip).astype(np.float32), (PB * flip).astype(np.float32)
Gl = J.GLB(); j = Gl.j
buf = io.BytesIO(); Image.fromarray(A['pal']).save(buf, 'PNG')
j['images'] = [dict(bufferView=Gl.view(buf.getvalue()), mimeType='image/png')]
j['samplers'] = [dict(magFilter=9729, minFilter=9729)]; j['textures'] = [dict(source=0, sampler=0)]
j['materials'] = [dict(name='palette', doubleSided=True, pbrMetallicRoughness=dict(baseColorTexture=dict(index=0), metallicFactor=0, roughnessFactor=0.8))]
prim = dict(attributes=dict(POSITION=Gl.acc(pa, 5126, 'VEC3', 34962, True), TEXCOORD_0=Gl.acc(UV.astype(np.float32), 5126, 'VEC2', 34962)),
            indices=Gl.acc(T[:, ::-1].astype(np.uint32).reshape(-1), 5125, 'SCALAR', 34963), material=0,
            targets=[dict(POSITION=Gl.acc(pb - pa, 5126, 'VEC3', None, True))])
j['meshes'] = [dict(name='merge_family', primitives=[prim], weights=[0.0], extras=dict(targetNames=['Macebulk']))]
j['nodes'] = [dict(name='Tuskrunner_x_Macebulk', mesh=0)]; j['scenes'] = [dict(nodes=[0])]; j['scene'] = 0
Gl.write(sys.argv[1]); print('wrote', sys.argv[1])
