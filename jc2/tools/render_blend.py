#!/usr/bin/env python3
"""render_blend.py <A.bin> <B.bin> <fit.npz> <out.png> : A | fitted B | real B, then 0..100% blend sweep."""
import sys, math, numpy as np
from PIL import Image, ImageDraw
sys.path.insert(0, __file__.rsplit('/', 1)[0])
import jc2_gltf as J, render_tex as RT, template_fit as TF

def tris_uv(P, UV, tris): return P[tris], UV[tris]

A = J.parse_model(open(sys.argv[1], 'rb').read()); B = J.parse_model(open(sys.argv[2], 'rb').read())
F = np.load(sys.argv[3]); AP, AUV, At = TF.mesh(A); BP, BUV, Bt = TF.mesh(B)
V, texB = F['V'], F['tex']; texA = A['tex']
yaw = math.pi * 0.32; S = 230
def view(P, UV, tris, tex, label, fixed=None):
    T, U = tris_uv(P, UV, tris)
    im = Image.fromarray(RT.render(T, U, tex, yaw, False, size=S)); ImageDraw.Draw(im).text((4, 4), label, fill=(255, 255, 0)); return np.array(im)
row1 = [view(AP, AUV, At, texA, 'A 1821 (template)'), view(V, AUV, At, texB, 'B on A topology'), view(BP, BUV, Bt, B['tex'], 'B 1858 original')]
row2 = []
for t in (0, .25, .5, .75, 1):
    P = (1 - t) * AP + t * V
    tex = ((1 - t) * texA.astype(float) + t * texB.astype(float)).astype(np.uint8)
    row2.append(view(P, AUV, At, tex, '%d%%' % (t * 100)))
pad = np.full((S, S * 5 - S * 3, 3), 30, np.uint8)
Image.fromarray(np.vstack([np.hstack(row1 + [pad]), np.hstack(row2)])).save(sys.argv[4])
