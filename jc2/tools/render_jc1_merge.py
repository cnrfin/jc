#!/usr/bin/env python3
"""render_jc1_merge.py <a.npz> <b.npz> <a.bin> <b.bin> <out.png> : originals vs rebuilt, then a JC1-style merge sweep."""
import sys, os, math, numpy as np
from PIL import Image, ImageDraw
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import render_tex as RT, jc2_gltf as J, template_fit as TF
A, B = np.load(sys.argv[1]), np.load(sys.argv[2]); S = 230; yaw = math.pi * 0.3
def vis(Z): return Z['tris'][np.all(np.isin(Z['part'][Z['tris']], Z['visibleParts']), 1)]
def tile(P, UV, T, tex, label):
    im = Image.fromarray(RT.render(P[T], UV[T], tex, yaw, False, size=S)); ImageDraw.Draw(im).text((4, 4), label, fill=(255, 255, 0)); return np.array(im)
def orig(path, label):
    m = J.parse_model(open(path, 'rb').read()); P, UV, T = TF.mesh(m); return tile(P, UV, T, m['tex'], label)
row1 = [orig(sys.argv[3], 'JC2 original A'), tile(A['P'], A['UV'], vis(A), A['tex'], 'A on JC1 mesh'),
        orig(sys.argv[4], 'JC2 original B'), tile(B['P'], B['UV'], vis(B), B['tex'], 'B on JC1 mesh'), np.full((S, S, 3), 30, np.uint8)]
rows = [np.hstack(row1)]
# merge: vertex lerp; visible parts OR-ed like JC1; one texture
T = A['tris'][np.all(np.isin(A['part'][A['tris']], np.union1d(A['visibleParts'], B['visibleParts'])), 1)]
for tex, lab in ((A['tex'], 'tex A'), (B['tex'], 'tex B')):
    r = []
    for w in (0, .25, .5, .75, 1):
        r.append(tile((1 - w) * A['P'] + w * B['P'], A['UV'], T, tex, '%s %d%%' % (lab, w * 100)))
    rows.append(np.hstack(r))
Image.fromarray(np.vstack(rows)).save(sys.argv[5])
