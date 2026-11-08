#!/usr/bin/env python3
"""render_cuv_merge.py <cuvA.npz> <cuvB.npz> <A.bin> <B.bin> <out.png>
Row 1: originals vs rebuilt on the shared mesh using the ORIGINAL textures (per-corner UVs).
Rows 2-3: merge sweep; the look (texture + its UVs) comes from one parent, geometry blends."""
import sys, os, math, numpy as np
from PIL import Image, ImageDraw
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import render_tex as RT, jc2_gltf as J, template_fit as TF
A, B = np.load(sys.argv[1]), np.load(sys.argv[2]); S = 240; yaw = -math.pi * 0.62
def lab(img, t): im = Image.fromarray(img); ImageDraw.Draw(im).text((4, 4), t, fill=(255, 255, 0)); return np.array(im)
def orig(path, t):
    m = J.parse_model(open(path, 'rb').read()); P, UV, T = TF.mesh(m); return lab(RT.render(P[T], UV[T], m['tex'], yaw, False, size=S), t)
def look(Z, P, sel=None):
    key = {tuple(t): i for i, t in enumerate(Z['ctris'])}
    return key
row1 = [orig(sys.argv[3], 'original A'), lab(RT.render(A['P'][A['ctris']], A['cuv'], A['otex'], yaw, False, size=S), 'A rebuilt, original tex'),
        orig(sys.argv[4], 'original B'), lab(RT.render(B['P'][B['ctris']], B['cuv'], B['otex'], yaw, False, size=S), 'B rebuilt, original tex'),
        np.full((S, S, 3), 30, np.uint8)]
rows = [np.hstack(row1)]
for Z, name in ((A, 'look A'), (B, 'look B')):
    r = []
    for w in (0, .25, .5, .75, 1):
        P = (1 - w) * A['P'] + w * B['P']
        r.append(lab(RT.render(P[Z['ctris']], Z['cuv'], Z['otex'], yaw, False, size=S), '%s  %d%% B shape' % (name, w * 100)))
    rows.append(np.hstack(r))
Image.fromarray(np.vstack(rows)).save(sys.argv[5])
