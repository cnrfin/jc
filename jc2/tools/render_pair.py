#!/usr/bin/env python3
"""render_pair.py <pair.npz> <A.bin> <B.bin> <out.png> : originals vs family members, then merge sweeps."""
import sys, os, math, numpy as np
from PIL import Image, ImageDraw
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import jc2_gltf as J, template_fit as TF, option2_merge as O
Z = np.load(sys.argv[1]); S = 240; yaw = -math.pi * 0.62
def lab(img, t): im = Image.fromarray(img); ImageDraw.Draw(im).text((4, 4), t, fill=(255, 255, 0)); return np.array(im)
def orig(path, t):
    m = J.parse_model(open(path, 'rb').read()); P, UV, T = TF.mesh(m); return lab(O.render([(P[T], UV[T], m['tex'])], yaw, S)[0], t)
tex = [Z['texA'], Z['texB']]
def draw(P, look, groups, t):
    uv, tx = Z['look%s_uv' % look], Z['look%s_tex' % look]
    sel = np.isin(Z['group'], groups); parts = []
    for k in (0, 1):
        m = sel & (tx == k)
        if m.any(): parts.append((P[Z['tris'][m]], uv[m], tex[k]))
    return lab(O.render(parts, yaw, S)[0], t)
row1 = [orig(sys.argv[2], 'original A'), draw(Z['shapeA'], 'A', [0, 1], 'family member A'),
        orig(sys.argv[3], 'original B'), draw(Z['shapeB'], 'B', [0, 2], 'family member B'), np.full((S, S, 3), 30, np.uint8)]
rows = [np.hstack(row1)]
for look in ('A', 'B'):
    r = []
    for w in (0, .25, .5, .75, 1):
        P = (1 - w) * Z['shapeA'] + w * Z['shapeB']
        r.append(draw(P, look, [0, 1, 2], 'look %s  %d%%' % (look, w * 100)))      # JC1 OR rule: both groups on
    rows.append(np.hstack(r))
Image.fromarray(np.vstack(rows)).save(sys.argv[4])
