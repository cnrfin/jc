#!/usr/bin/env python3
"""render_pair2.py <pairAB.npz> <pairBA.npz> <A.bin> <B.bin> <out.png>
Look A uses the family built on A's mesh, look B the family built on B's mesh.
w = amount of B shape in both rows."""
import sys, os, math, numpy as np
from PIL import Image, ImageDraw
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import jc2_gltf as J, template_fit as TF, option2_merge as O
ZA, ZB = np.load(sys.argv[1]), np.load(sys.argv[2]); S = 240; yaw = -math.pi * 0.62
def lab(img, t): im = Image.fromarray(img); ImageDraw.Draw(im).text((4, 4), t, fill=(255, 255, 0)); return np.array(im)
def orig(path, t):
    m = J.parse_model(open(path, 'rb').read()); P, UV, T = TF.mesh(m); return lab(O.render([(P[T], UV[T], m['tex'])], yaw, S)[0], t)
def draw(Z, P, groups, t):
    """Z built on its own base ('A' slot = base): look = base look, tinted inherited parts."""
    tex = [Z['texA'], Z['texB']]; h = [O.mean_hue(x) for x in tex]
    uv, tx = Z['lookA_uv'], Z['lookA_tex']; sel = np.isin(Z['group'], groups); parts = []
    for k in (0, 1):
        m = sel & (tx == k)
        if not m.any(): continue
        tt = tex[k] if k == 0 else O.hue_rotate(tex[k], ((h[0] - h[1] + .5) % 1) - .5)
        parts.append((P[Z['tris'][m]], uv[m], tt))
    return lab(O.render(parts, yaw, S)[0], t)
rows = [np.hstack([orig(sys.argv[3], 'original A'), draw(ZA, ZA['shapeA'], [0, 1, 3], 'member A'),
                   orig(sys.argv[4], 'original B'), draw(ZB, ZB['shapeA'], [0, 1, 3], 'member B'), np.full((S, S, 3), 30, np.uint8)])]
for Z, name, base_is_A in ((ZA, 'look A', True), (ZB, 'look B', False)):
    r = []
    for w in (0, .25, .5, .75, 1):
        a = w if base_is_A else 1 - w                 # amount of the OTHER beast's shape in this family
        P = (1 - a) * Z['shapeA'] + a * Z['shapeB']
        r.append(draw(Z, P, [0, 1, 2, 3], '%s  %d%% B shape' % (name, w * 100)))
    rows.append(np.hstack(r))
Image.fromarray(np.vstack(rows)).save(sys.argv[5])
