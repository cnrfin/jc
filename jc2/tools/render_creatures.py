#!/usr/bin/env python3
"""render_creatures.py <out.png> : the two designed creatures and a JC1-style merge sweep."""
import sys, os, math, numpy as np
from PIL import Image, ImageDraw
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import creature_gen as G, option2_merge as O
A, B = G.CREATURES['Tuskrunner'], G.CREATURES['Macebulk']
PA, UV, T, part = G.build(A); PB, UVB, TB, _ = G.build(B)
assert len(PA) == len(PB) and (T == TB).all() and np.allclose(UV, UVB)        # one shared topology
S = 250; yaw = -math.pi * 0.62
def lab(img, t): im = Image.fromarray(img); ImageDraw.Draw(im).text((4, 4), t, fill=(255, 255, 0)); return np.array(im)
dh = ((O.mean_hue(B['pal']) - O.mean_hue(A['pal']) + .5) % 1) - .5
allP = np.r_[PA, PB]; c, s = math.cos(yaw), math.sin(yaw); R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
V = allP @ R.T; V[:, 1] *= -1; frame = (V.min(0) - 0.05, V.max(0) + 0.05)
def draw(P, tex, t): return lab(O.render([(P[T], UV[T], tex)], yaw, S, frame, shade=True)[0], t)
rows = [np.hstack([draw(PA, A['pal'], 'A  Tuskrunner'), draw(PB, B['pal'], 'B  Macebulk'),
                   np.full((S, S * 3, 3), 30, np.uint8)])]
for look, base, name in ((A['pal'], 1, 'look A'), (B['pal'], -1, 'look B')):
    r = []
    for w in (0, .25, .5, .75, 1):
        P = (1 - w) * PA + w * PB
        hue = (w * dh) if base == 1 else (-(1 - w) * dh)               # one palette, hue moves toward the other parent
        r.append(draw(P, O.hue_rotate(look, hue), '%s  %d%% B' % (name, w * 100)))
    rows.append(np.hstack(r))
Image.fromarray(np.vstack(rows)).save(sys.argv[1]); print('vertices', len(PA), 'triangles', len(T))
