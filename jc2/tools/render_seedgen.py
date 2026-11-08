#!/usr/bin/env python3
"""render_seedgen.py <out.png> : the two designed seed creatures beside their JC2 inspirations, then the merge sweep."""
import sys, os, math, numpy as np
from PIL import Image, ImageDraw
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import seedgen as G, option2_merge as O, jc2_gltf as J, template_fit as TF
A, B = G.DESIGNS['Puffluch'], G.DESIGNS['Shellcoch']
PA, UV, T = G.build(A); PB, UVB, TB = G.build(B); assert len(PA) == len(PB) and (T == TB).all()
S = 250; yaw = math.pi * 0.22
def lab(a, t): im = Image.fromarray(a); ImageDraw.Draw(im).text((4, 4), t, fill=(255, 255, 0)); return np.array(im)
cwd = '/home/user/jc2-work/cddata/'
def orig(f, t):
    m = J.parse_model(open(cwd + '%d.bin' % f, 'rb').read()); P, U, TT = TF.mesh(m); return lab(O.render([(P[TT], U[TT], m['tex'])], yaw, S, shade=True)[0], t)
allP = np.r_[PA, PB]; c, s = math.cos(yaw), math.sin(yaw); R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]]); W = allP @ R.T; W[:, 1] *= -1
frame = (W.min(0) - 0.05, W.max(0) + 0.05)
def draw(P, tex, t): return lab(O.render([(P[T], UV[T], tex)], yaw, S, frame, shade=True)[0], t)
dh = ((O.mean_hue(B['pal']) - O.mean_hue(A['pal']) + .5) % 1) - .5
rows = [np.hstack([orig(1785, 'JC2 Devaluch'), draw(PA, A['pal'], 'Puffluch'), orig(1893, 'JC2 Dethalcoch'), draw(PB, B['pal'], 'Shellcoch'), np.full((S, S, 3), 30, np.uint8)])]
for tex, sg, name in ((A['pal'], 1, 'Puffluch look'), (B['pal'], -1, 'Shellcoch look')):
    r = []
    for w in (0, .25, .5, .75, 1):
        share = w if sg == 1 else 1 - w
        r.append(draw((1 - w) * PA + w * PB, O.hue_rotate(tex, sg * dh * share), '%s %d%%' % (name, w * 100)))
    rows.append(np.hstack(r))
Image.fromarray(np.vstack(rows)).save(sys.argv[1]); print('verts', len(PA), 'tris', len(T))
