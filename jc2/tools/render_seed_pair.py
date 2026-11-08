#!/usr/bin/env python3
"""render_seed_pair.py <AB.npz> <BA.npz> <A.bin> <B.bin> <out.png>
Row 1: originals and both family members. Rows 2-3: merge in A's look (A's remesh family) and B's look."""
import sys, os, math, numpy as np
from PIL import Image, ImageDraw
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import jc2_gltf as J, template_fit as TF, option2_merge as O
AB, BA = np.load(sys.argv[1]), np.load(sys.argv[2]); S = 240; yaw = -math.pi * 0.62
def lab(a, t): im = Image.fromarray(a); ImageDraw.Draw(im).text((4, 4), t, fill=(255, 255, 0)); return np.array(im)
def orig(p, t):
    m = J.parse_model(open(p, 'rb').read()); P, UV, T = TF.mesh(m); return lab(O.render([(P[T], UV[T], m['tex'])], yaw, S, shade=True)[0], t)
def draw(Z, P, tex, t): return lab(O.render([(P[Z['T']], Z['cuv'], tex)], yaw, S, shade=True)[0], t)
dh = ((O.mean_hue(AB['texB']) - O.mean_hue(AB['texA']) + .5) % 1) - .5
rows = [np.hstack([orig(sys.argv[3], 'original A'), draw(AB, AB['PA'], AB['texA'], 'rebuilt A'), orig(sys.argv[4], 'original B'),
                   draw(BA, BA['PA'], BA['texA'], 'rebuilt B'), draw(AB, AB['PB'], AB['texB'], 'A mesh as B')])]
for Z, base, name in ((AB, 'A', 'A look'), (BA, 'B', 'B look')):
    r = []
    for w in (0, .25, .5, .75, 1):
        a = w if base == 'A' else 1 - w                     # amount of the OTHER shape in this family
        P = (1 - a) * Z['PA'] + a * Z['PB']
        hue = (dh * w) if base == 'A' else (-dh * (1 - w))
        r.append(draw(Z, P, O.hue_rotate(Z['texA'], hue), '%s  %d%% B' % (name, w * 100)))
    rows.append(np.hstack(r))
Image.fromarray(np.vstack(rows)).save(sys.argv[5])
