#!/usr/bin/env python3
"""fit_overlay.py <model.bin> <params.json> <out.png>
Side / front / top orthographic views of a JC2 model with a creature_gen mesh drawn over it
as a translucent magenta silhouette plus outline, for tuning generator parameters by eye."""
import sys, os, json, numpy as np
from PIL import Image, ImageDraw
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import jc2_gltf as J, template_fit as TF, creature_gen as G, ortho_grid as OG

def overlay(model_path, params, out, size=560):
    m = J.parse_model(open(model_path, 'rb').read()); P, UV, T = TF.mesh(m)
    Q, _, QT, _ = G.build(params)
    pad = 0.08; allp = np.r_[P, Q]; lo, hi = allp.min(0) - pad, allp.max(0) + pad
    panels = []
    for ax_h, ax_v, dax, flip, lo2, hi2, name in ((2, 1, 0, False, (lo[2], -hi[1]), (hi[2], -lo[1]), 'SIDE z / up'),
                                                  (0, 1, 2, False, (lo[0], -hi[1]), (hi[0], -lo[1]), 'FRONT x / up'),
                                                  (0, 2, 1, True, (lo[0], lo[2]), (hi[0], hi[2]), 'TOP x / z')):
        im, f = OG.raster(P[T], UV[T], m['tex'], ax_h, ax_v, dax, 1, lo2, hi2, size, flip)
        lay = Image.new('RGBA', im.size, (0, 0, 0, 0)); d = ImageDraw.Draw(lay)
        for t in QT:
            pts = []
            for v in t:
                h = Q[v][ax_h]; vv = -Q[v][ax_v] if ax_v == 1 else Q[v][ax_v]; pts.append(f(h, vv))
            d.polygon(pts, fill=(255, 0, 255, 28), outline=(255, 0, 255, 70))
        im = Image.alpha_composite(im.convert('RGBA'), lay).convert('RGB')
        ImageDraw.Draw(im).text((size // 2 - 40, size - 14), name, fill=(0, 255, 255)); panels.append(np.array(im))
    Image.fromarray(np.hstack(panels)).save(out)

if __name__ == '__main__':
    params = dict(G.CREATURES['Tuskrunner']); params.update(json.load(open(sys.argv[2])))
    params['pal'] = G.palette((150, 40, 35), (215, 150, 120), (235, 230, 215), (250, 220, 70), (90, 80, 90), (200, 200, 210))
    overlay(sys.argv[1], params, sys.argv[3])
