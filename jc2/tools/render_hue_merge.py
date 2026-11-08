#!/usr/bin/env python3
"""render_hue_merge.py <pairAB.npz> <A.bin> <B.bin> <out.png>
JC1's texture rule: ONE texture (the base parent's), hue-rotated toward the other parent by
the merge weight. Inherited parts are recoloured to the base texture's palette (mean/std
colour transfer in Lab-like space), so nothing carries a second look."""
import sys, os, math, numpy as np
from PIL import Image, ImageDraw
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import jc2_gltf as J, template_fit as TF, option2_merge as O

def match_palette(src, ref):
    """Recolour src so its colour statistics match ref (per channel in YCbCr)."""
    s = np.array(Image.fromarray(src[..., :3]).convert('YCbCr'), float); r = np.array(Image.fromarray(ref[..., :3]).convert('YCbCr'), float)
    ma = ref[..., 3] > 0; mb = src[..., 3] > 0
    for c in range(3):
        rs, rm = r[..., c][ma].std() + 1e-6, r[..., c][ma].mean(); ss, sm = s[..., c][mb].std() + 1e-6, s[..., c][mb].mean()
        k = rs / ss if c else 1.0                      # keep the part's own shading (luma contrast)
        s[..., c] = (s[..., c] - sm) * k + rm
    out = src.copy(); out[..., :3] = np.array(Image.fromarray(np.clip(s, 0, 255).astype(np.uint8), 'YCbCr').convert('RGB')); return out

Z = np.load(sys.argv[1]); S = 240; yaw = -math.pi * 0.62
texA, texB = Z['texA'], Z['texB']; hA, hB = O.mean_hue(texA), O.mean_hue(texB)
dh = ((hB - hA + .5) % 1) - .5
partB = match_palette(texB, texA)
def lab(img, t): im = Image.fromarray(img); ImageDraw.Draw(im).text((4, 4), t, fill=(255, 255, 0)); return np.array(im)
def orig(path, t):
    m = J.parse_model(open(path, 'rb').read()); P, UV, T = TF.mesh(m); return lab(O.render([(P[T], UV[T], m['tex'])], yaw, S)[0], t)
row1 = [orig(sys.argv[2], 'parent A'), orig(sys.argv[3], 'parent B')]
rows = []
for hue_scale, name in ((1.0, 'hue toward B'), (0.0, 'base hue')):
    r = []
    for w in (0, .25, .5, .75, 1):
        P = (1 - w) * Z['shapeA'] + w * Z['shapeB']
        rot = hue_scale * w * dh
        tA = O.hue_rotate(texA, rot); tB = O.hue_rotate(partB, rot)
        uv, tx = Z['lookA_uv'], Z['lookA_tex']; sel = np.isin(Z['group'], [0, 1, 2, 3]); parts = []
        for k, tex in ((0, tA), (1, tB)):
            m = sel & (tx == k)
            if m.any(): parts.append((P[Z['tris'][m]], uv[m], tex))
        r.append(lab(O.render(parts, yaw, S)[0], '%d%% B shape, %s' % (w * 100, name)))
    rows.append(np.hstack(r))
pad = np.full((S, S * 3, 3), 30, np.uint8)
Image.fromarray(np.vstack([np.hstack(row1 + [pad])] + rows)).save(sys.argv[4])
