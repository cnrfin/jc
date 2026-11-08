#!/usr/bin/env python3
"""Orthographic side / front / top views of a model with a labelled coordinate grid,
for reading feature positions off the image.

usage: ortho_grid.py <model.bin> <out.png> [--points pts.json]

Axes are the model's own (JC2: -Y up, beasts face -Z). Labels are in model units.
Side view: horizontal = Z, vertical = -Y. Front: horizontal = X, vertical = -Y. Top: horizontal = X, vertical = Z.
"""
import sys, os, json, numpy as np
from PIL import Image, ImageDraw
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import jc2_gltf as J, template_fit as TF

def raster(T, U, tex, ax_h, ax_v, depth_ax, depth_sign, lo, hi, size, flip_v):
    img = np.full((size, size, 3), 25, np.uint8); zb = np.full((size, size), 1e9)
    sc = (size - 60) / max(hi[0] - lo[0], hi[1] - lo[1])
    def to_px(h, v):
        x = 40 + (h - lo[0]) * sc; y = (size - 20) - (v - lo[1]) * sc if not flip_v else 20 + (v - lo[1]) * sc
        return x, y
    th, tw = tex.shape[:2]
    H = T[..., ax_h]; Vv = -T[..., ax_v] if ax_v == 1 else T[..., ax_v]
    D = depth_sign * T[..., depth_ax]
    X, Y = to_px(H, Vv)
    for t in range(len(T)):
        xs, ys, zs = X[t], Y[t], D[t]
        x0, x1 = int(max(0, xs.min())), int(min(size - 1, xs.max())) + 1
        y0, y1 = int(max(0, ys.min())), int(min(size - 1, ys.max())) + 1
        if x1 <= x0 or y1 <= y0: continue
        gx, gy = np.meshgrid(np.arange(x0, x1) + .5, np.arange(y0, y1) + .5)
        d = (ys[1] - ys[2]) * (xs[0] - xs[2]) + (xs[2] - xs[1]) * (ys[0] - ys[2])
        if abs(d) < 1e-9: continue
        l0 = ((ys[1] - ys[2]) * (gx - xs[2]) + (xs[2] - xs[1]) * (gy - ys[2])) / d
        l1 = ((ys[2] - ys[0]) * (gx - xs[2]) + (xs[0] - xs[2]) * (gy - ys[2])) / d
        l2 = 1 - l0 - l1; m = (l0 >= 0) & (l1 >= 0) & (l2 >= 0)
        z = l0 * zs[0] + l1 * zs[1] + l2 * zs[2]; sub = zb[y0:y1, x0:x1]; m &= z < sub
        u = l0 * U[t, 0, 0] + l1 * U[t, 1, 0] + l2 * U[t, 2, 0]; v = l0 * U[t, 0, 1] + l1 * U[t, 1, 1] + l2 * U[t, 2, 1]
        px = np.clip((u % 1) * tw, 0, tw - 1).astype(int); py = np.clip((v % 1) * th, 0, th - 1).astype(int)
        sub[m] = z[m]; img[y0:y1, x0:x1][m] = (tex[py[m], px[m], :3] * 0.85).astype(np.uint8)
    im = Image.fromarray(img); dr = ImageDraw.Draw(im)
    step = 0.1
    for k in np.arange(np.floor(lo[0] / step) * step, hi[0] + 1e-6, step):
        x, _ = to_px(k, lo[1]); major = abs(k / 0.5 - round(k / 0.5)) < 1e-6
        dr.line([(x, 0), (x, size)], fill=(90, 90, 90) if major else (50, 50, 50))
        if major: dr.text((x + 2, 2), '%.1f' % k, fill=(255, 255, 0))
    for k in np.arange(np.floor(lo[1] / step) * step, hi[1] + 1e-6, step):
        _, y = to_px(lo[0], k); major = abs(k / 0.5 - round(k / 0.5)) < 1e-6
        dr.line([(0, y), (size, y)], fill=(90, 90, 90) if major else (50, 50, 50))
        if major: dr.text((2, y + 2), '%.1f' % k, fill=(255, 255, 0))
    return im, to_px

def views(path, out, points=None, size=620):
    m = J.parse_model(open(path, 'rb').read()); P, UV, T = TF.mesh(m)
    TT, UU = P[T], UV[T]; pad = 0.05
    lo, hi = P.min(0) - pad, P.max(0) + pad
    side, sp = raster(TT, UU, m['tex'], 2, 1, 0, 1, (lo[2], -hi[1]), (hi[2], -lo[1]), size, False)     # look from +X (left side)
    front, fp = raster(TT, UU, m['tex'], 0, 1, 2, 1, (lo[0], -hi[1]), (hi[0], -lo[1]), size, False)    # look from -Z (face)
    top, tp = raster(TT, UU, m['tex'], 0, 2, 1, 1, (lo[0], lo[2]), (hi[0], hi[2]), size, True)          # look from above
    if points:
        for name, (x, y, z) in points.items():
            for im, f, h, v in ((side, sp, z, -y), (front, fp, x, -y), (top, tp, x, z)):
                px, py = f(h, v); d = ImageDraw.Draw(im); d.ellipse([px - 4, py - 4, px + 4, py + 4], outline=(255, 0, 255), width=2); d.text((px + 5, py - 5), name, fill=(255, 0, 255))
    for im, t in ((side, 'SIDE  horiz=Z  vert=-Y (up)'), (front, 'FRONT horiz=X  vert=-Y'), (top, 'TOP   horiz=X  vert=Z')):
        ImageDraw.Draw(im).text((size // 2 - 80, size - 14), t, fill=(0, 255, 255))
    Image.fromarray(np.hstack([np.array(side), np.array(front), np.array(top)])).save(out)

if __name__ == '__main__':
    pts = json.load(open(sys.argv[sys.argv.index('--points') + 1])) if '--points' in sys.argv else None
    views(sys.argv[1], sys.argv[2], pts)
