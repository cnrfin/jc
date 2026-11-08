#!/usr/bin/env python3
"""Tiny software renderer for checking JC2 meshes: render.py <model.bin> <out.png>"""
import sys, math, numpy as np
from PIL import Image
sys.path.insert(0, __file__.rsplit('/', 1)[0]); import xmdl

def tris_of(model_bytes):
    g = xmdl.geometry(model_bytes); ns, end = xmdl.nodes(g)
    T = []
    for b in xmdl.batches(g, end):
        P = b['pos']
        for k in range(len(P) - 2):
            a, c = (P[k], P[k + 1]) if k % 2 == 0 else (P[k + 1], P[k])
            T.append((a, c, P[k + 2]))
    return np.array(T, dtype=float)

def render(T, yaw, size=360):
    c, s = math.cos(yaw), math.sin(yaw)
    R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    V = T @ R.T
    V[..., 1] *= -1                                  # model is Y-down
    lo, hi = V.reshape(-1, 3).min(0), V.reshape(-1, 3).max(0)
    sc = (size - 20) / max(hi[0] - lo[0], hi[1] - lo[1])
    X = (V[..., 0] - (lo[0] + hi[0]) / 2) * sc + size / 2
    Y = size / 2 - (V[..., 1] - (lo[1] + hi[1]) / 2) * sc
    Z = V[..., 2]
    img = np.full((size, size), 0.12); zb = np.full((size, size), 1e9)
    n = np.cross(V[:, 1] - V[:, 0], V[:, 2] - V[:, 0]); n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
    shade = 0.25 + 0.75 * np.abs(n @ np.array([0.3, 0.5, -0.8]) / np.linalg.norm([0.3, 0.5, -0.8]))
    for t in range(len(T)):
        xs, ys, zs = X[t], Y[t], Z[t]
        x0, x1 = int(max(0, xs.min())), int(min(size - 1, xs.max())) + 1
        y0, y1 = int(max(0, ys.min())), int(min(size - 1, ys.max())) + 1
        if x1 <= x0 or y1 <= y0: continue
        gx, gy = np.meshgrid(np.arange(x0, x1) + .5, np.arange(y0, y1) + .5)
        d = (ys[1] - ys[2]) * (xs[0] - xs[2]) + (xs[2] - xs[1]) * (ys[0] - ys[2])
        if abs(d) < 1e-9: continue
        l0 = ((ys[1] - ys[2]) * (gx - xs[2]) + (xs[2] - xs[1]) * (gy - ys[2])) / d
        l1 = ((ys[2] - ys[0]) * (gx - xs[2]) + (xs[0] - xs[2]) * (gy - ys[2])) / d
        l2 = 1 - l0 - l1
        m = (l0 >= 0) & (l1 >= 0) & (l2 >= 0)
        z = l0 * zs[0] + l1 * zs[1] + l2 * zs[2]
        sub = zb[y0:y1, x0:x1]; m &= z < sub
        sub[m] = z[m]; img[y0:y1, x0:x1][m] = shade[t]
    return (img * 255).astype(np.uint8)

if __name__ == '__main__':
    T = tris_of(open(sys.argv[1], 'rb').read())
    views = [render(T, a) for a in (0, math.pi / 2, math.pi / 4 * 3)]
    Image.fromarray(np.hstack(views)).save(sys.argv[2])
    print(len(T), 'triangles')
