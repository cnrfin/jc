#!/usr/bin/env python3
"""Textured preview: render_tex.py <model.bin> <out.png> [flipv]"""
import sys, math, numpy as np
from PIL import Image
sys.path.insert(0, __file__.rsplit('/', 1)[0]); import xmdl, tim2

def load(d):
    g = xmdl.geometry(d); ns, end = xmdl.nodes(g)
    T, U = [], []
    for b in xmdl.batches(g, end):
        P, UV = b['pos'], b['uv']
        for k in range(len(P) - 2):
            i = (k, k + 1, k + 2) if k % 2 == 0 else (k + 1, k, k + 2)
            T.append([P[j] for j in i]); U.append([UV[j] for j in i])
    tex = [tim2.decode(s)[0] for s in xmdl.sections(d) if s[:4] == b'TIM2'][0]
    return np.array(T, float), np.array(U, float), tex

def render(T, U, tex, yaw, flipv, size=400):
    c, s = math.cos(yaw), math.sin(yaw)
    V = T @ np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]]).T; V[..., 1] *= -1
    lo, hi = V.reshape(-1, 3).min(0), V.reshape(-1, 3).max(0)
    sc = (size - 20) / max(hi[0] - lo[0], hi[1] - lo[1])
    X = (V[..., 0] - (lo[0] + hi[0]) / 2) * sc + size / 2
    Y = size / 2 - (V[..., 1] - (lo[1] + hi[1]) / 2) * sc
    Z = V[..., 2]
    th, tw = tex.shape[:2]
    img = np.full((size, size, 3), 30, np.uint8); zb = np.full((size, size), 1e9)
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
        u = l0 * U[t, 0, 0] + l1 * U[t, 1, 0] + l2 * U[t, 2, 0]
        v = l0 * U[t, 0, 1] + l1 * U[t, 1, 1] + l2 * U[t, 2, 1]
        if flipv: v = 1 - v
        px = np.clip((u % 1) * tw, 0, tw - 1).astype(int); py = np.clip((v % 1) * th, 0, th - 1).astype(int)
        sub[m] = z[m]; img[y0:y1, x0:x1][m] = tex[py[m], px[m], :3]
    return img

if __name__ == '__main__':
    T, U, tex = load(open(sys.argv[1], 'rb').read())
    flip = len(sys.argv) > 3
    Image.fromarray(np.hstack([render(T, U, tex, a, flip) for a in (0.3, math.pi / 2, math.pi * 0.8)])).save(sys.argv[2])
