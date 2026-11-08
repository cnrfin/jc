#!/usr/bin/env python3
"""Option 2 merge prototype: keep the base beast's own mesh and texture, inherit from the material
beast through (1) skeleton proportions, (2) hue, (3) transplanted parts that keep their own texture.

usage: option2_merge.py <base.bin> <material.bin> <out.png>
"""
import sys, os, math, colorsys, numpy as np
from PIL import Image, ImageDraw
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import jc2_gltf as J, beast_parts as BP, rig_roles as RR

# ------------------------------------------------------------------ skeleton proportions
def bind_world(m): return [np.linalg.inv(x['IBM'].T) for x in m['joints']]

def path_nodes(m, top, eff):
    out, i = [], eff
    while i >= 0:
        out.append(i)
        if i == top: break
        i = m['nodes'][i]['parent']
    return out

def scaled_skeleton(m, scales, overall):
    """New bind world matrices: each node's local offset scaled by scales.get(node,1) * overall."""
    Wb = bind_world(m); out = [None] * len(Wb)
    for i, n in enumerate(m['nodes']):
        p = n['parent']
        L = Wb[i] if p < 0 else np.linalg.inv(Wb[p]) @ Wb[i]
        L = L.copy(); L[:3, 3] *= scales.get(i, 1.0) * overall
        out[i] = L if p < 0 else out[p] @ L
    return out

def body_size(R):
    L = [R[k]['length'] for k in ('legFL', 'legFR', 'legBL', 'legBR') if k in R]
    return float(np.mean(L)) if L else 1.0

def chain_scales(mA, mB, w):
    """Per-node length factors for matched legs (relative to each beast's mean leg), blended by w,
    plus the overall size factor. The hip's own offset is kept, so legs stay attached."""
    RA, RB = RR.roles(mA), RR.roles(mB); s = {}
    sA, sB = body_size(RA), body_size(RB)
    overall = 1 + w * (sB / sA - 1)
    Wb = bind_world(mA)
    for k in ('legFL', 'legFR', 'legBL', 'legBR'):
        if k in RA and k in RB:
            ratio = (RB[k]['length'] / sB) / (RA[k]['length'] / sA)
            f = 1 + w * (ratio - 1)
            eff = RA[k]['root'] + 3; top = eff
            while mA['nodes'][top]['parent'] >= 0 and not np.allclose(Wb[top][:3, 3], RA[k]['pts'][0], atol=1e-3):
                top = mA['nodes'][top]['parent']
            for nd in path_nodes(mA, top, eff):
                if nd != top: s[nd] = f
    return s, overall

def skin(P, W4, mA, Wnew):
    IB = [x['IBM'].T for x in mA['joints']]
    out = np.zeros_like(P)
    for i, p in enumerate(P):
        v = np.zeros(3)
        for j, w in W4[i]:
            if w: v += w * (Wnew[j] @ IB[j] @ np.append(p, 1))[:3]
        out[i] = v
    return out

# ------------------------------------------------------------------ colour
def mean_hue(tex):
    t = tex[..., :3].reshape(-1, 3) / 255.0; a = tex[..., 3].reshape(-1)
    t = t[a > 0]
    mx, mn = t.max(1), t.min(1); sat = (mx - mn) / (mx + 1e-6)
    h = np.array([colorsys.rgb_to_hsv(*c)[0] for c in t[::17]]); s = sat[::17]
    return math.atan2((s * np.sin(2 * np.pi * h)).sum(), (s * np.cos(2 * np.pi * h)).sum()) / (2 * np.pi)

def hue_rotate(tex, dh):
    img = Image.fromarray(tex[..., :3]).convert('HSV'); a = np.array(img)
    a[..., 0] = (a[..., 0].astype(int) + int(round(dh * 255))) % 256
    out = tex.copy(); out[..., :3] = np.array(Image.fromarray(a, 'HSV').convert('RGB')); return out

# ------------------------------------------------------------------ multi-texture renderer
def render(parts, yaw, size=260, frame=None, shade=False):
    c, s_ = math.cos(yaw), math.sin(yaw); Rm = np.array([[c, 0, s_], [0, 1, 0], [-s_, 0, c]])
    allV = np.concatenate([p[0].reshape(-1, 3) for p in parts]) @ Rm.T; allV[:, 1] *= -1
    lo, hi = (allV.min(0), allV.max(0)) if frame is None else frame
    sc = (size - 20) / max(hi[0] - lo[0], hi[1] - lo[1])
    img = np.full((size, size, 3), 30, np.uint8); zb = np.full((size, size), 1e9)
    for T, U, tex in parts:
        V = T @ Rm.T; V[..., 1] *= -1
        X = (V[..., 0] - (lo[0] + hi[0]) / 2) * sc + size / 2; Y = size / 2 - (V[..., 1] - (lo[1] + hi[1]) / 2) * sc; Z = V[..., 2]
        th, tw = tex.shape[:2]
        if shade:
            nrm = np.cross(V[:, 1] - V[:, 0], V[:, 2] - V[:, 0]); nrm /= np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-12
            L = np.array([-0.4, 0.75, -0.5]); L /= np.linalg.norm(L); lit = 0.45 + 0.55 * np.abs(nrm @ L)
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
            l2 = 1 - l0 - l1; m = (l0 >= 0) & (l1 >= 0) & (l2 >= 0)
            z = l0 * zs[0] + l1 * zs[1] + l2 * zs[2]; sub = zb[y0:y1, x0:x1]; m &= z < sub
            u = l0 * U[t, 0, 0] + l1 * U[t, 1, 0] + l2 * U[t, 2, 0]; v = l0 * U[t, 0, 1] + l1 * U[t, 1, 1] + l2 * U[t, 2, 1]
            px = np.clip((u % 1) * tw, 0, tw - 1).astype(int); py = np.clip((v % 1) * th, 0, th - 1).astype(int)
            col = tex[py[m], px[m], :3]
            if shade: col = (col * lit[t]).astype(np.uint8)
            sub[m] = z[m]; img[y0:y1, x0:x1][m] = col
    return img, (lo, hi)

# ------------------------------------------------------------------ the merge
def merge(mA, mB, w, swaps=(), recolor=True):
    PA, UVA, DA, W4A, TA = BP.mesh(mA)
    scales, overall = chain_scales(mA, mB, w)
    Wn = scaled_skeleton(mA, scales, overall)
    P = skin(PA, W4A, mA, Wn)
    tex = hue_rotate(mA['tex'], w * (((mean_hue(mB['tex']) - mean_hue(mA['tex']) + .5) % 1) - .5)) if recolor else mA['tex']
    keep = np.ones(len(TA), bool); parts = []
    if swaps:
        PB, UVB, DB, W4B, TB = BP.mesh(mB); WbB = bind_world(mB)
        for sw in swaps:
            # ('swap', a, b): replace A's subtree at a with B's subtree at b, frame a <- frame b
            # ('add',  a, b): keep A, attach B's subtree at b to A's node a, frame a <- frame parent(b)
            mode, a_node, b_node = sw[:3]
            sel = BP.part_tris(mB, DB, TB, [b_node])
            k = body_size(RR.roles(mA)) * overall / body_size(RR.roles(mB))
            if mode == 'swap':
                keep &= ~BP.part_tris(mA, DA, TA, [a_node]); src = WbB[b_node]
            else:                                           # translation + scale only, anchor to anchor
                b_anchor = sw[3] if len(sw) > 3 else mB['nodes'][b_node]['parent']
                M = np.eye(4); M[:3, :3] *= k; M[:3, 3] = Wn[a_node][:3, 3] - k * WbB[b_anchor][:3, 3]
                src = None
            if src is not None: M = Wn[a_node] @ np.diag([k, k, k, 1]) @ np.linalg.inv(src)
            Q = (M @ np.c_[PB, np.ones(len(PB))].T).T[:, :3]
            parts.append((Q[TB[sel]], UVB[TB[sel]], mB['tex']))
    return [(P[TA[keep]], UVA[TA[keep]], tex)] + parts

if __name__ == '__main__':
    mA = J.parse_model(open(sys.argv[1], 'rb').read()); mB = J.parse_model(open(sys.argv[2], 'rb').read())
    swaps = []
    for a in sys.argv[4:]:
        f = a.split(':'); swaps.append((f[0],) + tuple(int(v) for v in f[1:]))
    yaw = -math.pi * 0.62; S = 260
    def lab(img, t): im = Image.fromarray(img); ImageDraw.Draw(im).text((4, 4), t, fill=(255, 255, 0)); return np.array(im)
    PA, UVA, _, _, TA = BP.mesh(mA); PB, UVB, _, _, TB = BP.mesh(mB)
    row1 = [lab(render([(PA[TA], UVA[TA], mA['tex'])], yaw)[0], 'base A'), lab(render([(PB[TB], UVB[TB], mB['tex'])], yaw)[0], 'material B')]
    row2 = []
    for w in (0.25, 0.5, 0.75, 1.0):
        row2.append(lab(render(merge(mA, mB, w, swaps), yaw)[0], 'merge %d%%' % (w * 100)))
    pad = np.full((S, S * 2, 3), 30, np.uint8)
    Image.fromarray(np.vstack([np.hstack(row1 + [pad]), np.hstack(row2)])).save(sys.argv[3])
