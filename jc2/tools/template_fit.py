#!/usr/bin/env python3
"""Fit a template beast's mesh onto another beast (non-rigid ICP), so both share one topology.

usage: template_fit.py <template.bin> <target.bin> <out.npz>

Output npz: V (fitted vertex positions, template order), tex (target texture baked into
the template's UV layout), plus the template's own data for blending.
"""
import sys, math, numpy as np
from scipy.spatial import cKDTree
import scipy.sparse as sp
import scipy.sparse.linalg as spla
sys.path.insert(0, __file__.rsplit('/', 1)[0])
import jc2_gltf as J

def mesh(m):
    P, UV, tris = [], [], []
    for b in m['batches']:
        base = len(P); P += b['pos']; UV += b['uv']
        for k in range(len(b['pos']) - 2):
            t = (base + k, base + k + 1, base + k + 2) if k % 2 == 0 else (base + k + 1, base + k, base + k + 2)
            if len({tuple(P[t[0]]), tuple(P[t[1]]), tuple(P[t[2]])}) == 3: tris.append(t)
    return np.array(P, float), np.array(UV, float), np.array(tris)

def weld(P, eps=1e-5):
    """Map each vertex to a welded index (strip batches duplicate seam vertices)."""
    key = np.round(P / eps).astype(np.int64)
    _, inv = np.unique(key, axis=0, return_inverse=True)
    return inv.reshape(-1)

def sample_surface(P, UV, tris, n=80000, seed=0):
    rng = np.random.default_rng(seed)
    a, b, c = P[tris[:, 0]], P[tris[:, 1]], P[tris[:, 2]]
    area = np.linalg.norm(np.cross(b - a, c - a), axis=1) / 2
    pick = rng.choice(len(tris), n, p=area / area.sum())
    r1, r2 = rng.random(n), rng.random(n); s = np.sqrt(r1)
    w = np.stack([1 - s, s * (1 - r2), s * r2], 1)
    T = tris[pick]
    pts = (P[T] * w[..., None]).sum(1); uvs = (UV[T] * w[..., None]).sum(1)
    nrm = np.cross(b - a, c - a)[pick]; nrm /= np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-12
    return pts, uvs, nrm

def vertex_normals(V, tris):
    N = np.zeros_like(V); f = np.cross(V[tris[:, 1]] - V[tris[:, 0]], V[tris[:, 2]] - V[tris[:, 0]])
    for k in range(3): np.add.at(N, tris[:, k], f)
    return N / (np.linalg.norm(N, axis=1, keepdims=True) + 1e-12)

def laplacian(nv, tris):
    e = np.concatenate([tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]])
    e = np.unique(np.sort(e, 1), axis=0)
    A = sp.coo_matrix((np.ones(len(e) * 2), (np.r_[e[:, 0], e[:, 1]], np.r_[e[:, 1], e[:, 0]])), shape=(nv, nv)).tocsr()
    A.data[:] = 1
    deg = np.asarray(A.sum(1)).ravel(); deg[deg == 0] = 1
    return sp.eye(nv) - sp.diags(1 / deg) @ A

def landmarks(P):
    """Generic quadruped landmarks: 4 feet (ground contact by quadrant), nose, tail, top, centre."""
    h = np.ptp(P[:, 1]); g = P[P[:, 1] > -0.04 * h]
    zc = np.median(g[:, 2]); out = []
    for sx in (1, -1):
        for front in (True, False):
            q = g[(np.sign(g[:, 0] + 1e-9) == sx) & ((g[:, 2] >= zc) == front)]
            out.append(q.mean(0) if len(q) else np.array([sx * np.ptp(P[:, 0]) / 4, 0, zc]))
    out += [P[P[:, 2].argmax()], P[P[:, 2].argmin()], P[P[:, 1].argmin()], P.mean(0)]
    return np.array(out)

def fit(A_P, A_tris, B_pts, B_nrm, pairs=None, iters=((40, 4), (12, 4), (4, 4), (1.5, 4), (0.5, 6))):
    wid = weld(A_P); nW = wid.max() + 1
    Vw = np.zeros((nW, 3)); np.add.at(Vw, wid, A_P); cnt = np.bincount(wid, minlength=nW); Vw /= cnt[:, None]
    Tw = wid[A_tris]; Tw = Tw[(Tw[:, 0] != Tw[:, 1]) & (Tw[:, 1] != Tw[:, 2]) & (Tw[:, 0] != Tw[:, 2])]
    L = laplacian(nW, Tw)
    # initial alignment: map template bbox onto target bbox (both stand on y=0 facing +z)
    from scipy.interpolate import RBFInterpolator
    la, lb = landmarks(Vw)[4:], landmarks(B_pts)[4:]                 # nose, tail, top, centre
    if pairs is not None and len(pairs[0]):
        la, lb = np.r_[la, pairs[0]], np.r_[lb, pairs[1]]            # skeleton roles: legs, neck, head, tail
    warp = RBFInterpolator(la, lb - la, kernel='thin_plate_spline', smoothing=1e-3)
    V = Vw + warp(Vw)                                                 # landmarks onto landmarks
    treeB = cKDTree(B_pts)
    for stiff, n in iters:
        for _ in range(n):
            delta = L @ V
            N = vertex_normals(V, Tw)
            d, j = treeB.query(V)
            ok = (np.einsum('ij,ij->i', N, B_nrm[j]) > 0.0)
            w = ok.astype(float)
            # reverse term: every target sample pulls its nearest template vertex (covers protrusions)
            treeV = cKDTree(V); d2, i2 = treeV.query(B_pts)
            acc = np.zeros_like(V); cw = np.zeros(nW)
            np.add.at(acc, i2, B_pts); np.add.at(cw, i2, 1)
            rev = cw > 0; revT = np.zeros_like(V); revT[rev] = acc[rev] / cw[rev, None]
            wr = np.where(rev, 0.5, 0.0)
            Wd = sp.diags(w + wr)
            M = (stiff * (L.T @ L) + Wd).tocsc()
            rhs = stiff * (L.T @ delta) + (w[:, None] * B_pts[j] + wr[:, None] * revT)
            solve = spla.factorized(M)
            V = np.stack([solve(rhs[:, k]) for k in range(3)], 1)
    return V[wid]

def bake(A_UV, A_tris, Vfit, B_pts, B_uvs, B_tex, size=256):
    """Bake target texture into template UV layout: texel -> fitted 3D point -> nearest target point -> target UV."""
    tree = cKDTree(B_pts); out = np.zeros((size, size, 4), np.uint8); filled = np.zeros((size, size), bool)
    th, tw = B_tex.shape[:2]
    for t in A_tris:
        uv = A_UV[t] * size; P3 = Vfit[t]
        x0, y0 = np.floor(uv.min(0)).astype(int); x1, y1 = np.ceil(uv.max(0)).astype(int)
        x0, y0 = max(x0, 0), max(y0, 0); x1, y1 = min(x1, size - 1), min(y1, size - 1)
        if x1 < x0 or y1 < y0: continue
        gx, gy = np.meshgrid(np.arange(x0, x1 + 1) + .5, np.arange(y0, y1 + 1) + .5)
        (ax, ay), (bx, by), (cx, cy) = uv
        den = (by - cy) * (ax - cx) + (cx - bx) * (ay - cy)
        if abs(den) < 1e-12: continue
        l0 = ((by - cy) * (gx - cx) + (cx - bx) * (gy - cy)) / den
        l1 = ((cy - ay) * (gx - cx) + (ax - cx) * (gy - cy)) / den
        l2 = 1 - l0 - l1
        m = (l0 >= -0.02) & (l1 >= -0.02) & (l2 >= -0.02)
        if not m.any(): continue
        pts = l0[m, None] * P3[0] + l1[m, None] * P3[1] + l2[m, None] * P3[2]
        _, j = tree.query(pts); u = B_uvs[j]
        px = np.clip((u[:, 0] % 1) * tw, 0, tw - 1).astype(int); py = np.clip((u[:, 1] % 1) * th, 0, th - 1).astype(int)
        yy, xx = gy[m].astype(int), gx[m].astype(int)
        out[yy, xx] = B_tex[py, px]; filled[yy, xx] = True
    # dilate to hide seams
    for _ in range(4):
        sh = [np.roll(out, s, a) for a in (0, 1) for s in (1, -1)]; fs = [np.roll(filled, s, a) for a in (0, 1) for s in (1, -1)]
        for o, f in zip(sh, fs):
            m = (~filled) & f; out[m] = o[m]; filled |= m
    return out

if __name__ == '__main__':
    A = J.parse_model(open(sys.argv[1], 'rb').read()); B = J.parse_model(open(sys.argv[2], 'rb').read())
    AP, AUV, At = mesh(A); BP, BUV, Bt = mesh(B)
    pts, uvs, nrm = sample_surface(BP, BUV, Bt)
    import rig_roles
    la, lb, used = rig_roles.landmark_pairs(A, B); print('matched roles:', used)
    V = fit(AP, At, pts, nrm, (la, lb))
    d, _ = cKDTree(pts).query(V); dB, _ = cKDTree(V).query(pts)
    print('template->target mean %.4f max %.4f | target->template mean %.4f max %.4f (target size %.2f)' % (d.mean(), d.max(), dB.mean(), dB.max(), np.ptp(BP, 0).max()))
    tex = bake(AUV, At, V, pts, uvs, B['tex'])
    np.savez(sys.argv[3], V=V, tex=tex)
