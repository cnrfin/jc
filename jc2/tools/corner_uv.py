#!/usr/bin/env python3
"""Give a rebuilt (shared-mesh) beast per-triangle-corner UVs into its ORIGINAL JC2 texture.

usage: corner_uv.py <rebuilt.npz> <jc2 model.bin> <out.npz>

No baking: every corner takes the UV of the nearest point on the original surface, and
all three corners of a triangle are taken from the same UV island of the original, so a
triangle never straddles an atlas seam. The original texture is used pixel for pixel.
"""
import sys, os, numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import jc2_gltf as J, template_fit as TF

def islands(P, UV, tris):
    """Connected components of triangles that share a vertex with the same position AND uv."""
    key = np.round(np.c_[P, UV] * 1e4).astype(np.int64)
    _, wid = np.unique(key, axis=0, return_inverse=True); wid = wid.ravel()
    nt = len(tris); rows = np.repeat(np.arange(nt), 3); cols = wid[tris].ravel()
    A = coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(nt, wid.max() + 1)).tocsr()
    G = A @ A.T
    return connected_components(G, directed=False)[1]

def closest_on_tris(p, A, B, C):
    """Closest point on each triangle (vectorised, Ericson); returns barycentric (u,v,w) and distance^2."""
    ab, ac, ap = B - A, C - A, p - A
    d1, d2 = (ab * ap).sum(1), (ac * ap).sum(1)
    bp = p - B; d3, d4 = (ab * bp).sum(1), (ac * bp).sum(1)
    cp = p - C; d5, d6 = (ab * cp).sum(1), (ac * cp).sum(1)
    va = d3 * d6 - d5 * d4; vb = d5 * d2 - d1 * d6; vc = d1 * d4 - d3 * d2
    den = np.where(np.abs(va + vb + vc) < 1e-20, 1e-20, va + vb + vc)
    v = vb / den; w = vc / den; bary = np.stack([1 - v - w, v, w], 1)
    # regions outside the face: clamp by projecting onto edges / vertices
    def seg(Pa, Pb, ia, ib):
        e = Pb - Pa; t = np.clip(((p - Pa) * e).sum(1) / np.maximum((e * e).sum(1), 1e-20), 0, 1)
        b = np.zeros((len(p), 3)); b[np.arange(len(p)), ia] = 1 - t; b[np.arange(len(p)), ib] = t; return b
    out = (bary.min(1) < 0)
    if out.any():
        cands = [seg(A, B, 0, 1), seg(B, C, 1, 2), seg(C, A, 2, 0)]
        best = None; bd = None
        for b in cands:
            q = b[:, :1] * A + b[:, 1:2] * B + b[:, 2:] * C; dd = ((q - p) ** 2).sum(1)
            if best is None: best, bd = b, dd
            else: m = dd < bd; best[m], bd[m] = b[m], dd[m]
        bary[out] = best[out]
    q = bary[:, :1] * A + bary[:, 1:2] * B + bary[:, 2:] * C
    return bary, ((q - p) ** 2).sum(1)

def transfer(Z, m, k=12):
    P, UV, T = TF.mesh(m)
    isl = islands(P, UV, T)
    cent = P[T].mean(1); tree = cKDTree(cent)
    vis = np.all(np.isin(Z['part'][Z['tris']], Z['visibleParts']), 1)
    tris = Z['tris'][vis]; V = Z['P']
    # 1) island per shared triangle: the original triangle closest to its centroid
    c = V[tris].mean(1)
    _, cand = tree.query(c, k=k)
    best_t = np.zeros(len(tris), int); bestd = np.full(len(tris), np.inf)
    for j in range(k):
        t = cand[:, j]; _, dd = closest_on_tris(c, P[T[t, 0]], P[T[t, 1]], P[T[t, 2]])
        m_ = dd < bestd; bestd[m_] = dd[m_]; best_t[m_] = t[m_]
    tri_island = isl[best_t]
    # 2) each corner: nearest point on the original surface, restricted to that island
    CUV = np.zeros((len(tris), 3, 2))
    for I in np.unique(tri_island):
        sel = np.where(tri_island == I)[0]; own = np.where(isl == I)[0]
        tr = cKDTree(cent[own]); kk = min(k, len(own))
        for corner in range(3):
            p = V[tris[sel, corner]]
            _, cc = tr.query(p, k=kk); cc = cc.reshape(len(sel), -1)
            bu = np.zeros((len(sel), 2)); bd = np.full(len(sel), np.inf)
            for j in range(cc.shape[1]):
                t = own[cc[:, j]]; b, dd = closest_on_tris(p, P[T[t, 0]], P[T[t, 1]], P[T[t, 2]])
                uv = b[:, :1] * UV[T[t, 0]] + b[:, 1:2] * UV[T[t, 1]] + b[:, 2:] * UV[T[t, 2]]
                m_ = dd < bd; bd[m_] = dd[m_]; bu[m_] = uv[m_]
            CUV[sel, corner] = bu
    return tris, CUV, np.sqrt(bestd)

if __name__ == '__main__':
    Z = dict(np.load(sys.argv[1])); m = J.parse_model(open(sys.argv[2], 'rb').read())
    tris, CUV, dist = transfer(Z, m)
    print('triangles', len(tris), 'centroid-to-surface mean %.4f max %.4f' % (dist.mean(), dist.max()))
    Z.update(ctris=tris, cuv=CUV, otex=m['tex']); np.savez(sys.argv[3], **Z)
