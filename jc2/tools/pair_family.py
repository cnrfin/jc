#!/usr/bin/env python3
"""Build a two-member JC1-style merge family from two JC2 beasts.

usage: pair_family.py <A.bin> <B.bin> <out.npz> --a-only NODE[,NODE] --b-only NODE[,NODE]

The single mesh is A's own mesh plus B's unique parts (skeleton subtrees given by
--b-only). A's unique parts (--a-only) become a part group hidden on B, B's a group hidden
on A - JC1's arms/wings/legs rule. Only the shared body is morphed into B's shape:
  1. bone map A->B (legs by arc length, spine/head/tail chains, the rest follow ancestors)
  2. every shared vertex moves with its bones' matched positions (linear blend, translation
     only + body-size scale, since all beasts face -Z with -Y up)
  3. a gentle non-rigid snap onto B's body surface
Looks: A keeps its own texture/UVs; B's look uses per-corner UVs transferred from B's
original texture on the shared body, and B's own UVs on B's parts.
"""
import sys, os, collections, numpy as np
from scipy.spatial import cKDTree
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import jc2_gltf as J, beast_parts as BP, rig_roles as RR, template_fit as TF, corner_uv as CU

def bind_pos(m): return np.array([np.linalg.inv(x['IBM'].T)[:3, 3] for x in m['joints']])

def node_path(m, eff, w):
    Wb = bind_pos(m); path = [eff]; i = eff
    while m['nodes'][i]['parent'] >= 0:
        p = m['nodes'][i]['parent']
        if abs(Wb[p][0]) < 0.06 * w: break
        path.append(p); i = p
    return path[::-1]

def arclen(pos):
    s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(pos, axis=0), axis=1))]; return s / (s[-1] or 1)

def bone_map(mA, mB):
    RA, RB = RR.roles(mA), RR.roles(mB); pA, pB = bind_pos(mA), bind_pos(mB)
    wA = np.ptp([p for b in mA['batches'] for p in b['pos']], 0)[0]; wB = np.ptp([p for b in mB['batches'] for p in b['pos']], 0)[0]
    M = {0: 0, 1: 1, 2: 2}
    for k in RA:
        if k not in RB: continue
        if k.startswith('leg'):
            na, nb = node_path(mA, RA[k]['root'] + 3, wA), node_path(mB, RB[k]['root'] + 3, wB)
            sa, sb = arclen(pA[na]), arclen(pB[nb])
            for i, n in enumerate(na): M[n] = nb[int(np.argmin(np.abs(sb - sa[i])))]
        else:
            ra, rb = RA[k]['root'], RB[k]['root']
            for d in range(3): M[ra + d] = rb + d
    def follow(i):
        while i not in M: i = mA['nodes'][i]['parent']
        return M[i]
    return {i: follow(i) for i in range(len(mA['nodes']))}, RR.body_size(RA) if hasattr(RR, 'body_size') else None

def align(a, b):
    """Rotation taking direction a onto direction b."""
    a = a / (np.linalg.norm(a) or 1); b = b / (np.linalg.norm(b) or 1)
    v = np.cross(a, b); c = float(np.dot(a, b))
    if np.linalg.norm(v) < 1e-8: return np.eye(3) if c > 0 else -np.eye(3) + 2 * np.outer([1, 0, 0], [1, 0, 0])
    K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + K + K @ K * (1 / (1 + c))

def joint_xforms(mSrc, mDst, jmap, pS, pD, s):
    """Per source joint: rotation aligning its bone (joint -> furthest mapped descendant joint) to the matched bone."""
    ch = collections.defaultdict(list)
    for i, n in enumerate(mSrc['nodes']):
        if n['parent'] >= 0: ch[n['parent']].append(i)
    X = {}
    for j in range(len(mSrc['nodes'])):
        R = np.eye(3); best = 0
        for c in ch[j]:
            dS = pS[c] - pS[j]; dD = pD[jmap[c]] - pD[jmap[j]]
            if np.linalg.norm(dS) > best and np.linalg.norm(dS) > 1e-4 and np.linalg.norm(dD) > 1e-4:
                best = np.linalg.norm(dS); R = align(dS, dD)
        X[j] = R
    return X

def body_size(m):
    R = RR.roles(m); L = [R[k]['length'] for k in R if k.startswith('leg')]; return float(np.mean(L)) if L else 1.0

def build(A_bytes, B_bytes, a_only, b_only, a_head=(), b_head=()):
    mA, mB = J.parse_model(A_bytes), J.parse_model(B_bytes)
    PA, UVA, DA, W4A, TA = BP.mesh(mA); PB, UVB, DB, W4B, TB = BP.mesh(mB)
    pA, pB = bind_pos(mA), bind_pos(mB)
    bmap, _ = bone_map(mA, mB); s = body_size(mB) / body_size(mA)
    a_part = BP.part_tris(mA, DA, TA, a_only) if a_only else np.zeros(len(TA), bool)
    b_part = BP.part_tris(mB, DB, TB, b_only) if b_only else np.zeros(len(TB), bool)
    a_hd = BP.part_tris(mA, DA, TA, a_head) if a_head else np.zeros(len(TA), bool)
    b_hd = BP.part_tris(mB, DB, TB, b_head) if b_head else np.zeros(len(TB), bool)
    a_part |= a_hd; b_part |= b_hd
    XA = joint_xforms(mA, mB, bmap, pA, pB, s)
    # --- shared body of A in B's shape: translation-only skinning through the bone map
    PAB = np.zeros_like(PA)
    for i, p in enumerate(PA):
        v = np.zeros(3); tw = 0
        for j, w in W4A[i]:
            if w: v += w * (pB[bmap[j]] + s * (p - pA[j])); tw += w
        PAB[i] = v / (tw or 1)
    # gentle snap of the shared body onto B's body surface
    RA_ = RR.roles(mA); wA_ = np.ptp(PA, 0)[0]; legn = set()
    for k in RA_:
        if k.startswith('leg'):
            for n in node_path(mA, RA_[k]['root'] + 3, wA_): legn |= BP.subtree(mA, n)
    leg_v = np.isin(DA, list(legn))
    body_tris = TA[~a_part]; used = np.unique(body_tris)
    remap = -np.ones(len(PA), int); remap[used] = np.arange(len(used))
    Bb = TB[~b_part]; pts, uvs, nrm = TF.sample_surface(PB, UVB, Bb, n=60000)
    snapped = TF.fit(PAB[used], remap[body_tris], pts, nrm, prealigned=True, iters=((30, 3), (10, 3), (3, 4)))
    PAB[used] = snapped
    # --- A's own parts in B's shape: ride their anchor bone's matched position
    #     (already done by the bone-map skinning above; they stay attached)
    # --- B's parts: B shape = B's own geometry; A shape = carried back through the inverse bone offset
    inv = {}
    for a, b in bmap.items(): inv.setdefault(b, a)
    PBA = np.zeros_like(PB)
    for i, p in enumerate(PB):
        v = np.zeros(3); tw = 0
        for j, w in W4B[i]:
            if not w: continue
            jj = j
            while jj not in inv: jj = mB['nodes'][jj]['parent']
            a = inv[jj]; v += w * (pA[a] + (p - pB[jj]) / s); tw += w
        PBA[i] = v / (tw or 1)
    # --- heads move as rigid pieces, head root onto head root (no per-bone stretching)
    if a_head and b_head:
        def neck_ring(P, T, hd):
            """Where the head meets the body: head vertices close to non-head vertices of the same beast."""
            hv = np.unique(T[hd]); bv = np.setdiff1d(np.unique(T[~hd]), hv)
            d, _ = cKDTree(P[bv]).query(P[hv]); size = np.ptp(P, 0).max()
            ring = hv[d < 0.03 * size]
            return P[ring].mean(0) if len(ring) else P[hv].mean(0)
        attA, attB = neck_ring(PA, TA, a_hd), neck_ring(PB, TB, b_hd)
        # where A's neck ring lands in B's shape (moves with the body), and vice versa
        attA_inB = attB; attB_inA = attA
        va = np.unique(TA[a_hd]); PAB[va] = attA_inB + s * (PA[va] - attA)
        vb = np.unique(TB[b_hd]); PBA[vb] = attB_inA + (PB[vb] - attB) / s
    # --- union mesh: A vertices then B-part vertices
    nA = len(PA); bt = TB[b_part]; bused = np.unique(bt); bre = -np.ones(len(PB), int); bre[bused] = nA + np.arange(len(bused))
    shapeA = np.r_[PA, PBA[bused]]; shapeB = np.r_[PAB, PB[bused]]
    tris = np.r_[TA, bre[bt]]
    group = np.r_[np.where(a_hd, 3, np.where(a_part, 1, 0)), np.where(b_hd[b_part], 4, 2)]   # 0 shared 1 A-only 2 B-only 3 A-head 4 B-head
    # --- looks: corner UVs + texture id per triangle (0 = A tex, 1 = B tex)
    lookA_uv = np.r_[UVA[TA], UVB[bt]]; lookA_tex = np.r_[np.zeros(len(TA), int), np.ones(len(bt), int)]
    # B look on the shared body: transfer from B's body triangles only
    cuvB = corner_transfer(PAB, TA[~a_part], PB, UVB, TB[~b_part])
    lookB_uv = np.r_[UVA[TA], UVB[bt]].copy(); lookB_tex = lookA_tex.copy()
    lookB_uv[np.where(~a_part)[0]] = cuvB; lookB_tex[np.where(~a_part)[0]] = 1
    return dict(shapeA=shapeA, shapeB=shapeB, tris=tris, group=group, lookA_uv=lookA_uv, lookA_tex=lookA_tex,
                lookB_uv=lookB_uv, lookB_tex=lookB_tex, texA=mA['tex'], texB=mB['tex'])

def corner_transfer(V, tris, P, UV, T, k=12):
    """Per-corner UVs for triangles (V, tris) from the surface (P, UV, T), island-consistent."""
    isl = CU.islands(P, UV, T); cent = P[T].mean(1); tree = cKDTree(cent)
    c = V[tris].mean(1); _, cand = tree.query(c, k=k)
    best_t = np.zeros(len(tris), int); bestd = np.full(len(tris), np.inf)
    for j in range(k):
        t = cand[:, j]; _, dd = CU.closest_on_tris(c, P[T[t, 0]], P[T[t, 1]], P[T[t, 2]])
        m_ = dd < bestd; bestd[m_] = dd[m_]; best_t[m_] = t[m_]
    ti = isl[best_t]; out = np.zeros((len(tris), 3, 2))
    for I in np.unique(ti):
        sel = np.where(ti == I)[0]; own = np.where(isl == I)[0]; tr = cKDTree(cent[own]); kk = min(k, len(own))
        for corner in range(3):
            p = V[tris[sel, corner]]; _, cc = tr.query(p, k=kk); cc = cc.reshape(len(sel), -1)
            bu = np.zeros((len(sel), 2)); bd = np.full(len(sel), np.inf)
            for j in range(cc.shape[1]):
                t = own[cc[:, j]]; b, dd = CU.closest_on_tris(p, P[T[t, 0]], P[T[t, 1]], P[T[t, 2]])
                uv = b[:, :1] * UV[T[t, 0]] + b[:, 1:2] * UV[T[t, 1]] + b[:, 2:] * UV[T[t, 2]]
                m_ = dd < bd; bd[m_] = dd[m_]; bu[m_] = uv[m_]
            out[sel, corner] = bu
    return out

if __name__ == '__main__':
    args = sys.argv[1:]
    def opt(name):
        if name in args: v = args[args.index(name) + 1]; return [int(x) for x in v.split(',') if x]
        return []
    r = build(open(args[0], 'rb').read(), open(args[1], 'rb').read(), opt('--a-only'), opt('--b-only'), opt('--a-head'), opt('--b-head'))
    np.savez(args[2], **r); print('vertices', len(r['shapeA']), 'triangles', len(r['tris']), 'groups', np.bincount(r['group']))
