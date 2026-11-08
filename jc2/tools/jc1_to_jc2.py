#!/usr/bin/env python3
"""Rebuild a JC2 beast on the JC1 family mesh (the JC1 way: one topology, one UV layout).

usage: jc1_to_jc2.py <jc1 start creature> <jc2 model.bin> <out.npz> [--no-subdiv]

1. take a JC1 family creature (all 49 share one mesh and UV layout), subdivide it once
2. warp it onto the JC2 beast using matched skeleton roles (legs, neck, head, tail)
3. wrap the visible part groups onto the JC2 surface (non-rigid ICP); hidden part
   groups follow the warp, as they do in JC1
4. bake the JC2 texture into the JC1 UV layout

Output npz: P (vertex positions in JC2 model units), N, UV, tris, part, joint, tex,
visibleParts, plus the JC1 start creature and JC2 file for reference.
"""
import sys, os, numpy as np
from scipy.interpolate import RBFInterpolator
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import jc1_family as F, jc2_gltf as J, template_fit as TF, rig_roles as RR

def subdivide(P, UV, tris, extra):
    """Linear 1->4 split; midpoints keyed by vertex-index pair, so UV seams stay seams."""
    P, UV = list(P), list(UV); ex = {k: list(v) for k, v in extra.items()}; mid = {}; out = []
    def m(a, b):
        k = (min(a, b), max(a, b))
        if k not in mid:
            mid[k] = len(P); P.append((np.asarray(P[a]) + P[b]) / 2); UV.append((np.asarray(UV[a]) + UV[b]) / 2)
            for kk in ex: ex[kk].append(ex[kk][a])            # joint / part from first endpoint
        return mid[k]
    for a, b, c in tris:
        ab, bc, ca = m(a, b), m(b, c), m(c, a)
        out += [(a, ab, ca), (ab, b, bc), (ca, bc, c), (ab, bc, ca)]
    return np.array(P, float), np.array(UV, float), np.array(out), {k: np.array(v) for k, v in ex.items()}

def jc1_roles(m, P):
    """JC1 rig bone roles as landmark point lists, matching rig_roles.roles() keys."""
    W = {k: v[1] for k, v in m['bones'].items()}; j = m['joint']
    def extreme(bones, axis, sign):
        sel = P[np.isin(j, bones)]; return sel[(sign * sel[:, axis]).argmax()]
    def foot(b): return extreme([b], 1, +1)                    # lowest point of the foot bone
    R = {'legFL': [W[3], W[4], foot(5)], 'legFR': [W[6], W[7], foot(8)],
         'legBL': [W[17], W[18], foot(19)], 'legBR': [W[20], W[21], foot(22)],
         'neck': [W[9], W[10]], 'head': [W[2], extreme([10, 11], 2, +1)],
         'tail_base': [W[16], W[23]], 'tail_tip': [W[23], extreme([24], 2, -1)]}
    return {k: np.array(v, float) for k, v in R.items()}

def pairs(RA, RB):
    A, B = [], []
    for k in RA:
        if k not in RB: continue
        pa, pb = RA[k], RB[k]['pts']
        if len(pa) == len(pb): A += list(pa); B += list(pb)
        else: A += [pa[0], pa[-1]]; B += [pb[0], pb[-1]]
    return np.array(A), np.array(B)

def convert(jc1_file, jc2_bytes, subdiv=True):
    m1 = F.load(jc1_file); B = J.parse_model(jc2_bytes)
    P, UV, tris = m1['pos'], m1['uv'], m1['tris']
    extra = dict(part=m1['part'], joint=m1['joint'])
    RA = jc1_roles(m1, P)                                       # roles before subdivision
    if subdiv: P, UV, tris, extra = subdivide(P, UV, tris, extra)
    part = extra['part']
    vis_groups = [0] + [g + 1 for g, f in enumerate(m1['partFlagsA']) if f >= 2]
    # target surface
    BP, BUV, Bt = TF.mesh(B); pts, uvs, nrm = TF.sample_surface(BP, BUV, Bt)
    RB = RR.roles(B); la, lb = pairs(RA, RB)
    # similarity (scale + translate) from the landmark pairs, then thin-plate residual warp
    s = np.sqrt(((lb - lb.mean(0)) ** 2).sum() / ((la - la.mean(0)) ** 2).sum())
    t = lb.mean(0) - s * la.mean(0)
    P0 = P * s + t; la0 = la * s + t
    warp = RBFInterpolator(la0, lb - la0, kernel='thin_plate_spline', smoothing=1e-3)
    Pw = P0 + warp(P0)
    # wrap the visible part of the mesh onto the JC2 surface
    tv = np.all(np.isin(part[tris], vis_groups), axis=1)
    used = np.unique(tris[tv]); remap = -np.ones(len(Pw), int); remap[used] = np.arange(len(used))
    Vsub = TF.fit(Pw[used], remap[tris[tv]], pts, nrm, pairs=None, prealigned=True)
    V = Pw.copy(); V[used] = Vsub
    tex = TF.bake(UV, tris[tv], V, pts, uvs, B['tex'], size=296, aspect=m1['tex'].shape[0] / m1['tex'].shape[1])
    d, _ = __import__('scipy.spatial', fromlist=['cKDTree']).cKDTree(V[used]).query(pts)
    return dict(P=V, UV=UV, tris=tris, part=part, joint=extra['joint'], tex=tex,
                visibleParts=np.array(vis_groups), cover_mean=d.mean(), cover_max=d.max(),
                size=np.ptp(BP, 0).max(), roles=sorted(set(RA) & set(RB)))

if __name__ == '__main__':
    r = convert(int(sys.argv[1]), open(sys.argv[2], 'rb').read(), '--no-subdiv' not in sys.argv)
    print('roles', r.pop('roles'), 'verts', len(r['P']), 'tris', len(r['tris']),
          'coverage mean %.4f max %.4f (size %.2f)' % (r['cover_mean'], r['cover_max'], r['size']))
    np.savez(sys.argv[3], **r)
