#!/usr/bin/env python3
"""Split a JC2 beast into parts by skeleton subtree (dominant skin joint per vertex)."""
import sys, os, collections, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import jc2_gltf as J

def subtree(m, r):
    ch = collections.defaultdict(list)
    for i, n in enumerate(m['nodes']):
        if n['parent'] >= 0: ch[n['parent']].append(i)
    out, st = [], [r]
    while st: x = st.pop(); out.append(x); st += ch[x]
    return set(out)

def mesh(m):
    """Triangles plus per-vertex dominant joint (node index)."""
    pal = [j for j, x in enumerate(m['joints']) if x['flags'] & 1]
    P, UV, D, W4, tris = [], [], [], [], []
    for b in m['batches']:
        base = len(P)
        for k in range(len(b['pos'])):
            P.append(b['pos'][k]); UV.append(b['uv'][k]); w = np.array(b['wgt'][k], float)
            D.append(pal[b['palette'][int(np.argmax(w))]])
            W4.append([(pal[b['palette'][i]] if i < len(b['palette']) else 0, w[i] / 255) for i in range(4)])
        for k in range(len(b['pos']) - 2):
            t = (base + k, base + k + 1, base + k + 2) if k % 2 == 0 else (base + k + 1, base + k, base + k + 2)
            tris.append(t)
    return np.array(P, float), np.array(UV, float), np.array(D), W4, np.array(tris)

def part_tris(m, D, tris, roots):
    nodes = set().union(*[subtree(m, r) for r in roots])
    return np.all(np.isin(D[tris], list(nodes)), axis=1)
