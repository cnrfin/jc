#!/usr/bin/env python3
"""render_anim.py <model.bin> <clip> <out.png> : side view at 6 frames, skinned in Python."""
import sys, math, numpy as np
from PIL import Image
sys.path.insert(0, __file__.rsplit('/', 1)[0])
import jc2_gltf as J, render_tex as RT

def pose(m, clip, fr):
    nodes, joints = m['nodes'], m['joints']
    Tl, Rl, Sl = J.base_pose(m)
    for node, comps in clip['chans'].items():
        for c, fn in comps.items():
            (Tl if c < 3 else Rl if c < 6 else Sl)[node][c % 3] = fn(fr)
    W = J.world(nodes, joints, Tl, Rl, Sl)
    return [W[j] @ joints[j]['IBM'].T for j in range(len(joints))]

def skinned_tris(m, clip, fr):
    K = pose(m, clip, fr); T, U = [], []
    for b in m['batches']:
        Ps = []
        for k, p in enumerate(b['pos']):
            v = np.zeros(3); w = np.array(b['wgt'][k], float) / 255
            for s, ww in zip(b['palette'], w):
                if ww: v += ww * (K[s] @ np.append(p, 1))[:3] if False else ww * (K[pal[s]] @ np.append(p, 1))[:3]
            Ps.append(v)
        for k in range(len(Ps) - 2):
            i = (k, k + 1, k + 2) if k % 2 == 0 else (k + 1, k, k + 2)
            T.append([Ps[x] for x in i]); U.append([b['uv'][x] for x in i])
    return np.array(T), np.array(U)

if __name__ == '__main__':
    d = open(sys.argv[1], 'rb').read(); m = J.parse_model(d)
    pal = [j for j, X in enumerate(m['joints']) if X['flags'] & 1]
    clip = J.parse_clips(m['anim'])[int(sys.argv[2])]
    F = clip['frames']; imgs = []
    for fr in np.linspace(0, F, 6):
        T, U = skinned_tris(m, clip, fr)
        imgs.append(RT.render(T, U, m['tex'], math.pi / 2, False, size=260))
    Image.fromarray(np.hstack(imgs)).save(sys.argv[3]); print('frames', F)
