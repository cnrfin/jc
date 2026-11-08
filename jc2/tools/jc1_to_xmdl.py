#!/usr/bin/env python3
"""Convert a JC1 merge-family creature into a Jade Cocoon 2 model file.

usage: jc1_to_xmdl.py <jc1 family file> <template jc2 model.bin> <out.bin> [--height 2.0]

The JC1 bind pose (rest offsets + frame-0 rotations) becomes the JC2 node table; the
creature is scaled to JC2 beast height and turned to face the JC2 way and stood on y = 0. Every
vertex is rigid on its JC1 bone (JC1 has no blended skinning). Hidden part groups are
left out. The JC1 texture atlas is padded to 256x256.
"""
import sys, os, math, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import jc1_family as F, xmdl_write as XW

def convert(fid, height=2.0):
    m = F.load(fid); T = F.topology(); rig = T['rig']; fam = m['fam']
    P = m['pos']; s = height / np.ptp(P[:, 1])
    tu = 2 * math.pi / 4096
    ground = -(P[:, 1].max() * s)                      # JC2 beasts stand with their lowest point on y = 0
    nodes = [dict(parent=-1, T=(0, ground, 0), R=(0, math.pi, 0), S=(1, 1, 1), palette=False)]   # turn to face the JC2 way
    for b in range(rig['boneCount']):
        slf, par = rig['self'][b], rig['parent'][b]
        t = np.array(fam['root'] if slf == 1 else fam['rest'][slf], float) * s
        r = (0, 0, 0) if slf == 0 else tuple(a * tu for a in fam['rot0'][slf])
        nodes.append(dict(parent=0 if par == slf else par + 1, T=tuple(t), R=r, S=(1, 1, 1), palette=True))
    # vertices: JC1 world bind * scale, turned to face the JC2 way and dropped onto the ground
    Ry = np.diag([-1, 1, -1]); off = np.array([0, ground, 0])
    pos = (P * s) @ Ry.T + off; nrm = m['nrm'] @ Ry.T
    vis = np.all(np.isin(m['part'][m['tris']], [0] + [g + 1 for g, f in enumerate(m['partFlagsA']) if f >= 2]), 1)
    tris = m['tris'][vis]
    tex = m['tex']; h, w = tex.shape[:2]
    canvas = np.zeros((256, 256, 4), np.uint8); canvas[:h, :w] = tex
    uv = m['uv'] * [w / 256, h / 256]
    # check: rebuilt skeleton reproduces the JC1 bone frames (so rigid skinning is exact)
    W = XW.bind_world(nodes); err = 0
    for b in range(rig['boneCount']):
        R1, t1 = m['bones'][rig['self'][b]]
        err = max(err, np.abs(W[b + 1][:3, 3] - (Ry @ (t1 * s) + off)).max())
    return dict(nodes=nodes, pos=pos, nrm=nrm, uv=uv, bone=m['joint'] + 1, tris=tris, tex=canvas), err

if __name__ == '__main__':
    a = sys.argv[1:]; h = float(a[a.index('--height') + 1]) if '--height' in a else 2.0
    mesh, err = convert(int(a[0]), h)
    data = XW.build_model(mesh, open(a[1], 'rb').read())
    open(a[2], 'wb').write(data)
    print('nodes', len(mesh['nodes']), 'tris', len(mesh['tris']), 'bytes', len(data), 'skeleton check %.5f' % err)
