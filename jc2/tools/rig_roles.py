#!/usr/bin/env python3
"""Classify a JC2 rig's IK chains into roles (legs, head line, tail line, side appendages)."""
import sys, numpy as np
sys.path.insert(0, __file__.rsplit('/', 1)[0]); import jc2_gltf as J

def chains(m):
    Wb = [np.linalg.inv(x['IBM'].T)[:3, 3] for x in m['joints']]
    out = []
    for i, n in enumerate(m['nodes']):
        if n['a'] != 4: continue
        three = bool(m['joints'][i]['flags'] & 4)
        eff = i + (3 if three else 2)
        pts = [Wb[i]] + ([Wb[i + 2]] if three else []) + [Wb[eff]]
        out.append(dict(root=i, three=three, pts=np.array(pts), length=sum(np.linalg.norm(np.diff(pts, axis=0), axis=1))))
    return out

def limb_path(m, eff, w):
    """Joint positions from the body midline down to a ground effector."""
    Wb = [np.linalg.inv(x['IBM'].T)[:3, 3] for x in m['joints']]
    path = [eff]; i = eff
    while m['nodes'][i]['parent'] >= 0:
        p = m['nodes'][i]['parent']
        if abs(Wb[p][0]) < 0.06 * w: break
        path.append(p); i = p
    pts = np.array([Wb[j] for j in reversed(path)])
    seg = np.r_[0, np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))]
    mid = pts[np.searchsorted(seg, seg[-1] / 2)] if len(pts) > 2 else (pts[0] + pts[-1]) / 2
    return np.array([pts[0], mid, pts[-1]]), seg[-1]

def roles(m):
    P = np.array([p for b in m['batches'] for p in b['pos']]); h = np.ptp(P[:, 1]); c = P.mean(0)
    C = chains(m); R = {}
    w = np.ptp(P[:, 0])
    feet = [k for k in C if k['three'] and -0.12 * h < k['pts'][-1][1] < 0.06 * h]
    byhip = {}
    for k in feet:
        pts, L = limb_path(m, k['root'] + 3, w)
        if L < 0.15 * h: continue
        key = tuple(np.round(pts[0], 3))
        # one limb per hip: prefer the shortest joint path (the leg itself, not a toe hanging off the foot)
        depth = 0; i = k['root']
        while m['nodes'][i]['parent'] >= 0: i = m['nodes'][i]['parent']; depth += 1
        if key not in byhip or depth < byhip[key][0]: byhip[key] = (depth, dict(root=k['root'], pts=pts, length=L))
    legs = [v for _, v in byhip.values()]
    zmid = np.mean([k['pts'][0][2] for k in legs]) if legs else c[2]
    for k in legs:
        side = 'L' if k['pts'][0][0] > 0 else 'R'; fb = 'F' if k['pts'][0][2] >= zmid else 'B'
        R['leg' + fb + side] = k
    mid = [k for k in C if not k['three'] and abs(k['pts'][-1][0]) < 0.05 * h]
    front = sorted([k for k in mid if k['pts'][-1][2] > c[2]], key=lambda k: k['pts'][-1][2])
    back = sorted([k for k in mid if k['pts'][-1][2] <= c[2]], key=lambda k: -k['pts'][-1][2])
    if front: R['head'] = front[-1]; R['neck'] = front[0]
    if back: R['tail_tip'] = back[-1]; R['tail_base'] = back[0]
    return R

def landmark_pairs(mA, mB):
    RA, RB = roles(mA), roles(mB); A, B = [], []
    for k in RA:
        if k in RB:
            pa, pb = RA[k]['pts'], RB[k]['pts']
            if len(pa) == len(pb): A += list(pa); B += list(pb)
            else: A += [pa[0], pa[-1]]; B += [pb[0], pb[-1]]
    return np.array(A), np.array(B), sorted(set(RA) & set(RB))

if __name__ == '__main__':
    for f in sys.argv[1:]:
        m = J.parse_model(open(f, 'rb').read()); R = roles(m)
        print(f); [print('   %-9s root %2d' % (k, v['root']), v['pts'].round(2).tolist()) for k, v in sorted(R.items())]
