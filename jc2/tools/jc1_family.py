#!/usr/bin/env python3
"""Load a JC1 merge-family creature (models/family) as an assembled, textured mesh.

Uses the shared topology in family.json and one jc_NNNN.family.bin. The pose is the
bind pose: rest offsets + frame-0 rotations, built the way Merge Studio's poseBones()
does it (FUN_80047E4C's Euler triple, slot 0 forced flat, slot 1 takes the root offset).
"""
import json, math, os, struct, sys
import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
FAMILY = os.path.join(HERE, '..', '..', 'models', 'family')
sys.path.insert(0, os.path.join(HERE, '..', '..', 'tools'))
from export_family import read_family_bin           # noqa: E402

def rot_matrix(a0, a1, a2):
    t = 2 * math.pi / 4096
    s0, c0, s1, c1, s2, c2 = math.sin(a0 * t), math.cos(a0 * t), math.sin(a1 * t), math.cos(a1 * t), math.sin(a2 * t), math.cos(a2 * t)
    return np.array([[c2 * c1, c2 * s1 * s0 - s2 * c0, s2 * s0 + c2 * s1 * c0],
                     [s2 * c1, c2 * c0 + s2 * s1 * s0, s2 * s1 * c0 - c2 * s0],
                     [-s1, c1 * s0, c1 * c0]])

_topo = None
def topology():
    global _topo
    if _topo is None:
        _topo = json.load(open(os.path.join(FAMILY, 'family.json')))
    return _topo

def bone_world(fam, rest=None, rot=None):
    """World (R, t) per bone slot, in JC1 units (s16 space)."""
    T = topology(); rig = T['rig']
    rest = fam['rest'] if rest is None else rest
    rot = fam['rot0'] if rot is None else rot
    W = {}
    for i in range(rig['boneCount']):
        slf, par = rig['self'][i], rig['parent'][i]
        r = (0, 0, 0) if slf == 0 else rot[slf]
        R = rot_matrix(*r)
        t = np.array(fam['root'] if slf == 1 else rest[slf], float)
        if par == slf: W[slf] = (R, t)
        else:
            PR, Pt = W[par]; W[slf] = (PR @ R, Pt + PR @ t)
    return W

def load(file_idx):
    fam = read_family_bin(open(os.path.join(FAMILY, 'jc_%04d.family.bin' % file_idx), 'rb').read())
    T = topology()
    W = bone_world(fam)
    joint = T['vertex']['joint']
    P = np.array([W[joint[i]][0] @ np.array(p, float) + W[joint[i]][1] for i, p in enumerate(fam['pos'])])
    N = np.array([W[joint[i]][0] @ np.array(n, float) for i, n in enumerate(fam['nrm'])])
    N /= np.linalg.norm(N, axis=1, keepdims=True) + 1e-12
    tris = np.array(T['indices']).reshape(-1, 3)
    tex = np.array(Image.open(os.path.join(FAMILY, 'jc_%04d_tex.png' % file_idx)).convert('RGBA'))
    return dict(fam=fam, pos=P, nrm=N, uv=np.array(fam['uv']), tris=tris, tex=tex,
                joint=np.array(joint), part=np.array(T['vertex']['partGroup']),
                partFlagsA=fam['partFlagsA'], bones=W)

def models():
    return [m['file'] for m in topology()['models'] if m.get('status') == 'ok']
