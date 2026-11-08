#!/usr/bin/env python3
"""Export a Jade Cocoon 2 (SLUS_203.09) beast model file to glTF binary (.glb).

usage: jc2_gltf.py <model.bin> <out.glb> [--fps 30]

Everything here is read from the file; see jc2/README.md for the format.
"""
import io, json, math, struct, sys
import numpy as np
from PIL import Image
sys.path.insert(0, __file__.rsplit('/', 1)[0])
import xmdl, tim2

# ---------------------------------------------------------------- model
def parse_model(d):
    g = xmdl.geometry(d)
    nobj, nnode, nmesh, nmat, npart, njoint = struct.unpack_from('<6I', g, 0x20)
    nodes = []
    for i in range(nnode):
        o = 0x40 + nobj * 0x20 + i * 0x50
        t, _, _, a, depth, par, ch, sib = struct.unpack_from('<3I4hI', g, o)
        nodes.append(dict(type=t, a=a, parent=par if i else -1,
                          T=np.array(struct.unpack_from('<3f', g, o + 0x20)),
                          R=np.array(struct.unpack_from('<3f', g, o + 0x30)),
                          S=np.array(struct.unpack_from('<3f', g, o + 0x40))))
        nodes[-1]['T0'] = nodes[-1]['T'].copy()
    p = 0x40 + nobj * 0x20 + nnode * 0x50
    for _ in range(nmesh):
        k = struct.unpack_from('<I', g, p + 4)[0]; p += 0x30 + k * 8 + (8 if k & 1 else 0)
    p += nmat * 0x60
    for _ in range(npart):
        qwc = struct.unpack_from('<I', g, p + 8)[0]; p += 0x30 + (qwc + 1) * 0x10
    joints = []
    for j in range(njoint):
        o = p + j * 0x90
        joints.append(dict(flags=struct.unpack_from('<I', g, o + 4)[0],
                           roll=math.radians(struct.unpack_from('<f', g, o + 0x10)[0]),
                           IBM=np.array(struct.unpack_from('<16f', g, o + 0x50)).reshape(4, 4)))
    vstart = 0x40 + nobj * 0x20 + nnode * 0x50
    batches = list(xmdl.batches(g, vstart))
    tex = [tim2.decode(s)[0] for s in xmdl.sections(d) if s[:4] == b'TIM2']
    secs = xmdl.sections(d)
    anim = secs[3] if len(secs) > 3 and secs[3][:4] != b'TIM2' else None
    return dict(nodes=nodes, joints=joints, batches=batches, tex=tex[0] if tex else None, anim=anim)

# ---------------------------------------------------------------- animation
VALS = {0: lambda n: n, 1: lambda n: n + 1, 2: lambda n: n + 1, 3: lambda n: n + 1,
        4: lambda n: 2 * n + 1, 5: lambda n: 2 * n + 1, 6: lambda n: 3 * n + 1, 7: lambda n: 2 * n + 2}
COMP_BITS = list(range(5, 14))
# bit -> slot (0..2 T, 3..5 R, 6..8 S) from func_00145460: bits 7,6,5 -> +A0,+A4,+A8 ; 10,9,8 -> +B0.. ; 13,12,11 -> +C0..
SLOT = {7: 0, 6: 1, 5: 2, 10: 3, 9: 4, 8: 5, 13: 6, 12: 7, 11: 8}

def hermite(p0, m0, p1, m1, t):
    t2, t3 = t * t, t * t * t
    return (2 * t3 - 3 * t2 + 1) * p0 + (t3 - 2 * t2 + t) * m0 + (-2 * t3 + 3 * t2) * p1 + (t3 - t2) * m1

def curve(kind, frames, v):
    """Return f(frame) for one component. frames excludes the implicit key at 0."""
    if kind == 0:
        return (lambda f: v[0]) if v else None
    if not frames:
        return (lambda f: v[0]) if v else None
    F = [0] + list(frames)
    if kind in (1, 2, 3):                                    # value per key; 3 = linear, 1/2 treated as step
        P = v[:len(F)]; M = None
    elif kind == 6:                                          # v0, out0, (in, v, out)..., (in, v)
        P = [v[0]]; Mo = [v[1]]; Mi = [0.0]; q = 2
        for k in range(1, len(F)):
            Mi.append(v[q]); P.append(v[q + 1]); q += 2
            if k < len(F) - 1: Mo.append(v[q]); q += 1
        Mo.append(0.0); M = (Mi, Mo)
    elif kind == 7:                                          # (v, t) per key, frame 0 included
        P = v[0::2][:len(F)]; T = v[1::2][:len(F)]; M = (T, T)
    else:                                                    # 4/5: (v, t) per key, last tangent omitted
        P = v[0::2][:len(F)]; T = list(v[1::2]) + [0.0]; T = T[:len(F)]; M = (T, T)
    def f(x):
        if x <= F[0]: return P[0]
        for k in range(len(F) - 1):
            if x <= F[k + 1]:
                t = (x - F[k]) / max(1, F[k + 1] - F[k])
                if M is None:
                    return P[k] + (P[k + 1] - P[k]) * t if kind == 3 else P[k]
                return hermite(P[k], M[1][k], P[k + 1], M[0][k + 1], t)
        return P[-1]
    return f

def base_pose(m):
    """Rest values per node; a==6 joints hold their bind world position (func_00143E00)."""
    Tl, Rl, Sl = [], [], []
    for i, n in enumerate(m['nodes']):
        t = n['T0'].copy()
        if n['a'] == 6 and i < len(m['joints']):
            t = np.linalg.inv(m['joints'][i]['IBM'].T)[:3, 3].copy()
        Tl.append(t); Rl.append(n['R'].copy()); Sl.append(n['S'].copy())
    return Tl, Rl, Sl

def parse_clips(S):
    n = struct.unpack_from('<I', S, 0)[0]
    w = struct.unpack_from('<%dI' % n, S, 4)
    out = []
    for i in range(0, n - 1, 2):
        off, size = w[i], w[i + 1]
        a = S[off:off + size]
        if len(a) < 0x58: continue
        T = struct.unpack_from('<I', a, 12)[0]
        nch, flg = struct.unpack_from('<HH', a, T)
        chans = {}
        for k in range(nch):
            node, m, f8, vo, ko = struct.unpack_from('<HBBII', a, T + 4 + 12 * k)
            if not ko: continue
            mask = (f8 << 8) | m
            p, q = T + ko, T + vo
            comps = {}
            for b in sorted(COMP_BITS, reverse=True):          # game order: bit 13 first
                if not mask >> b & 1: continue
                hd = struct.unpack_from('<H', a, p)[0]; kind, cnt = hd >> 13, hd & 0x1FFF
                frames = struct.unpack_from('<%dH' % cnt, a, p + 2) if kind else ()
                p += 2 + (2 * cnt if kind else 0)
                nv = VALS[kind](cnt)
                vals = struct.unpack_from('<%df' % nv, a, q); q += 4 * nv
                c = curve(kind, frames, vals)
                if c: comps[SLOT[b]] = c
            chans[node] = comps
        out.append(dict(frames=flg & 0x7FFF, loop=bool(flg & 0x8000), chans=chans))
    return out

# ---------------------------------------------------------------- skeleton maths (Euler Z*Y*X)
def euler(r):
    cx, sx, cy, sy, cz, sz = math.cos(r[0]), math.sin(r[0]), math.cos(r[1]), math.sin(r[1]), math.cos(r[2]), math.sin(r[2])
    X = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]]); Y = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Z = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Z @ Y @ X

def trs(T, R, S):
    M = np.eye(4); M[:3, :3] = euler(R) * S; M[:3, 3] = T; return M

def Rz(c, s_):
    M = np.eye(4); M[0, 0] = c; M[0, 1] = -s_; M[1, 0] = s_; M[1, 1] = c; return M

def Ry_neg(c, s_):                       # rotation taking +X toward +Z by angle with cos c, sin s_
    M = np.eye(4); M[0, 0] = c; M[0, 2] = -s_; M[2, 0] = s_; M[2, 2] = c; return M

def Rx(a):
    M = np.eye(4); c, s_ = math.cos(a), math.sin(a); M[1, 1] = c; M[1, 2] = -s_; M[2, 1] = s_; M[2, 2] = c; return M

def aim(W0, target, roll, yaw_first=False):
    """func_00143960: rotate frame W0 so local +X points at target. Returns (W, distance)."""
    x, y, z = (np.linalg.inv(W0) @ np.append(target, 1))[:3]
    L = math.sqrt(x * x + y * y + z * z) or 1e-9
    if yaw_first:
        r = math.sqrt(x * x + z * z) or 1e-9
        M1 = np.eye(4); M1[0, 0] = x / r; M1[0, 2] = -z / r; M1[2, 0] = z / r; M1[2, 2] = x / r   # yaw
        M2 = Rz(r / L, y / L); roll = roll - math.pi
    else:
        r = math.sqrt(x * x + y * y) or 1e-9
        M1 = Rz(x / r, y / r); M2 = Ry_neg(r / L, z / L)
    if roll * roll > 9.2e-9: M2 = M2 @ Rx(roll)
    return W0 @ M1 @ M2, L

def tx(L):
    M = np.eye(4); M[0, 3] = L; return M

def world(nodes, joints, T, R, S):
    """Skeleton update after func_00142B00: FK, with IK chains starting at a==4 joints."""
    n_ = len(nodes); W = [None] * n_
    fl = [joints[i]['flags'] if i < len(joints) else 0 for i in range(n_)]
    roll = [joints[i].get('roll', 0.0) if i < len(joints) else 0.0 for i in range(n_)]
    def place(i, M):
        P = W[nodes[i]['parent']] if nodes[i]['parent'] >= 0 else np.eye(4)
        f = fl[i]
        if f & 0x400:
            X = np.eye(4); X[:3, :3] = M[:3, :3]; X[0, 3] = P[0, 3]; X[2, 3] = P[2, 3]; return X
        Wm = P @ M
        if f & 2: Wm[:3, :3] = M[:3, :3]
        return Wm
    def reset_rot(Wm, i):
        if fl[i] & 2: Wm = Wm.copy(); Wm[:3, :3] = euler(R[i]) * S[i]
        return Wm
    i = 0
    while i < n_:
        n = nodes[i]
        if W[i] is not None: i += 1; continue
        W[i] = place(i, trs(T[i], R[i], S[i]))
        if n['a'] == 4 and i + 2 < n_:
            three = bool(fl[i] & 4) and i + 3 < n_
            eff = i + 3 if three else i + 2
            target = np.array(T[eff], float)                      # a==6 slot holds a model-space position
            W1, d = aim(W[i], target, roll[i + 1], bool(fl[i] & 0x80))
            if not three:
                W[i + 1] = W1
                W[i + 2] = reset_rot(W1 @ trs(nodes[i + 2]['T0'], R[i + 2], S[i + 2]), i + 2)
            else:
                L1, L2 = nodes[i + 2]['T0'][0], nodes[i + 3]['T0'][0]
                if L1 + L2 <= d:
                    W[i + 1] = W1; W[i + 2] = W1 @ tx(L1)
                else:
                    ca = max(-1, min(1, (L1 * L1 + d * d - L2 * L2) / (2 * L1 * d))); sa = math.sqrt(1 - ca * ca)
                    sgn = -1 if fl[i] & 0x10 else 1
                    W[i + 1] = W1 @ Rz(ca, sgn * sa)
                    cb = max(-1, min(1, (L1 * L1 + L2 * L2 - d * d) / (2 * L1 * L2))); sb = math.sqrt(1 - cb * cb)
                    K = Rz(-cb, -sgn * sb)
                    if roll[i + 2] * roll[i + 2] > 9.2e-9: K = K @ Rx(roll[i + 2])
                    W[i + 2] = W[i + 1] @ tx(L1) @ K
                W[i + 3] = reset_rot(W[i + 2] @ trs(nodes[i + 3]['T0'], R[i + 3], S[i + 3]), i + 3)
        i += 1
    return W

def mat_to_trs(M):
    t = M[:3, 3].copy(); A = M[:3, :3]
    s = np.linalg.norm(A, axis=0); s[s == 0] = 1
    Rm = A / s
    if np.linalg.det(Rm) < 0: s[0] = -s[0]; Rm[:, 0] = -Rm[:, 0]
    tr = np.trace(Rm)
    if tr > 0:
        w = math.sqrt(1 + tr) * 2; q = [(Rm[2, 1] - Rm[1, 2]) / w, (Rm[0, 2] - Rm[2, 0]) / w, (Rm[1, 0] - Rm[0, 1]) / w, w / 4]
    elif Rm[0, 0] > Rm[1, 1] and Rm[0, 0] > Rm[2, 2]:
        w = math.sqrt(1 + Rm[0, 0] - Rm[1, 1] - Rm[2, 2]) * 2; q = [w / 4, (Rm[0, 1] + Rm[1, 0]) / w, (Rm[0, 2] + Rm[2, 0]) / w, (Rm[2, 1] - Rm[1, 2]) / w]
    elif Rm[1, 1] > Rm[2, 2]:
        w = math.sqrt(1 + Rm[1, 1] - Rm[0, 0] - Rm[2, 2]) * 2; q = [(Rm[0, 1] + Rm[1, 0]) / w, w / 4, (Rm[1, 2] + Rm[2, 1]) / w, (Rm[0, 2] - Rm[2, 0]) / w]
    else:
        w = math.sqrt(1 + Rm[2, 2] - Rm[0, 0] - Rm[1, 1]) * 2; q = [(Rm[0, 2] + Rm[2, 0]) / w, (Rm[1, 2] + Rm[2, 1]) / w, w / 4, (Rm[1, 0] - Rm[0, 1]) / w]
    return t, np.array(q), s

# ---------------------------------------------------------------- glTF writer
class GLB:
    def __init__(s):
        s.bin = bytearray(); s.j = dict(asset=dict(version='2.0', generator='jc2_gltf.py'), buffers=[], bufferViews=[], accessors=[])
    def view(s, data, target=None):
        while len(s.bin) % 4: s.bin += b'\0'
        v = dict(buffer=0, byteOffset=len(s.bin), byteLength=len(data))
        if target: v['target'] = target
        s.bin += data; s.j['bufferViews'].append(v); return len(s.j['bufferViews']) - 1
    def acc(s, arr, ctype, typ, target=None, minmax=False):
        arr = np.ascontiguousarray(arr)
        a = dict(bufferView=s.view(arr.tobytes(), target), componentType=ctype, count=len(arr), type=typ)
        if minmax:
            a['min'] = arr.reshape(len(arr), -1).min(0).tolist(); a['max'] = arr.reshape(len(arr), -1).max(0).tolist()
        s.j['accessors'].append(a); return len(s.j['accessors']) - 1
    def write(s, path):
        while len(s.bin) % 4: s.bin += b'\0'
        s.j['buffers'] = [dict(byteLength=len(s.bin))]
        js = json.dumps(s.j, separators=(',', ':')).encode()
        while len(js) % 4: js += b' '
        out = struct.pack('<III', 0x46546C67, 2, 12 + 8 + len(js) + 8 + len(s.bin))
        out += struct.pack('<II', len(js), 0x4E4F534A) + js + struct.pack('<II', len(s.bin), 0x004E4942) + s.bin
        open(path, 'wb').write(out)

def export(d, path, name='JC2Beast', fps=30):
    m = parse_model(d)
    nodes, joints = m['nodes'], m['joints']
    palette = [j for j, J in enumerate(joints) if J['flags'] & 1]
    G = GLB(); j = G.j
    # mesh: strips -> triangles
    P, N, UV, JI, WT, idx = [], [], [], [], [], []
    for b in m['batches']:
        base = len(P)
        pal = b['palette'] + [0] * (4 - len(b['palette']))
        for k in range(len(b['pos'])):
            P.append(b['pos'][k]); nx, ny, nz, _ = b['nrm'][k]
            n = np.array([nx, ny, nz], float); n /= (np.linalg.norm(n) or 1); N.append(n)
            UV.append(b['uv'][k])
            w = np.array(b['wgt'][k], float); w = w / (w.sum() or 1)
            JI.append([pal[i] if w[i] > 0 else 0 for i in range(4)]); WT.append(w)
        for k in range(len(b['pos']) - 2):
            tri = (base + k, base + k + 1, base + k + 2) if k % 2 == 0 else (base + k + 1, base + k, base + k + 2)
            idx.append(tri)
    attrs = dict(POSITION=G.acc(np.array(P, np.float32), 5126, 'VEC3', 34962, True),
                 NORMAL=G.acc(np.array(N, np.float32), 5126, 'VEC3', 34962),
                 TEXCOORD_0=G.acc(np.array(UV, np.float32), 5126, 'VEC2', 34962),
                 JOINTS_0=G.acc(np.array(JI, np.uint16), 5123, 'VEC4', 34962),
                 WEIGHTS_0=G.acc(np.array(WT, np.float32), 5126, 'VEC4', 34962))
    prim = dict(attributes=attrs, indices=G.acc(np.array(idx, np.uint32).reshape(-1), 5125, 'SCALAR', 34963))
    if m['tex'] is not None:
        buf = io.BytesIO(); Image.fromarray(m['tex']).save(buf, 'PNG')
        j['images'] = [dict(bufferView=G.view(buf.getvalue()), mimeType='image/png')]
        j['samplers'] = [dict(magFilter=9729, minFilter=9729)]
        j['textures'] = [dict(source=0, sampler=0)]
        j['materials'] = [dict(name='tex', doubleSided=True, alphaMode='MASK',
                               pbrMetallicRoughness=dict(baseColorTexture=dict(index=0), metallicFactor=0, roughnessFactor=1))]
        prim['material'] = 0
    j['meshes'] = [dict(name=name, primitives=[prim])]
    # skeleton in bind pose: local matrices from the file's own bind matrices
    Wb = [np.linalg.inv(J['IBM'].T) for J in joints]
    gl_nodes = []
    for i, n in enumerate(nodes):
        L = Wb[i] if n['parent'] < 0 else np.linalg.inv(Wb[n['parent']]) @ Wb[i]
        t, q, s = mat_to_trs(L)
        gl_nodes.append(dict(name='n%02d' % i, translation=t.tolist(), rotation=q.tolist(), scale=s.tolist()))
    for i, n in enumerate(nodes):
        if n['parent'] >= 0: gl_nodes[n['parent']].setdefault('children', []).append(i)
    ib = np.array([joints[p]['IBM'] for p in palette], np.float32)    # row-vector layout == glTF column-major
    j['skins'] = [dict(joints=palette, inverseBindMatrices=G.acc(ib, 5126, 'MAT4'), skeleton=0)]
    mesh_node = len(gl_nodes); gl_nodes.append(dict(name=name + '_mesh', mesh=0, skin=0))
    root = len(gl_nodes); gl_nodes.append(dict(name=name, rotation=[1, 0, 0, 0], children=[0, mesh_node]))  # Y-down -> Y-up
    j['nodes'] = gl_nodes; j['scenes'] = [dict(nodes=[root])]; j['scene'] = 0
    # animations
    if m['anim'] is not None:
        anims = []
        for ci, c in enumerate(parse_clips(m['anim'])):
            F = max(1, c['frames']); times = np.arange(F + 1, dtype=np.float32) / fps
            T = np.zeros((F + 1, len(nodes), 3)); Q = np.zeros((F + 1, len(nodes), 4)); S = np.zeros((F + 1, len(nodes), 3))
            for fr in range(F + 1):
                Tl, Rl, Sl = base_pose(m)
                for node, comps in c['chans'].items():
                    if node >= len(nodes): continue
                    for ci2, fn in comps.items():
                        v = fn(fr)
                        (Tl if ci2 < 3 else Rl if ci2 < 6 else Sl)[node][ci2 % 3] = v
                W = world(nodes, joints, Tl, Rl, Sl)
                for i, n in enumerate(nodes):
                    L = W[i] if n['parent'] < 0 else np.linalg.inv(W[n['parent']]) @ W[i]
                    T[fr, i], Q[fr, i], S[fr, i] = mat_to_trs(L)
            ti = G.acc(times, 5126, 'SCALAR', None, True)
            samplers, channels = [], []
            for i in range(len(nodes)):
                for path_, arr, typ in (('translation', T, 'VEC3'), ('rotation', Q, 'VEC4'), ('scale', S, 'VEC3')):
                    col = arr[:, i]
                    if np.allclose(col, col[0], atol=1e-5) and path_ != 'rotation': continue
                    if path_ == 'rotation':                                # keep quaternions continuous
                        for k in range(1, len(col)):
                            if np.dot(col[k], col[k - 1]) < 0: col[k] = -col[k]
                    samplers.append(dict(input=ti, output=G.acc(col.astype(np.float32), 5126, typ), interpolation='LINEAR'))
                    channels.append(dict(sampler=len(samplers) - 1, target=dict(node=i, path=path_)))
            anims.append(dict(name='clip%02d_%df%s' % (ci, c['frames'], '_loop' if c['loop'] else ''), samplers=samplers, channels=channels))
        j['animations'] = anims
    G.write(path)
    return dict(nodes=len(nodes), palette=len(palette), verts=len(P), tris=len(idx), clips=len(j.get('animations', [])))

if __name__ == '__main__':
    fps = 30
    if '--fps' in sys.argv: fps = float(sys.argv[sys.argv.index('--fps') + 1])
    print(export(open(sys.argv[1], 'rb').read(), sys.argv[2], fps=fps))
