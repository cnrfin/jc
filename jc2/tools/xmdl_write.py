#!/usr/bin/env python3
"""Write a Jade Cocoon 2 model file (XMDL geometry + TIM2) from a mesh, rig and texture.

The layout follows the game's loader (func_00175EB0) and the batch structure of the
game's own beast files byte for byte; fields whose meaning is not known are copied from
a template beast (1821 by default). See jc2/README.md.

Entry point: build_model(mesh, template_bytes) -> bytes, where mesh is a dict:
  nodes: list of dict(parent, T(3), R(3) radians, S(3), palette(bool))   node 0 = root
  pos (V,3) bind-pose positions, nrm (V,3), uv (V,2) in 0..1, bone (V,) node index (rigid)
  tris (F,3), tex RGBA uint8 (256x256)
"""
import math, struct, numpy as np
from PIL import Image
import xmdl

def euler(r):
    cx, sx, cy, sy, cz, sz = math.cos(r[0]), math.sin(r[0]), math.cos(r[1]), math.sin(r[1]), math.cos(r[2]), math.sin(r[2])
    X = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]]); Y = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Z = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]]); return Z @ Y @ X

def bind_world(nodes):
    W = []
    for n in nodes:
        M = np.eye(4); M[:3, :3] = euler(n['R']) * np.asarray(n['S']); M[:3, 3] = n['T']
        W.append(M if n['parent'] < 0 else W[n['parent']] @ M)
    return W

def f4(v, w=1.0): return struct.pack('<4f', v[0], v[1], v[2], w)

# ------------------------------------------------------------------ TIM2 (8-bit, CSM1 CLUT)
def csm1(idx):
    blk = (idx // 8) % 4
    return np.where(blk == 1, idx + 8, np.where(blk == 2, idx - 8, idx))

def tim2(rgba, template_tim):
    h, w = rgba.shape[:2]
    img = Image.fromarray(rgba, 'RGBA')
    q = img.quantize(256, method=Image.Quantize.FASTOCTREE)
    pal = np.array(q.getpalette('RGBA')[:1024] if 'A' in q.palette.mode else
                   sum(([*q.getpalette()[i * 3:i * 3 + 3], 255] for i in range(256)), []), np.uint8).reshape(-1, 4)
    if len(pal) < 256: pal = np.r_[pal, np.zeros((256 - len(pal), 4), np.uint8)]
    pal = pal[:256].copy(); pal[:, 3] = (pal[:, 3].astype(int) * 128 + 127) // 255     # PS2 alpha 0..128
    clut = pal[csm1(np.arange(256))]
    pix = np.array(q, np.uint8).tobytes()
    th = bytearray(template_tim[16:16 + 0x30])                   # picture header from the template
    total = 0x30 + len(pix) + 1024
    struct.pack_into('<IIIHH', th, 0, total, 1024, len(pix), 0x30, 256)
    th[0x10:0x14] = bytes([0, 1, 0x03, 5]); struct.pack_into('<HH', th, 0x14, w, h)
    return b'TIM2' + bytes([4, 0]) + struct.pack('<H', 1) + b'\0' * 8 + bytes(th) + pix + clut.tobytes()

# ------------------------------------------------------------------ geometry
def strip_of(tris):
    """Join triangles into one strip with degenerate links, keeping winding parity."""
    s = []
    for t in tris:
        a, b, c = map(int, t)
        if not s: s += [a, b, c]; continue
        s += [s[-1], a]
        if len(s) % 2 == 1: s.append(a)                       # next triangle starts on an even index
        s += [a, b, c]
    return s

def batches(tris, bone, slot, max_verts=30):
    """Group triangles by their (<=4) palette slots, then cut into strips of <= max_verts."""
    groups = {}
    for t in tris:
        key = tuple(sorted({slot[bone[v]] for v in t})); groups.setdefault(key, []).append(t)
    # merge small groups whose slot sets fit in 4
    keys = sorted(groups, key=len, reverse=True); merged = []
    for k in keys:
        for m in merged:
            u = tuple(sorted(set(m[0]) | set(k)))
            if len(u) <= 4: m[0] = u; m[1] += groups[k]; break
        else: merged.append([k, list(groups[k])])
    out = []
    for pal, ts in merged:
        cur = []
        for t in ts:
            trial = strip_of(cur + [t])
            if len(trial) > max_verts and cur: out.append((pal, strip_of(cur))); cur = [t]
            else: cur.append(t)
        if cur: out.append((pal, strip_of(cur)))
    return out

def vif_batch(pal, strip, mesh, slot):
    P, N, UV, B = mesh['pos'], mesh['nrm'], mesh['uv'], mesh['bone']
    n = len(strip); body = bytearray()
    # matrix uploads: two REF tags per palette slot (patched by the loader, func_0014C4E0)
    for i, s in enumerate(pal):
        for base in (0x00, 0x20):
            body += struct.pack('<IIII', 0x30000008, 2, 0x01000404 | (s << 16), 0x6C088000 | (base + 8 * i))
    data = bytearray()
    data += struct.pack('<II', 0x01000404, 0x6C018040) + struct.pack('<4I', 0, 1, 0xFF, 0xFF)
    last = P[strip[-1]]
    gif0 = struct.pack('<IIII', 0x8000 | n, 0x303EC000, 0x412, 0x3F800000)
    gif1 = struct.pack('<IIII', 0x8000 | n, 0x303E4000, 0x412, 0x3F800000)
    data += struct.pack('<II', 0x01000404, 0x6C048041) + gif0 + f4(last) + f4(last) + gif1
    data += struct.pack('<I', 0x01000104)
    data += struct.pack('<I', 0x68008045 | (n << 16)) + b''.join(struct.pack('<3f', *P[v]) for v in strip)
    data += struct.pack('<I', 0x64008046 | (n << 16)) + b''.join(struct.pack('<2f', *UV[v]) for v in strip)
    data += struct.pack('<I', 0x6D008047 | (n << 16)) + b''.join(
        struct.pack('<4h', *[int(round(c * 32767)) for c in N[v]], 0x7FFF) for v in strip)
    data += struct.pack('<I', 0x6E00C048 | (n << 16))
    for v in strip:
        w = [0, 0, 0, 0]; w[pal.index(slot[B[v]])] = 255; data += bytes(w)
    data += struct.pack('<I', 0x17000000)                        # MSCNT
    while len(data) % 16: data += struct.pack('<I', 0)           # pad with NOPs to a qword
    pb = bytes(pal) + b'\xff' * (4 - len(pal))
    qwc = len(data) // 16                                          # qwords after the 16-byte tag qword
    body += struct.pack('<I', 0x10000000 | qwc) + pb + struct.pack('<II', 0, 3) + data
    return body, n

def build_model(mesh, template):
    tsecs = xmdl.sections(template); tg = tsecs[1]
    nodes = mesh['nodes']; nn = len(nodes)
    pal_nodes = [i for i, n in enumerate(nodes) if n['palette']]; slot = {nd: k for k, nd in enumerate(pal_nodes)}
    W = bind_world(nodes)
    # node table with child/sibling links
    kids = {i: [j for j, m in enumerate(nodes) if m['parent'] == i] for i in range(nn)}
    sib = {}
    for i, ch in kids.items():
        for a, b in zip(ch, ch[1:]): sib[a] = b
    depth = [0] * nn
    for i in range(1, nn): depth[i] = depth[nodes[i]['parent']] + 1
    nt = bytearray()
    for i, n in enumerate(nodes):
        child = kids[i][0] if kids[i] else 0; s = sib.get(i, 0)
        typ = (0x32 if i == 0 else 1 | (2 if child else 0) | (4 if s else 0))
        nt += struct.pack('<3I4hI', typ, 0, 0, 1 if i == 0 else 0, depth[i], max(n['parent'], 0), child, s)
        nt += b'\0' * 8 + f4(n['T']) + f4(n['R']) + f4(n['S'])
    leaves = sum(1 for i in range(nn) if not kids[i])
    # geometry: one mesh, one part
    B = batches(mesh['tris'], mesh['bone'], slot)
    chain = bytearray(); nverts = 0
    for pal, strip in B:
        b, n = vif_batch(list(pal), strip, mesh, slot); chain += b; nverts += n
    qwc = len(chain) // 16 - 1
    lo, hi = mesh['pos'].min(0), mesh['pos'].max(0)
    part = struct.pack('<4I', nverts, len(B), qwc, 0) + f4(lo) + f4(hi) + bytes(chain)
    meshrec = struct.pack('<4I', 0, 1, 0, 0) + f4(lo) + f4(hi) + struct.pack('<II', 0, 0) + b'\0' * 8
    # materials: copied from the template
    nobj, tnn, tnm, tnmat, tnp, tnj = struct.unpack_from('<6I', tg, 0x20)
    p = 0x40 + nobj * 0x20 + tnn * 0x50
    for _ in range(tnm):
        k = struct.unpack_from('<I', tg, p + 4)[0]; p += 0x30 + k * 8 + (8 if k & 1 else 0)
    mats = tg[p:p + tnmat * 0x60]
    # joints: one per node, inverse bind matrix in row-vector layout
    jt = bytearray()
    for i, n in enumerate(nodes):
        child = kids[i][0] if kids[i] else 0
        jt += struct.pack('<iIII', 1 if i == 0 else 0, 1 if n['palette'] else 0, child, sib.get(i, 0))
        jt += f4((0, 0, 0)) + f4(n['T']) + f4(n['R']) + f4(n['S'])
        jt += np.linalg.inv(W[i]).T.astype('<f4').tobytes()
    head = bytearray(tg[:0x20])
    head += struct.pack('<6I', 1, nn, 1, tnmat, 1, nn) + b'\0' * 8
    obj = struct.pack('<8I', 1, 0, 0, 0, nn, leaves, 0, nn)
    geo = bytes(head) + obj + bytes(nt) + meshrec + mats + part + bytes(jt) + b'\0' * 16
    tim = tim2(mesh['tex'], [s for s in tsecs if s[:4] == b'TIM2'][0])
    secs = [tsecs[0], geo, tim]
    off0 = ((4 + 4 * len(secs) + 15) & ~15) + 0x10
    out = bytearray(struct.pack('<I', len(secs)) + b'\0' * (4 * len(secs)))
    while len(out) < off0: out += b'\0'
    offs = []
    for s in secs:
        while len(out) % 16: out += b'\0'
        offs.append(len(out)); out += s
    struct.pack_into('<%dI' % len(secs), out, 4, *offs)
    return bytes(out)
