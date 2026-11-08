#!/usr/bin/env python3
"""Jade Cocoon 2 XMDL model parser (work in progress).

File: u32 n; u32 off[n]; sections. The geometry section starts '0IVX00.1'.
Geometry: header, node table at +0xB0 (stride 0x50), then a VIF DMA chain.
"""
import struct, math

class Node:
    def __init__(s, i, raw):
        (s.type, _, _, s.a, s.b, s.parent1, s.child1, s.sib1) = struct.unpack_from('<3I4hI', raw, 0)
        s.i = i
        s.t = struct.unpack_from('<3f', raw, 0x20)
        s.r = struct.unpack_from('<3f', raw, 0x30)
        s.s = struct.unpack_from('<3f', raw, 0x40)
        s.parent = s.parent1 - 1          # -1 = root

def sections(d):
    n = struct.unpack_from('<I', d, 0)[0]
    offs = list(struct.unpack_from('<%dI' % n, d, 4)) + [len(d)]
    return [d[offs[k]:offs[k + 1]] for k in range(n)]

def geometry(d):
    for s in sections(d):
        if s[:4] == b'0IVX': return s

def nodes(g):
    count = struct.unpack_from('<I', g, 0x24)[0]
    out = []
    for k in range(count):
        o = 0xB0 + k * 0x50
        n = Node(k, g[o:o + 0x50])
        if n.type not in (1, 3, 5, 7): break
        out.append(n)
    return out, 0xB0 + len(out) * 0x50

UNPACK_SIZE = {0x0: (4, 1), 0x1: (2, 1), 0x2: (1, 1),     # S-32/16/8: (bytes per elem, elems)
               0x4: (4, 2), 0x5: (2, 2), 0x6: (1, 2),
               0x8: (4, 3), 0x9: (2, 3), 0xA: (1, 3),
               0xC: (4, 4), 0xD: (2, 4), 0xE: (1, 4), 0xF: (2, 1)}

def batches(g, start):
    """Yield dicts: palette (list of slot ids), pos, uv, nrm, wgt, gif (raw qwords)."""
    o = start
    cur = None
    while o + 16 <= len(g):
        tag, hi = struct.unpack_from('<II', g, o)
        qwc, tid = tag & 0xFFFF, (tag >> 28) & 7
        if tid == 1 and (tag & 0x0FFF0000) == 0:              # CNT: batch body follows
            pal = [b for b in g[o + 4:o + 8] if b != 0xFF]
            cur = dict(palette=pal, pos=[], uv=[], nrm=[], wgt=[], off=o)
            end = o + 16 + qwc * 16
            p = o + 8
            while p + 4 <= end:
                code = struct.unpack_from('<I', g, p)[0]; p += 4
                cmd, num, imm = code >> 24, (code >> 16) & 0xFF, code & 0xFFFF
                if cmd & 0x60 == 0x60:
                    vnvl = cmd & 0xF; addr = imm & 0x3FF
                    bpe, ne = UNPACK_SIZE[vnvl]
                    n = num or 256
                    size = (bpe * ne * n + 3) & ~3
                    data = g[p:p + bpe * ne * n]; p += size
                    if addr == 0x45 and vnvl == 0x8:
                        cur['pos'] = [struct.unpack_from('<3f', data, 12 * k) for k in range(n)]
                    elif addr == 0x46 and vnvl == 0x4:
                        cur['uv'] = [struct.unpack_from('<2f', data, 8 * k) for k in range(n)]
                    elif addr == 0x47 and vnvl == 0xD:
                        cur['nrm'] = [struct.unpack_from('<4h', data, 8 * k) for k in range(n)]
                    elif addr == 0x48 and vnvl == 0xE:
                        cur['wgt'] = [tuple(data[4 * k:4 * k + 4]) for k in range(n)]
                    elif addr == 0x41:
                        cur['gif'] = data
                elif cmd == 0x30 or cmd == 0x31: p += 16          # STROW / STCOL
                elif cmd == 0x50 or cmd == 0x51: p += (imm or 0x10000) * 16
            if cur['pos']: yield cur
            o = end
        else:
            o += 16

def euler(r, order):
    x, y, z = r
    cx, sx, cy, sy, cz, sz = math.cos(x), math.sin(x), math.cos(y), math.sin(y), math.cos(z), math.sin(z)
    X = [[1, 0, 0], [0, cx, -sx], [0, sx, cx]]
    Y = [[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]]
    Z = [[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]]
    M = {'X': X, 'Y': Y, 'Z': Z}
    R = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    for a in order:                                            # R = M[order[0]] @ M[order[1]] @ ...
        R = [[sum(R[i][k] * M[a][k][j] for k in range(3)) for j in range(3)] for i in range(3)]
    return R

def world(ns, order='ZYX'):
    W = {}
    for n in ns:
        R = euler(n.r, order); T = list(n.t)
        if n.parent >= 0:
            PR, PT = W[n.parent]
            R2 = [[sum(PR[i][k] * R[k][j] for k in range(3)) for j in range(3)] for i in range(3)]
            T2 = [PT[i] + sum(PR[i][k] * T[k] for k in range(3)) for i in range(3)]
            R, T = R2, T2
        W[n.i] = (R, T)
    return W
