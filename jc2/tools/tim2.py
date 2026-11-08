#!/usr/bin/env python3
"""Minimal TIM2 decoder -> list of RGBA numpy images."""
import struct, numpy as np

def _clut(raw, colors, ctype):
    fmt = ctype & 0x3F
    if fmt == 3:   c = np.frombuffer(raw[:colors * 4], np.uint8).reshape(-1, 4).copy()
    elif fmt == 2: c = np.concatenate([np.frombuffer(raw[:colors * 3], np.uint8).reshape(-1, 3), np.full((colors, 1), 128, np.uint8)], 1)
    else:
        v = np.frombuffer(raw[:colors * 2], '<u2').astype(np.uint32)
        c = np.stack([(v & 31) << 3, ((v >> 5) & 31) << 3, ((v >> 10) & 31) << 3, np.where(v & 0x8000, 128, 0)], 1).astype(np.uint8)
    if colors == 256 and not (ctype & 0x80):          # CSM1: swap blocks 8..15 / 16..23 in each 32
        idx = np.arange(256); blk = (idx // 8) % 4
        idx = np.where(blk == 1, idx + 8, np.where(blk == 2, idx - 8, idx))
        c = c[idx]
    a = c[:, 3].astype(np.int32) * 2; c[:, 3] = np.clip(a, 0, 255)   # PS2 alpha 0..128
    return c

def decode(d):
    assert d[:4] == b'TIM2'
    count = struct.unpack_from('<H', d, 6)[0]
    o = 16 if d[5] == 0 else 128
    out = []
    for _ in range(count):
        total, clut_sz, img_sz, hdr, colors = struct.unpack_from('<IIIHH', d, o)
        fmt, mips, ctype, itype = d[o + 0x10:o + 0x14]
        w, h = struct.unpack_from('<HH', d, o + 0x14)
        img = d[o + hdr:o + hdr + img_sz]; cl = d[o + hdr + img_sz:o + hdr + img_sz + clut_sz]
        if itype == 5:   rgba = _clut(cl, colors, ctype)[np.frombuffer(img[:w * h], np.uint8)]
        elif itype == 4:
            b = np.frombuffer(img[:w * h // 2], np.uint8); ix = np.stack([b & 15, b >> 4], 1).reshape(-1)
            rgba = _clut(cl, colors, ctype)[ix]
        elif itype == 3: rgba = np.frombuffer(img[:w * h * 4], np.uint8).reshape(-1, 4).copy(); rgba[:, 3] = np.clip(rgba[:, 3].astype(int) * 2, 0, 255)
        else: raise ValueError('image type %d' % itype)
        out.append(rgba.reshape(h, w, 4))
        o += total
    return out
