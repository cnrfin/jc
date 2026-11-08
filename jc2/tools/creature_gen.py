#!/usr/bin/env python3
"""A tiny JC1-style merge family designed from scratch.

Every creature is built by ONE generator with a fixed topology (same vertex count, same
triangles, same UVs), so any two creatures are each other's morph targets - exactly the
property JC1's 49-creature family has. A creature is a dict of parameters; parts a
creature does not have are given zero size (JC1's hidden part groups).

Coordinates follow JC2: up = -Y, ground at y = 0, head toward -Z.
Colour: a small palette texture; u runs back -> belly within a material row.
"""
import math, numpy as np

N = 12                                   # vertices per ring
ROWS = ['body', 'horn', 'eye', 'claw', 'spike']

def palette(body_top, body_belly, horn, eye, claw, spike, w=64):
    """64 x 40 texture: one 8-pixel band per material, u = back -> belly gradient."""
    def band(c0, c1):
        t = np.linspace(0, 1, w)[:, None]; row = (1 - t) * np.array(c0, float) + t * np.array(c1, float)
        return np.repeat(row[None], 8, 0)
    dark = lambda c, k: [x * k for x in c]
    tex = np.concatenate([band(body_top, body_belly), band(horn, dark(horn, .55)), band(eye, dark(eye, .3)),
                          band(claw, dark(claw, .5)), band(spike, dark(spike, .5))], 0)
    out = np.zeros(tex.shape[:2] + (4,), np.uint8); out[..., :3] = np.clip(tex, 0, 255); out[..., 3] = 255
    return out

class Mesh:
    def __init__(s): s.P, s.UV, s.T, s.part = [], [], [], []
    def tube(s, centers, rx, ry, row, part, up_ref=(0, -1, 0)):
        """Closed tube through `centers` with elliptical rings (rx, ry); caps at both ends."""
        C = np.asarray(centers, float); M = len(C); base = len(s.P)
        U0 = np.array(up_ref, float)
        for i in range(M):
            t = C[min(i + 1, M - 1)] - C[max(i - 1, 0)]; t /= np.linalg.norm(t) + 1e-9
            ref = U0 if abs(np.dot(t, U0)) < 0.9 else np.array([0, 0, -1.0])
            side = np.cross(t, ref); side /= np.linalg.norm(side) + 1e-9; up = np.cross(side, t)
            for k in range(N):
                a = 2 * math.pi * k / N
                s.P.append(C[i] + rx[i] * math.cos(a) * side + ry[i] * math.sin(a) * up)
                s.UV.append(((1 - math.sin(a)) / 2 * 0.96 + 0.02, (ROWS.index(row) * 8 + 4) / 40))
                s.part.append(part)
        for i in range(M - 1):
            for k in range(N):
                a, b = base + i * N + k, base + i * N + (k + 1) % N
                c, d = a + N, b + N
                s.T += [(a, c, b), (b, c, d)]
        for end, i, flip in ((C[0] - (C[1] - C[0]) * 0.15, 0, True), (C[-1] + (C[-1] - C[-2]) * 0.15, M - 1, False)):
            cap = len(s.P); s.P.append(end if rx[i] > 1e-6 else C[i])
            s.UV.append((0.5, (ROWS.index(row) * 8 + 4) / 40)); s.part.append(part)
            for k in range(N):
                a, b = base + i * N + k, base + i * N + (k + 1) % N
                s.T.append((cap, b, a) if flip else (cap, a, b))

def bez(p0, p1, p2, n):
    t = np.linspace(0, 1, n)[:, None]; p0, p1, p2 = map(np.asarray, (p0, p1, p2))
    return (1 - t) ** 2 * p0 + 2 * (1 - t) * t * p1 + t ** 2 * p2

def P3(x, up, z): return np.array([x, -up, z], float)

def build(c):
    """c: creature parameters (see CREATURES). Returns positions, uvs, triangles, part ids."""
    m = Mesh()
    # torso: tail base -> shoulder, 10 rings, hump in the middle
    back, front = c['torso_back'], c['torso_front']
    sp = bez(P3(0, *back), P3(0, c['hump'], (back[1] + front[1]) / 2), P3(0, *front), 10)
    prof = np.sin(np.linspace(0.25, math.pi - 0.25, 10)) ** c['torso_round']
    m.tube(sp, c['torso_w'] * prof + 0.02, c['torso_h'] * prof + 0.02, 'body', 'torso')
    # neck + head: shoulder -> snout tip, 10 rings; head bulb then taper to the snout
    nk = bez(P3(0, *front), P3(0, *c['head_mid']), P3(0, *c['snout']), 10)
    hp = np.array(c['head_prof'])
    m.tube(nk, c['head_w'] * hp, c['head_h'] * hp, 'body', 'head')
    head_c = nk[5]
    # legs: hip -> knee -> foot, 6 rings; front legs splay by 'f_splay'
    for side in (1, -1):
        for (hipz, footz, splay, L, r, pre) in ((front[1] + c['f_hip'], front[1] + c['f_hip'] + c['f_reach'], c['f_splay'], c['f_len'], c['f_r'], 'f'),
                                                 (back[1] - c['b_hip'], back[1] - c['b_hip'] + c['b_reach'], c['b_splay'], c['b_len'], c['b_r'], 'b')):
            hip = P3(side * c['hip_x'], L, hipz)
            foot = P3(side * (c['hip_x'] + splay), 0.04, footz)
            knee = (hip + foot) / 2 + P3(side * splay * 0.3, 0, c['knee_bend'] * (1 if pre == 'b' else -1))
            pts = bez(hip, knee, foot, 6)
            rr = r * np.array([1.3, 1.1, 0.9, 0.8, 0.9, 1.25])
            m.tube(pts, rr, rr, 'body', pre + 'leg')
            claw = [foot + P3(0, 0.02, 0), foot + P3(side * splay * 0.15, -0.01, -c['claw'])]
            m.tube(bez(claw[0], (claw[0] + claw[1]) / 2, claw[1], 3), np.array([r, r * 0.6, 0.0]), np.array([r * 0.5, r * 0.3, 0.0]), 'claw', 'claw')
    # tail: tail base -> tip, 8 rings; optional mace bulb at the end
    tb = P3(0, *back); tt = P3(0, *c['tail_tip'])
    tl = bez(tb, (tb + tt) / 2 + P3(0, c['tail_arch'], 0), tt, 8)
    tr = np.linspace(c['tail_r'], c['tail_r'] * 0.4, 8); mace = c['mace']
    tr[-3:] = np.maximum(tr[-3:], np.array([0.4, 1.0, 0.7]) * mace)
    m.tube(tl, tr, tr, 'body', 'tail')
    # mace spikes (zero on creatures without a mace)
    mc = tl[-2]
    for k in range(6):
        a = 2 * math.pi * k / 6
        d = np.array([math.cos(a), math.sin(a) * 0.9, 0.35]); d /= np.linalg.norm(d)
        base = mc + d * mace * 0.8; tip = mc + d * mace * 1.8
        m.tube(bez(base, (base + tip) / 2, tip, 3), np.array([1, .5, 0]) * mace * 0.35, np.array([1, .5, 0]) * mace * 0.35, 'spike', 'mace')
    # back spikes along the torso top (zero size where absent)
    for k in range(5):
        i = 2 + k * 1.4; i0 = int(i); f = i - i0
        p = sp[i0] * (1 - f) + sp[min(i0 + 1, 9)] * f
        rad = c['torso_h'] * (np.sin(0.25 + (math.pi - 0.5) * i / 9) ** c['torso_round']) + 0.02
        base = p + P3(0, rad * 0.85, 0); L = c['spikes'] * (0.7 + 0.3 * math.sin(math.pi * (k + .5) / 5))
        tip = base + P3(0, L, L * 0.45)
        m.tube(bez(base, (base + tip) / 2, tip, 3), np.array([1, .5, 0]) * L * 0.3, np.array([1, .5, 0]) * L * 0.25, 'spike', 'spikes')
    # eyes, horns, ears (pairs)
    for side in (1, -1):
        e = head_c + P3(side * c['head_w'] * hp[5] * 0.85, c['eye_up'], c['eye_fwd'])
        er = c['eye_r']
        m.tube(bez(e + P3(0, 0, er), e, e - P3(0, 0, er), 4), np.array([.3, 1, 1, .3]) * er, np.array([.3, 1, 1, .3]) * er, 'eye', 'eye')
        hb = head_c + P3(side * c['horn_x'], c['head_h'] * 0.5, c['horn_z'])
        ht = hb + P3(side * c['horn_out'], c['horn_len'], c['horn_fwd'])
        m.tube(bez(hb, (hb + ht) / 2 + P3(side * 0.05, 0, 0), ht, 5), np.linspace(1, 0, 5) * c['horn_r'], np.linspace(1, 0, 5) * c['horn_r'], 'horn', 'horn')
        ab = head_c + P3(side * c['head_w'] * 0.6, c['head_h'] * 0.6, 0.15)
        at = ab + P3(side * c['ear_out'], c['ear_len'] * 0.7, c['ear_len'] * 0.7)
        m.tube(bez(ab, (ab + at) / 2, at, 4), np.array([1, .9, .6, 0]) * c['ear_w'], np.array([1, .9, .6, 0]) * c['ear_w'] * 0.3, 'body', 'ear')
    return np.array(m.P), np.array(m.UV), np.array(m.T), np.array(m.part)

# Two designs on the same generator.
CREATURES = {
    # A: after Redoble (1822) - long snout, tall tusk-horns, swept-back ears, slim legs, thin tail
    'Tuskrunner': dict(
        torso_back=(1.05, 0.75), torso_front=(1.15, -0.55), hump=1.35, torso_w=0.32, torso_h=0.36, torso_round=0.6,
        head_mid=(1.25, -1.0), snout=(0.95, -1.75), head_w=0.26, head_h=0.27,
        head_prof=[0.9, 0.85, 0.9, 1.0, 1.0, 0.95, 0.8, 0.6, 0.45, 0.3],
        hip_x=0.22, f_hip=0.1, f_reach=-0.05, f_splay=0.05, f_len=1.0, f_r=0.09,
        b_hip=0.1, b_reach=0.05, b_splay=0.05, b_len=1.0, b_r=0.1, knee_bend=0.12, claw=0.12,
        tail_tip=(1.35, 1.6), tail_r=0.07, tail_arch=0.25, mace=0.0, spikes=0.0,
        eye_r=0.06, eye_up=0.08, eye_fwd=-0.05,
        horn_x=0.16, horn_z=-0.25, horn_out=0.1, horn_len=0.75, horn_fwd=-0.35, horn_r=0.07,
        ear_len=0.55, ear_w=0.16, ear_out=0.25,
        pal=palette((150, 40, 35), (215, 150, 120), (235, 230, 215), (250, 220, 70), (90, 80, 90), (200, 200, 210))),
    # B: after Dongly (1858) - round and bulky, short face, big back spikes, splayed stubby legs, mace tail
    'Macebulk': dict(
        torso_back=(0.75, 0.55), torso_front=(0.8, -0.45), hump=1.3, torso_w=0.6, torso_h=0.58, torso_round=0.35,
        head_mid=(0.85, -0.7), snout=(0.65, -1.0), head_w=0.36, head_h=0.32,
        head_prof=[0.8, 0.95, 1.0, 1.05, 1.05, 1.0, 0.95, 0.85, 0.7, 0.45],
        hip_x=0.38, f_hip=0.05, f_reach=-0.15, f_splay=0.3, f_len=0.55, f_r=0.13,
        b_hip=0.05, b_reach=0.1, b_splay=0.18, b_len=0.55, b_r=0.14, knee_bend=0.05, claw=0.18,
        tail_tip=(0.4, 1.55), tail_r=0.11, tail_arch=0.05, mace=0.2, spikes=0.38,
        eye_r=0.08, eye_up=0.14, eye_fwd=-0.12,
        horn_x=0.2, horn_z=-0.1, horn_out=0.18, horn_len=0.25, horn_fwd=-0.1, horn_r=0.06,
        ear_len=0.0, ear_w=0.0, ear_out=0.0,
        pal=palette((190, 70, 40), (245, 200, 70), (170, 225, 240), (250, 200, 40), (210, 235, 245), (175, 225, 240))),
}
