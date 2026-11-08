#!/usr/bin/env python3
"""Seed-form creature generator: one fixed topology, features with continuous controls.

Designed so two creatures blend feature-by-feature (JC1 style): arms shrink, bumps grow
sharper toward a horn, legs stretch, a skirt fringe grows in. Every creature has the same
vertices/triangles/UVs. Coordinates: JC2 axes (up = -Y), head toward -Z.
"""
import math, numpy as np

N = 20                                     # ring resolution
MATS = ['body', 'horn', 'spike', 'eye', 'leaf', 'leg', 'claw', 'tail', 'fin', 'pupil']
ROWH = 8

def palette(c, w=64):
    """c: dict material -> (top colour, bottom colour)."""
    rows = []
    for mname in MATS:
        c0, c1 = c[mname]; t = np.linspace(0, 1, w)[:, None]
        rows.append(np.repeat(((1 - t) * np.array(c0, float) + t * np.array(c1, float))[None], ROWH, 0))
    tex = np.concatenate(rows, 0); out = np.zeros(tex.shape[:2] + (4,), np.uint8); out[..., :3] = np.clip(tex, 0, 255); out[..., 3] = 255
    return out

def V(row): return (MATS.index(row) * ROWH + ROWH / 2) / (len(MATS) * ROWH)
def P3(x, up, z): return np.array([x, -up, z], float)

class Mesh:
    def __init__(s): s.P, s.UV, s.T = [], [], []
    def tube(s, C, rx, ry, mat, ref=(0, -1, 0), flat_dir=None, shade='side'):
        C = np.asarray(C, float); M = len(C); base = len(s.P); U0 = np.array(ref, float)
        for i in range(M):
            t = C[min(i + 1, M - 1)] - C[max(i - 1, 0)]; t /= np.linalg.norm(t) + 1e-12
            r = U0 if abs(np.dot(t, U0)) < 0.95 else np.array([0, 0, -1.0])
            side = np.cross(t, r); side /= np.linalg.norm(side) + 1e-12; up = np.cross(side, t)
            for k in range(N):
                a = 2 * math.pi * k / N
                s.P.append(C[i] + rx[i] * math.cos(a) * side + ry[i] * math.sin(a) * up)
                u = (1 - math.sin(a)) / 2 if shade == 'side' else i / max(M - 1, 1)
                s.UV.append((0.03 + 0.94 * u, V(mat)))
        for i in range(M - 1):
            for k in range(N):
                a, b = base + i * N + k, base + i * N + (k + 1) % N; c, d = a + N, b + N
                s.T += [(a, c, b), (b, c, d)]
        for i, flip in ((0, True), (M - 1, False)):
            cap = len(s.P); s.P.append(C[i].copy()); s.UV.append((0.03 + 0.94 * (0.0 if (shade != 'side' and i == 0) else 1.0 if shade != 'side' else 0.5), V(mat)))
            for k in range(N):
                a, b = base + i * N + k, base + i * N + (k + 1) % N
                s.T.append((cap, b, a) if flip else (cap, a, b))

def lerp(a, b, n): t = np.linspace(0, 1, n)[:, None]; return (1 - t) * np.asarray(a, float) + t * np.asarray(b, float)
def bez(p0, p1, p2, n):
    t = np.linspace(0, 1, n)[:, None]; return (1 - t) ** 2 * np.asarray(p0) + 2 * (1 - t) * t * np.asarray(p1) + t ** 2 * np.asarray(p2)

def spike_dirs():
    """Fixed spike sites on the body (unit directions; x, up, z). Index 0 = crown (the horn)."""
    d = [(0, 1, 0)]
    for el, n, off in ((52, 6, 0.5), (22, 8, 0.0), (-8, 7, 0.5)):
        for k in range(n):
            az = 2 * math.pi * (k + off) / n; e = math.radians(el)
            v = (math.cos(e) * math.sin(az), math.sin(e), math.cos(e) * math.cos(az))
            if v[2] < -0.72 and el > 0: continue                 # keep the face clear
            d.append(v)
    return np.array(d, float)
DIRS = spike_dirs()

def build(c):
    m = Mesh(); cx, cy, cz = 0.0, c['body_h'], 0.0; rx, ry, rz = c['body_r']
    def surf(dx, dy, dz, k=1.0):
        return P3(cx + rx * dx * k, cy + ry * dy * k, cz + rz * dz * k)
    # body: rings along the vertical axis, bottom -> top
    n = 14; th = np.linspace(0.06, math.pi - 0.06, n)
    C = [P3(0, cy - ry * math.cos(t), 0) for t in th]
    prof = np.sin(th) ** c['body_round']
    m.tube(C, rx * prof, rz * prof, 'body', ref=(0, 0, -1), shade='len')
    # skirt fringe: a short frilled band just under the body's equator (Dethalcoch's purple underside)
    sk = c['skirt']
    for k in range(10):
        az = 2 * math.pi * (k + 0.5) / 10
        d = np.array([math.sin(az), -0.45, math.cos(az)]); d /= np.linalg.norm(d)
        b = surf(*d, k=0.93); o = d * np.array([rx, ry, rz]); o /= np.linalg.norm(o)
        tip = b + P3(o[0] * 0.12 * sk, -(0.16 * sk), o[2] * 0.12 * sk)
        w = np.array([0.11, 0.1, 0.07, 0.0]) * sk; m.tube(lerp(b, tip, 4), w, w * 0.35, 'tail', ref=(0, -1, 0), shade='len')
    # spikes (index 0 = horn)
    for i, d in enumerate(DIRS):
        L, R, tip = (c['horn_len'], c['horn_r'], c['horn_tip']) if i == 0 else (c['spike_len'] * (1.25 if d[1] > 0.5 else 1.0), c['spike_r'], c['spike_tip'])
        b = surf(*d, k=0.9); outward = surf(*d) - P3(cx, cy, cz); outward /= np.linalg.norm(outward)
        tipp = surf(*d) + outward * L + (P3(0, L * c['horn_curl'], -L * c['horn_curl']) if i == 0 else 0)
        mid = (b + tipp) / 2
        m.tube(bez(b, mid, tipp, 5), np.array([R, R * 0.95, (R + tip) / 2, tip + (R - tip) * 0.15, tip]),
               np.array([R, R * 0.95, (R + tip) / 2, tip + (R - tip) * 0.15, tip]), 'horn' if i == 0 else 'spike', shade='len')
    # eyes
    for sx in (1, -1):
        e = np.array([sx * c['eye_x'], c['eye_up'], -1.0]); e /= np.linalg.norm(e)
        p = surf(*e, k=0.97); o = surf(*e) - P3(cx, cy, cz); o /= np.linalg.norm(o); r = c['eye_r']
        m.tube(lerp(p - o * r * 0.3, p + o * r * 0.55, 4), r * np.array([1.0, 1.0, 0.85, 0.4]), r * c['eye_h'] * np.array([1.0, 1.0, 0.85, 0.4]), 'eye', shade='len')
        pr = r * c['pupil']; pp = p + o * r * 0.5
        m.tube(lerp(pp - o * pr * 0.2, pp + o * pr * 0.25, 3), pr * np.array([1, 1, 0.3]), pr * np.array([1, 1, 0.3]), 'pupil', shade='len')
    # leaf ears
    for sx in (1, -1):
        d = np.array([sx * 0.45, 0.75, -0.45]); d /= np.linalg.norm(d); b = surf(*d, k=0.95)
        L = c['leaf_len']; t = b + P3(sx * L * 0.9, L * 0.25, L * 0.3)
        w = c['leaf_w'] * np.array([0.3, 1.0, 0.9, 0.45, 0.0])
        m.tube(bez(b, (b + t) / 2 + P3(0, L * 0.15, 0), t, 5), w, w * 0.18, 'leaf', ref=(0, -1, 0), shade='len')
    # arms / fins with claw tips
    for sx in (1, -1):
        b = surf(sx, 0.15, 0.05, k=0.85); L = c['arm_len']; t = b + P3(sx * L, -L * 0.1, L * 0.1)
        w = c['arm_w'] * np.array([0.55, 1.0, 1.0, 0.85, 0.5]); hgt = w * c['arm_thick']
        m.tube(bez(b, (b + t) / 2 + P3(0, L * 0.12, 0), t, 5), hgt, w, 'fin', ref=(0, -1, 0), shade='len')
        for k, dz in enumerate((-0.6, 0.0, 0.6)):
            cb = t + P3(0, 0, dz * c['arm_w'] * 0.6); ct = cb + P3(sx * c['claw'], -c['claw'] * 0.15, dz * c['claw'] * 0.4)
            r = c['claw'] * 0.28 * np.array([1, 0.6, 0.0])
            m.tube(lerp(cb, ct, 3), r, r, 'claw', shade='len')
    # legs: hip -> knee -> ankle, feet with three front toes and one back toe
    for sx in (1, -1):
        hip = surf(sx * 0.55, -0.75, 0.1, k=0.8); ank = P3(sx * c['leg_x'], c['ankle_h'], 0.02)
        knee = (hip + ank) / 2 + P3(0, 0, -c['knee'])
        r = c['leg_r'] * np.array([1.4, 1.15, 1.0, 0.9, 0.85, 0.95])
        m.tube(bez(hip, knee, ank, 6), r, r, 'leg', shade='len')
        for k, az in enumerate((-0.5, 0.0, 0.5, math.pi)):
            ln = c['toe'] * (0.6 if k == 3 else 1.0)
            tip = ank + P3(sx * 0 + math.sin(az) * ln, -c['ankle_h'] + 0.02, -math.cos(az) * ln)
            rr = c['toe_r'] * np.array([1.0, 0.8, 0.0])
            m.tube(bez(ank, (ank + tip) / 2 + P3(0, 0.02, 0), tip, 3), rr, rr * 0.8, 'claw', shade='len')
    # tail
    b = surf(0, -0.35, 1.0, k=0.9); L = c['tail_len']
    t = b + P3(0, -L * c['tail_droop'], L)
    w = c['tail_w'] * np.array([1.0, 0.85, 0.75, 0.75, 0.9, 1.0, 0.7, 0.0]) ** np.array([1, 1, 1, 1, c['tail_flare'], c['tail_flare'], 1, 1])
    m.tube(bez(b, (b + t) / 2 + P3(0, L * 0.1, 0), t, 8), w, w * c['tail_flat'], 'tail', ref=(0, -1, 0), shade='len')
    return np.array(m.P), np.array(m.UV), np.array(m.T)

DESIGNS = {
    # after Devaluch (1785): round pink puffball, one tall sharp horn, leaf ears, broad clawed arms, stubby legs, short pointed tail
    'Puffluch': dict(
        body_h=0.6, body_r=(0.36, 0.38, 0.34), body_round=0.9, skirt=0.0,
        horn_len=0.32, horn_r=0.06, horn_tip=0.0, horn_curl=0.15,
        spike_len=0.0, spike_r=0.035, spike_tip=0.0,
        eye_x=0.55, eye_up=0.75, eye_r=0.085, eye_h=0.8, pupil=0.35,
        leaf_len=0.18, leaf_w=0.09,
        arm_len=0.5, arm_w=0.2, arm_thick=0.16, claw=0.13,
        leg_x=0.2, ankle_h=0.05, knee=0.02, leg_r=0.06, toe=0.11, toe_r=0.045,
        tail_len=0.3, tail_w=0.12, tail_flat=0.5, tail_flare=1.0, tail_droop=0.3,
        colours=dict(body=((245, 175, 175), (250, 228, 215)), horn=((205, 180, 230), (150, 120, 185)), spike=((240, 225, 225), (205, 170, 175)),
                     eye=((250, 225, 70), (235, 190, 40)), pupil=((35, 30, 30), (35, 30, 30)), leaf=((125, 175, 85), (85, 135, 60)),
                     leg=((240, 170, 170), (215, 145, 150)), claw=((250, 245, 240), (215, 205, 200)), tail=((240, 165, 170), (225, 140, 150)),
                     fin=((245, 180, 180), (250, 225, 215)))),
    # after Dethalcoch (1893): pale stone shell covered in blunt bumps, small dark eyes, purple skirt and tail, long yellow bird legs, big talons
    'Shellcoch': dict(
        body_h=0.78, body_r=(0.37, 0.37, 0.37), body_round=0.75, skirt=1.0,
        horn_len=0.16, horn_r=0.085, horn_tip=0.04, horn_curl=0.0,
        spike_len=0.11, spike_r=0.075, spike_tip=0.04,
        eye_x=0.35, eye_up=0.55, eye_r=0.05, eye_h=0.55, pupil=0.75,
        leaf_len=0.0, leaf_w=0.0,
        arm_len=0.0, arm_w=0.0, arm_thick=0.45, claw=0.0,
        leg_x=0.13, ankle_h=0.08, knee=0.06, leg_r=0.035, toe=0.19, toe_r=0.045,
        tail_len=0.5, tail_w=0.13, tail_flat=0.2, tail_flare=0.5, tail_droop=0.55,
        colours=dict(body=((215, 218, 190), (165, 170, 140)), horn=((235, 232, 215), (175, 175, 150)), spike=((235, 232, 215), (180, 182, 160)),
                     eye=((55, 50, 45), (30, 28, 25)), pupil=((10, 10, 10), (10, 10, 10)), leaf=((150, 160, 120), (120, 130, 100)),
                     leg=((215, 170, 70), (180, 130, 45)), claw=((240, 235, 220), (190, 185, 170)), tail=((185, 70, 165), (120, 40, 140)),
                     fin=((200, 205, 175), (165, 170, 140)))),
}
for _d in DESIGNS.values(): _d['pal'] = palette(_d['colours'])
