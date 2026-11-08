#!/usr/bin/env python3
"""Group JC2 XMDL models by their VIF vertex-batch signature (V3-32 UNPACK counts)."""
import json, struct, sys, collections
idx = json.load(open(sys.argv[1] + '/index.json'))
res = {}
for r in idx:
    f = r['file']
    if not f or f.endswith('mwo3'): continue
    d = open(sys.argv[1] + '/' + f, 'rb').read()
    if b'LDMX' not in d: continue
    n = struct.unpack_from('<I', d, 0)[0]
    if 0 < n < 64:
        offs = list(struct.unpack_from('<%dI' % n, d, 4)) + [len(d)]
        secs = [(offs[k], offs[k + 1]) for k in range(n) if offs[k] < len(d)]
    else:
        secs = [(0, len(d))]
    sig = []
    for a, b in secs:
        if d[a:a + 4] != b'0IVX': continue
        g = d[a:b]
        for i in range(0, len(g) - 3, 4):
            w = struct.unpack_from('<I', g, i)[0]
            if w >> 24 == 0x68 and 0 < (w >> 16) & 0xff:   # UNPACK V3-32 = positions
                sig.append((w >> 16) & 0xff)
    res[r['index']] = dict(skin=b'NIKS' in d, nverts=sum(sig), nbatch=len(sig), sig=sig, size=len(d))
json.dump(res, open(sys.argv[2], 'w'))
groups = collections.defaultdict(list)
for i, v in res.items():
    if v['skin'] and v['nverts']: groups[tuple(v['sig'])].append(i)
big = sorted(groups.values(), key=len, reverse=True)
print('skinned models with geometry:', sum(len(g) for g in groups.values()), 'distinct signatures:', len(groups))
for g in big[:15]:
    v = res[g[0]]; print(len(g), 'models, verts', v['nverts'], 'batches', v['nbatch'], 'e.g.', g[:10])
