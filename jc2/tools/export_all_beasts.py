#!/usr/bin/env python3
"""Export one GLB per distinct skinned mesh. usage: export_all_beasts.py <cddata> <xmdl_parsed.json> <outdir> <worker> <nworkers>"""
import sys, json, os, traceback
sys.path.insert(0, os.path.dirname(__file__)); import jc2_gltf
cd, pj, out, wk, nw = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5])
P = json.load(open(pj)); seen = {}; todo = []
for k in sorted(P, key=int):
    v = P[k]
    if 'err' in v or not v.get('verts'): continue
    if v['pos'] in seen: continue
    seen[v['pos']] = k; todo.append(int(k))
os.makedirs(out, exist_ok=True)
for n, i in enumerate(todo):
    if n % nw != wk: continue
    try:
        r = jc2_gltf.export(open('%s/%04d.bin' % (cd, i), 'rb').read(), '%s/jc2_%04d.glb' % (out, i), name='jc2_%04d' % i)
        print(i, 'ok', json.dumps(r), flush=True)
    except Exception as e:
        print(i, 'FAIL', repr(e), traceback.format_exc().splitlines()[-3:], flush=True)
