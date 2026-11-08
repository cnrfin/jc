#!/usr/bin/env python3
"""Put a JC2 beast's texture on a JC1 creature WITHOUT changing the JC1 creature's shape.

usage: jc1_reskin.py <jc1 family file> <jc2 model.bin> <out.npz>

The JC1 mesh is fitted onto the JC2 beast only to find, for every triangle corner,
where it lands on the JC2 surface (jc1_to_jc2 + corner_uv). Those UVs are then kept with
the ORIGINAL (subdivided) JC1 geometry. Output: P (JC1 positions), tris, cuv (per-corner
UVs into tex), tex (the JC2 texture), part, joint.
"""
import sys, os, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import jc1_family as F, jc1_to_jc2 as C, corner_uv as CU, jc2_gltf as J

def reskin(jc1, jc2_bytes):
    fitted = C.convert(jc1, jc2_bytes)                    # JC1 mesh shaped like the JC2 beast
    m2 = J.parse_model(jc2_bytes)
    tris, cuv, _ = CU.transfer(fitted, m2)                # where each corner lands on the JC2 skin
    m = F.load(jc1)
    P0, _, _, ex = C.subdivide(m['pos'], m['uv'], m['tris'], dict(part=m['part'], joint=m['joint']))
    return dict(P=P0, tris=tris, cuv=cuv, tex=m2['tex'], part=ex['part'], joint=ex['joint'])

if __name__ == '__main__':
    r = reskin(int(sys.argv[1]), open(sys.argv[2], 'rb').read())
    np.savez(sys.argv[3], **r); print('triangles', len(r['tris']))
