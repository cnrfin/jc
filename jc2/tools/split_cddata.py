#!/usr/bin/env python3
"""Split Jade Cocoon 2 (SLUS_203.09) CDDATA.000 using its CDDATA.LOC index.

CDDATA.LOC: u32 count, u32 0, then count records of 4 x u32. Record 0's first two
words are the header; every record is (flag, startSector, sizeBytes, nSectors).
Usage: split_cddata.py <CDDATA.LOC> <CDDATA.000> <outdir>
"""
import json, os, struct, sys

def records(loc):
    w = struct.unpack('<%dI' % (len(loc) // 4), loc)
    n = w[0]
    recs = [list(w[i:i + 4]) for i in range(0, n * 4, 4)]
    recs[0][0] = recs[0][1] = 0
    return recs

def ext_for(head):
    if head[:4] == b'MWo3': return 'mwo3'
    if head[:4] == b'TIM2': return 'tm2'
    return 'bin'

def main(locp, datp, out):
    os.makedirs(out, exist_ok=True)
    recs = records(open(locp, 'rb').read())
    index = []
    with open(datp, 'rb') as f:
        for i, (flag, sec, size, nsec) in enumerate(recs):
            f.seek(sec * 2048)
            data = f.read(size)
            ext = ext_for(data[:4])
            name = '%04d.%s' % (i, ext)
            if size:
                open(os.path.join(out, name), 'wb').write(data)
            index.append(dict(index=i, flag=flag, sector=sec, size=size, file=name if size else None,
                              magic=data[:4].hex()))
    json.dump(index, open(os.path.join(out, 'index.json'), 'w'), indent=0)
    print('wrote', sum(1 for r in index if r['file']), 'files')

if __name__ == '__main__':
    main(*sys.argv[1:4])
