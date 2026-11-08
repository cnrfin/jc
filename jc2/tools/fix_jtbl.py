#!/usr/bin/env python3
"""Make m2c-ready copies of functions that use jump tables.
Adds .L labels for every jump-table target and appends the tables as .rodata.
usage: fix_jtbl.py <data.s> <funcs_dir> <out_dir> <func names...>"""
import re, sys, os
data, fdir, out = sys.argv[1:4]
tbl = {}; cur = None
for line in open(data):
    m = re.match(r'dlabel (jtbl_[0-9A-F]{8})', line)
    if m: cur = m.group(1); tbl[cur] = []; continue
    if cur:
        m = re.search(r'\.word 0x([0-9A-F]{8})', line)
        if m and line.strip().startswith('/*'):
            v = int(m.group(1), 16)
            if 0x100000 <= v < 0x4e3e60: tbl[cur].append(v)
            else: cur = None
        elif line.startswith(('dlabel', 'nonmatching', 'enddlabel')) or (line.strip() and not line.strip().startswith('/*')):
            if not line.startswith('enddlabel'): cur = None
os.makedirs(out, exist_ok=True)
for name in sys.argv[4:]:
    src = open(os.path.join(fdir, name + '.s')).read().splitlines()
    used = sorted(set(re.findall(r'(jtbl_[0-9A-F]{8})', '\n'.join(src))))
    targets = {t for j in used for t in tbl.get(j, [])}
    lines = []
    for l in src:
        m = re.match(r'\s*/\* [0-9A-F]+ ([0-9A-F]{8}) ', l)
        if m and int(m.group(1), 16) in targets and not any(x.startswith('.L%s:' % m.group(1)) for x in lines[-2:]):
            lines.append('.L%s:' % m.group(1))
        lines.append(l)
    lines += ['', '.section .rodata']
    for j in used:
        lines.append('glabel ' + j)
        lines += ['    .word .L%08X' % t for t in tbl.get(j, [])]
    open(os.path.join(out, name + '.s'), 'w').write('\n'.join(lines) + '\n')
