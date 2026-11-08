# Jade Cocoon 2 — first pass

Disc: **SLUS_203.09** (NTSC-U, `VER = 1.01`), 1,513,456 sectors, 3,099,557,888 bytes,
SHA-1 `1677169b4bde5fcea6213aa005f8e8c3b630c59b`. No game data is committed here; the
image lives in the private `cnrfin/jc2-dump` repo as 32 split pieces.

## Disc layout

| path | size | what |
|------|------|------|
| `SLUS_203.09` | 6.1 MB | main ELF (EE, R5900, eabi64). Symbol table is empty |
| `CDDATA.000` | 1.34 GB | the game archive |
| `CDDATA.LOC` | 84 KB | its index |
| `VOICE/VOICE.BIN` | 669 MB | voice audio |
| `MOV/*.PSS` | ~1.1 GB | 32 FMVs. Note the names: ARMOR, BIRD, DOG, FISH, GINUI, INSECT, LUCHU, PLANT, each with A/B/C variants |
| `IRX/` | | IOP modules (sound, memcard, pad, HDD) |

## CDDATA archive — CONFIRMED

`CDDATA.LOC` is `u32 count (5249), u32 0`, then `count` records of four u32:
`(flag, startSector, sizeBytes, nSectors)`. Record 0's first two words are the header, so
record 0 starts at sector 0. Sectors are 2048 bytes. `nSectors == ceil(size/2048)` for all
5,249, and no records overlap. `tools/split_cddata.py` splits it:

```bash
python3 -I jc2/tools/split_cddata.py CDDATA.LOC CDDATA.000 cddata/
```

What is in it (5,095 non-empty files):

| count | type |
|------:|------|
| 1,383 | `MWo3` code modules. MIPS code with a 0x80 header (`MWo3`, load address `0x940000`, size, name such as `m2esAE05.bin`, `SC_HATCH.BIN`, `BTLEF_M_FIRE_S1.BIN`). Event scripts, scenes, battle effects. They load into the ELF's empty `0x940000` segments |
| 881 | `XMDL` models with a `SKIN` section: rigged characters, the beasts should be here |
| 718 | `XMDL` models without `SKIN`: props, maps |
| 124 | bare `TIM2` textures (most models also embed their own TIM2) |
| 1,989 | other: small tables, animation and parameter data, not yet identified |

## The model format — started

A model file is `u32 n; u32 offset[n]` followed by the sections. The geometry section
starts with ASCII tags `0IVX00.1`, `LDMX00.2` (or `LDMX00.1`), `NORM00.2`, then `SKIN00.1`
on skinned models. The body is PS2 VIF/GIF packet data (UNPACK `0x6C` commands, GIF
tags, float vertex data), so meshes are pre-built DMA chains, not a TMD-style list. A
typical beast-sized file (e.g. 1691) is 2 sections: geometry 0x58E0 bytes + a TIM2.

Each geometry section is a run of VIF batches: `UNPACK V4-32` (header, addr 0),
`UNPACK V3-32` (positions, addr 4, `num` = vertex count), `UNPACK V2-32` (UVs),
`UNPACK V4-8` (colour/normal). Here, one batch means one strip of up to `num` vertices.

## Do beasts share topology like JC1? — NO (CONFIRMED)

`tools/xmdl_sig.py` takes each model's list of V3-32 vertex counts (one entry per batch)
and groups the 881 skinned models by it:

- **195 distinct signatures.** The largest group is 20 files.
- Inside a group, the files are **the same beast**: the same mesh is stored once per element
  or stage (the pattern repeats every 120 files, e.g. 1821/1825/1829, 1941/1945/1949,
  2061..., 2181...). Positions are byte-identical, or at most 3 variants differing by
  <0.08 units.
- So different species never share a vertex layout. JC1 had 49 species on one mesh,
  which is what made its per-vertex merge possible. JC2 has no such family; a JC1-style
  merge needs a template mesh fitted onto each beast (see the plan in the chat log /
  top-level discussion).

### How the beast models are laid out — PATTERN CONFIRMED

The skinned models are one contiguous run, files **1785–2638**. From 1785 they come in
groups of 12, and in 31 of the first 40 groups slots 0, 4 and 8 are the same mesh.
Blocks of 120 files (10 groups) repeat with 283 of 360 meshes unchanged.

What this is NOT: merging. In JC2 a merge does not change appearance at all; it only
passes on attacks/abilities. Each beast belongs to a species (Chu, Og, Slimy...) with
four elemental variants (fire, earth, wind, water) that differ in colour and stats. The
repeats above are those variants, plus the growth forms, stored as separate files that
share a mesh. Which slot is which element/form is still to be mapped.

## Executable

- Entry `0x100008`, `$gp = 0x6CBC70`. Main code `0x100000–0x4E3E60`, data after it,
  a second small code block at `0x6B2B00–0x6B930C`, bss to `0x8C2800`.
- Disassembled with spimdisasm (R5900): **9,855 functions**.

```bash
python -m spimdisasm singleFileDisasm SLUS_203.09 disasm/asm \
  --endian little --instr-category r5900 --abi EABI64 --arch-level MIPS3 --compiler EEGCC \
  --vram 0x100000 --start 0xb000 --end 0x3eee60 --data-start 0x3eee60 --data-end 0x5cf680 \
  --gp 0x6cbc70 --split-functions disasm/funcs --function-info disasm/functions.csv
```

- Merge UI strings: `"Merge OK"` at `0x50C9C0`, `"Merge fee"` at `0x50ED70`. Neither is
  loaded by a direct `lui/addiu` pair, so they are reached through a pointer table.
  Following that table is the way into the merge code.
- **Decompiled to C with m2c** (`-t mipsee-gcc-c`): 9,818 of 9,855 functions. Switch
  statements need their jump tables: `tools/fix_jtbl.py` copies the table from the data
  section into the function file and labels its targets. The 37 that still fail are
  mostly data that the disassembler took for code.
- The screen that uses the "Merges:" layout is set up by `func_0018AFA0`. It installs
  `func_0018B0C0`, a switch with about 20 states (`func_0018B330` ... `func_0018C140`).
  The same layout is also used by `func_00241A70` and `func_004358C0`.
- Beast species names (Getchu, Volkaiser, Mothmanti...) sit around `0x688E00`. Item
  records (stride 0x48, e.g. "Cocoon containing a" + species) are at `0x52CEF0`.
