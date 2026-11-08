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

A model file is `u32 n; u32 offset[n]` followed by the sections. A beast (e.g. 1821) has
4: a 16-byte flags block, the geometry (`0IVX00.1` / `LDMX00.1` / `NIKS00.1` tags), one
`TIM2` texture, and an 11-entry block that looks like animation data.

**Decoded and verified** (`tools/xmdl.py`, `tools/tim2.py`, `tools/jc2_gltf.py`). A beast
exports to glTF that passes the Khronos validator with 0 errors and animates correctly in
three.js.

### Geometry section layout (from the loader `func_00175EB0`)

The counts are at `+0x20`: `objects, nodes, meshes, materials, parts, joints`. The tables
follow from `+0x40` in this order:

| table | stride / size | what |
|---|---|---|
| objects | 0x20 | |
| nodes | 0x50 | `u32 type; u32; u32; s16 a; s16 depth; s16 parent; s16 child; u32 sibling;` then `T` @+0x20, `R` (radians) @+0x30, `S` @+0x40. Node 0 is the root/mesh holder (type 0x32) |
| meshes | 0x30 + 8·parts (+8 if odd) | |
| materials | 0x60 | |
| parts | 0x30 + (qwc+1)·16 | the VIF DMA chain below |
| joints | 0x90 | one per node: `u32 a; u32 flags; u32 child; u32 sibling; f32 roll(deg) @+0x10;` T/R/S copy @+0x20; **inverse bind matrix** (4x4, row-vector) @+0x50 |

- **Rotation order:** local matrix = `T · Rz · Ry · Rx · S` (`func_00144170`). Checked
  against the stored inverse bind matrices: exact.
- **Joint flags** (`+0x04`):

  | flag | meaning |
  |---|---|
  | `1` | gets a skinning matrix; palette slot = its order among flagged joints (`func_00143D30`) |
  | `2` | does not inherit parent rotation (feet stay level) |
  | `4` | a=4 chain is 3-joint IK |
  | `0x10` | IK bend direction |
  | `0x80` | yaw-first aim |
  | `0x400` | ground marker (parent x,z at y=0) |

- **Geometry** is a VIF DMA chain. Each batch is:
  - `REF` tags, which the loader (`func_0014C4E0`) patches to
    `matrixBuffer + idx*0x80`, `idx` = byte 2 of the tag's `STCYCL` word.
  - A `CNT` tag whose spare address word lists the batch's 4 palette slots.
  - `UNPACK`s with `STCYCL 4,1`, giving four quadwords per vertex:
    - `0x45` V3-32 position (bind pose, world space)
    - `0x46` V2-32 UV (0..1, no flip)
    - `0x47` V4-16 normal (/32767)
    - `0x48` V4-8 weights (one byte per palette slot, sum 255)
  - Each batch is one triangle strip.
- **TIM2:** standard, 8-bit indexed with a CSM1-swizzled 256-colour CLUT, alpha 0..128.

### Animation (section 3 of the model file)

The section is `u32 n` then (offset, size) pairs, one per clip. A clip has a header of
`(offset, size)` tables; the second table is the channel table: `u16 count; u16 frames
(0x8000 = loop)`, then `count` × `{u16 node; u16 mask; u32 valuesOff; u32 keysOff}`
(offsets from the table start; 0 = no data).

- **Mask bits** are read 13 → 5 (`func_00145460`): 13–11 scale XYZ, 10–8 rotation XYZ,
  7–5 translation XYZ.
- **Keys:** per component, a `u16` header (`kind = hd>>13`, `n = hd & 0x1FFF`) and, for
  `kind != 0`, `n` frame numbers. Frame 0 is implicit.
- **Values (f32) per kind:**

  | kind | layout |
  |---|---|
  | 0 | n constants |
  | 1, 2, 3 | n+1 values: 3 = linear, 1 and 2 = step |
  | 4, 5 | 2n+1: (value, tangent) pairs, last tangent omitted |
  | 6 | 3n+1: `v0, out0, (in, v, out)…, (in, v)` |
  | 7 | 2n+2: (value, tangent) per key |

  Tangents are per segment (Hermite).

### Beasts are animated with IK (`func_00142B00`, `func_00143960`, `func_00143520`)

Joints come in `a` = 4/5/6 runs.
- **The target:** an `a`==6 joint's translation slot holds a **model-space target
  position**, not a local offset. At rest it is the bind world position.
- **The aim:** the `a`==4 joint takes its animated rotation as normal. The `a`==5 joint
  after it is then aimed so that local +X points at the target:
  `Rz(atan2(y,x)) · Ry(−atan2(z, r)) · Rx(roll)`.
- **3-joint chains (flag 4):** the hip and knee are then bent with the law of cosines,
  using the thigh and shin lengths (the X translation of the next two joints).
- **The joint after the IK chain** uses its own animated rotation, and flag 2 keeps it
  world-aligned.

`tools/jc2_gltf.py` implements all of this and bakes every frame into the glTF.

## Do beasts share topology like JC1? — NO (CONFIRMED)

Parsed exactly with `tools/xmdl.py`: 879 skinned models, **194 distinct meshes**
(254 distinct position sets) and 61–96 nodes each, so there is no shared rig either.
(The earlier `tools/xmdl_sig.py` heuristic gave 195; same answer.)

- The largest group of identical meshes is 20 files.
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

## Rebuilding JC2 beasts on the JC1 family mesh (the JC1 way) — PROTOTYPE

JC1's 49 family creatures share one mesh (1,464 vertices), **one UV layout** (checked:
833, 845, 860 and 867 have identical UV sets) and one 25-bone rig. A merge blends
vertices and bone lengths, ORs the part groups and takes **one** parent's texture,
hue-rotated. To get JC1-style merges in JC2, each mergeable JC2 species is rebuilt as a
member of that family:

| tool | what |
|---|---|
| `tools/jc1_family.py` | loads a JC1 family creature from `models/family`, assembled in bind pose (Merge Studio's `poseBones`) |
| `tools/rig_roles.py` | labels a JC2 rig's IK chains: four legs (hip → mid → foot), neck, head, tail |
| `tools/jc1_to_jc2.py` | rebuilds a JC2 beast on the JC1 mesh (steps below) |
| `tools/render_jc1_merge.py` | originals vs rebuilds, plus the merge sweep with one texture |
| `tools/rebuilt_to_glb.py` | rebuilt beast as a textured GLB for hand-finishing; hidden part groups go in a separate mesh |

`jc1_to_jc2.py` works in these steps:
1. Start from the closest JC1 creature.
2. Subdivide once (1,464 → about 3,300 vertices on the visible groups).
3. Warp by matched bone roles: JC1 bones 3–5 / 6–8 are the front legs, 17–19 / 20–22
   the back legs, 9–10 neck→head, 16 → 23 → 24 the tail.
4. Wrap the visible part groups onto the JC2 surface.
5. Bake the JC2 texture into the shared JC1 UV layout at 2×.

First test: 1821 and 1858 both start from Marrdreg (845).
- **1821 works:** it rebuilds recognisably, and a 50% merge using one texture reads as
  a real in-between creature.
- **1858 is rough:** its spikes, mace tail and splayed feet have no matching part on
  Marrdreg.
- **The fix is per-species feature mapping:** choose which JC1 part group carries ears,
  horns or spikes (as JC1 reuses arm bones for heads), plus eye and mouth landmarks.
  Then hand-finish in Blender.

## Option 2: merges that keep the original art — PROTOTYPE

The JC1-mesh rebuild (above) degrades the art: automatic wrapping can't place
features, and JC1's quality came from hand-made art. Option 2 leaves every JC2 beast
exactly as Genki made it and makes the merge visible through three channels.

The prototype is `tools/option2_merge.py base.bin material.bin out.png [swap:a:b | add:a:b:anchor]`.

1. **Proportions.** Matched leg lengths, relative to each beast's mean leg, and overall
   body size are blended into the base skeleton. The base mesh is re-skinned with its
   own weights.
2. **Hue.** The base texture's hue is rotated toward the material's mean hue (JC1's rule).
3. **Part transplant** (`tools/beast_parts.py`). A part is a skeleton subtree, found from
   each vertex's dominant joint, and it keeps its own texture.
   - `swap:a:b` replaces the base's subtree at node `a` with the material's subtree at `b`,
     mapping bone frame to bone frame.
   - `add:a:b:anchor` attaches the material's subtree `b` to base node `a`, using
     translation and scale only (all beasts face −Z with −Y up).

Examples:
- 1821 + 1858 `swap:24:6` gives the boar 1858's mace tail.
- 1858 + 1821 `add:31:46:36 add:31:52:36` gives 1858 the boar's ears.

**Facing correction:** these beasts face −Z. In `rig_roles.py`, "head" and "tail" are
therefore the opposite ends of what the names say; the matching is still consistent.

**What the game will need:**
- A transplanted part becomes extra DMA packets rigidly skinned to the base's anchor
  palette slot.
- Proportions mean patching node offsets. Because JC2 animates legs with IK, foot
  targets must be scaled about the hip as well.

### Reusing the original textures (per-corner UVs) — PROTOTYPE

`tools/corner_uv.py rebuilt.npz model.bin out.npz` gives a shared-mesh rebuild per-corner
UVs into the beast's **original** texture, with no baking:
- Each corner takes the UV of the nearest point on the original surface.
- All three corners of a triangle come from the same UV island of the original
  (islands = triangles sharing vertices with equal position and UV).

A merge then takes one parent's look (its texture and its corner UVs) and blends the
geometry; `tools/render_cuv_merge.py` renders this.

Result on 1821 (from JC1 845) and 1856 (from JC1 867):
- **Texture detail is preserved** (pixel-exact, no blur).
- **Wrong shape correspondence still breaks it.** Eyes smear, 1821 loses an ear,
  1856's spiral tail is missed, and in the merges features drift because vertex *i* is
  not the same feature on both beasts.
- **Conclusion:** texture reuse is solved; per-species shape correspondence is the
  remaining (manual) work.

### Two-beast merge family (single mesh built from the two references) — PROTOTYPE

`tools/pair_family.py A.bin B.bin out.npz --a-only NODES --b-only NODES` and
`tools/render_pair.py`.

The single mesh is A's own mesh plus B's unique parts. A's ears and tail are a group
hidden on B, B's mace tail a group hidden on A (JC1's part-group rule). Only the shared
body morphs into B's shape:
- through a joint map from A's bones to B's (legs by arc length, spine/head/tail chains),
- then a gentle snap onto B's surface.

Each beast keeps its original texture; B's look on the shared body uses per-corner UVs.

Result for 1821 + 1858 (`--a-only 46,52,24 --b-only 6`):
- **Member A is exactly the original.**
- **Member B is recognisable**: body, legs, mace tail.
- **The look-A merges read as hybrids** at 25–75%.
- **Still weak:** B's horned head and jaw are not reproduced (the head is morphed, not
  grouped), and legs distort near 100%.

Second round (`--a-head` / `--b-head`):
- **Heads are "look" parts:** shown only for the parent that supplies the look, like the
  texture. They move as rigid pieces, anchored at the ring where each head meets its body.
- **Inherited parts are hue-tinted** toward the chosen look.
- **Pairs:** 1821 + 1858 (`--a-only 46,52,24 --b-only 6 --a-head 36 --b-head 31`) and
  1821 + 1856 chameleon (`--b-only 6 --b-head 31`).
- **Result:** the look-A rows hold up across the sweep. The look-B rows, which put B's
  texture on the A-derived shared body, are the weakest: shards where the transferred UVs
  stretch.
- **A tried and rejected change:** rotating bones to follow the other beast's bone
  directions flattened the legs (1858's limb paths end in splayed toes).

Third round: **the body comes from the look parent** (`tools/render_pair2.py`). The family is
built twice, once on each beast's own mesh (A + B's parts, and B + A's parts). A merge
uses the family of whichever parent supplies the look, so both looks start from exact
original geometry and texture, and only the other parent's shape is a morph target.
- **Both members are exact.** The 50–100% rows of B's look are now clean (e.g. the
  chameleon with the boar's ears).
- **Remaining roughness is only at the far end of each row:** a body squeezed into the
  other beast's shape.
- **Storage scales with the number of species:** every species needs a morph target on
  every other species' mesh (N·(N−1) targets).

## Writing JC2 model files; JC1 creatures in JC2 format

`tools/xmdl_write.py` writes a JC2 model file (section 0 + XMDL geometry + TIM2) from a
node list, rigidly skinned vertices, triangles and a texture.
- **Layout:** follows the loader (`func_00175EB0`) and the game's own batch structure:
  REF matrix uploads, CNT with palette slots, `STCYCL 4,1` vertex UNPACKs at
  `0x45`–`0x48`, GIF tags, `MSCNT`.
- **Templates:** fields of unknown meaning (materials, GIF tag words, section 0) are
  copied from a template beast.
- **Strips:** triangles are joined into strips with degenerate links, at most 4 palette
  slots and 30 vertices per batch.
- **Texture:** 256-colour TIM2 with a CSM1 palette and alpha 0..128.

`tools/jc1_to_xmdl.py <jc1 family file> <template.bin> <out.bin>` converts a JC1 family
creature:
- **Skeleton:** the 25 bones become JC2 nodes (same Z·Y·X Euler order), every one a
  palette joint.
- **Pose and placement:** the frame-0 pose, scaled to height 2.0, turned to face the JC2
  way, lowest point on y = 0.
- **Visibility:** hidden part groups are left out.
- **Texture:** the JC1 atlas padded to 256×256.

**Verified:** the files read back through the same parser and exporter as the game's
own files. Every non-degenerate triangle round-trips; FK matches the written bind
matrices exactly; the exported GLB validates with 0 errors. 845, 867, 833 and 907
converted.

**Not yet verified:** loading in the game itself (needs the archive repack and PCSX2),
and animations (none written yet).

### JC1 family species that exist in JC2

- **Species list:** JC2's beast list (item records "Cocoon containing a …" at
  `0x52CEF0`, about 125 names) is the Divine Beast forms. JC1 lines reappear there,
  e.g. Marrdreg → Mardreck, Patdreg → Patadreck and the *-lchu* line.
- **Textures:** every form's model file (1785–2284) carries its own full-colour texture.
- **Greyscale models:** files 2495–2638 are greyscale copies of those meshes, probably
  the metal/statue variants.

Same-design pairs fitted with `jc1_to_jc2.py` + `corner_uv.py` (JC1 mesh shaped onto the
JC2 beast, wearing the JC2 texture):
- **Good:** Geenwee 856 → 1846 and Yolga 920 → 1843.
- **Messy:** Ehsp 867 → 1855 (JC1 stands tall, JC2 crouches).
- **Fails:** serpents (no legs → too few landmarks).
- **Merge 1846 ↔ 1843 on the shared mesh:** fine near each parent, rough at the far end
  of each sweep.

### The beast table: model file -> name — CONFIRMED

The beast records are 260 bytes each, from `0x4E9A18`:
- `u16 index` @+0x90, `u16 model` @+0x92 (archive file = model + 1305), name @+0x78.
- The first 120 records name every form in the first element block (1785–1904); the
  other three blocks are the element colourways of the same forms (file + 120·k).
- The full list is in `jc2_beast_names.json`. Families include the Luch / *-chu* line
  (1785–1796) and the boar's group: Vordrech 1821, Redoble, Mardreck, Blumen, Drech,
  Patadreck, Radreck, Paradrager, Tikidrech, Tiki, Tiral, Giralooza.

`jc1_jc2_species.json` lists the JC1 family creatures confirmed by eye to be the same
design in JC2, with how well each rebuild (JC1 mesh + JC2 texture) came out:
- **Good / decent (6 of 8):** Raddreg, Patdreg, Geenwee, Yolga, Marrdreg, and middling
  Dogle.
- **Poor:** Maryen (thin branches) and Ehsp (pose mismatch).
- **Misleading names:** Tweengo/Twingo, Ticker/Tiki and Ehsp/Chamelemon are different
  designs.
