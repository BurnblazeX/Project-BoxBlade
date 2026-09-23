# Generates the 12x12 texture sets for the two reflective test materials, in
# the LabPBR layout: <name>.png albedo, <name>_n.png normal, <name>_s.png
# specular. Deterministic (fixed seed), so re-running reproduces the files.
# Pure standard library - no PIL on this machine.
import random, struct, zlib, os

OUT = os.path.join(os.path.dirname(__file__), '..', 'assets', 'textures')
N = 12

def write_png(path, px):   # px: rows top-down of (r, g, b, a)
    raw = b''.join(b'\0' + bytes(c for p in row for c in p) for row in px)
    def chunk(t, d):
        return struct.pack('>I', len(d)) + t + d + struct.pack('>I', zlib.crc32(t + d) & 0xffffffff)
    png = (b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('>IIBBBBB', N, N, 8, 6, 0, 0, 0))
           + chunk(b'IDAT', zlib.compress(raw, 9)) + chunk(b'IEND', b''))
    open(os.path.join(OUT, path), 'wb').write(png)

def grid(f):
    return [[f(x, y) for x in range(N)] for y in range(N)]

clamp = lambda v: max(0, min(255, int(round(v))))
FLAT = (128, 128, 255, 255)   # LabPBR: flat normal, no AO, surface height

# --- Polished black marble (wall). Dielectric. ---
# Near-black stone with pale veins. Polished: smoothness high, F0 of a
# typical dielectric (~0.04 -> green 10), not porous, no SSS, no emission.
rnd = random.Random(7)
vein = set()
for start in [(0, 2), (3, 11), (11, 5)]:
    x, y = start
    for _ in range(14):
        vein.add((x % N, y % N))
        x += rnd.choice([1, 1, 0]); y += rnd.choice([-1, 0, 1, 1])
def marble(x, y):
    if (x, y) in vein:
        v = 150 + rnd.randint(-15, 15)
        return (v, v, clamp(v * 1.04), 255)
    v = 26 + rnd.randint(-5, 5)
    return (v, clamp(v * 1.05), clamp(v * 1.15), 255)
write_png('terrain_marble.png', grid(marble))
write_png('terrain_marble_n.png', grid(lambda x, y: FLAT))
# Veins are a touch rougher than the polished field around them.
write_png('terrain_marble_s.png',
          grid(lambda x, y: (225 if (x, y) in vein else 245, 10, 0, 255)))

# --- Polished iron plate (floor). Metal. ---
# Two plates per block (6 px), seams at the plate edges, a rivet in each
# plate's corner. Green 230 = LabPBR iron; blue 0 for metals.
def plate(x, y):
    seam = x % 6 == 0 or y % 6 == 0
    rivet = x % 6 == 1 and y % 6 == 1
    if seam:  return (70, 72, 78, 255)
    if rivet: return (190, 192, 198, 255)
    v = 150 + ((x * 7 + y * 13) % 5) - 2
    return (v, v + 2, v + 6, 255)
write_png('terrain_ironPlate.png', grid(plate))
def plate_n(x, y):
    # Seams bevel down into the gap; rivets bulge. Rows are top-down here,
    # and DirectX green is higher = tilted down the texture.
    if x % 6 == 0: return (128, 128, 180, 150)
    if y % 6 == 0: return (128, 128, 180, 150)
    if x % 6 == 1: return (90, 128, 240, 255)
    if x % 6 == 5: return (166, 128, 240, 255)
    if y % 6 == 1: return (128, 90, 240, 255)
    if y % 6 == 5: return (128, 166, 240, 255)
    return FLAT
write_png('terrain_ironPlate_n.png', grid(plate_n))
# Seams are rough and dielectric grime; the plate itself near-mirror iron.
write_png('terrain_ironPlate_s.png',
          grid(lambda x, y: (90, 10, 0, 255) if (x % 6 == 0 or y % 6 == 0)
                            else (240, 230, 0, 255)))
print('ok')
