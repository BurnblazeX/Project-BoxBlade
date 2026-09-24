# Generates the 12x12 glass texture sets in the LabPBR layout - clear, stained
# and frosted: <name>.png (albedo), <name>_n.png (normal), <name>_s.png
# (specular).
# Deterministic, so re-running reproduces the files. Pure standard library.
#
# For glass the albedo is not a diffuse colour - glass has none - but the TINT
# its transmitted light takes on (gpu.js glassNode). Near white with a faint
# cool cast, a slightly deeper rim where the glass reads thicker, and two
# highlight streaks. Alpha stays 255: glass is not alpha-blended here, it is a
# surface whose shading traces through it.
import struct, zlib, os

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

INTERIOR = (232, 244, 246, 255)
RIM = (196, 222, 226, 255)
STREAK = (250, 254, 255, 255)
streaks = {(3, 2), (2, 3), (4, 3), (3, 4), (8, 7), (9, 8)}

def glass(x, y):
    if x == 0 or y == 0 or x == N - 1 or y == N - 1:
        return RIM
    if (x, y) in streaks:
        return STREAK
    return INTERIOR

write_png('terrain_glass.png', grid(glass))
# Flat: glass is smooth. A generated normal (from the albedo) would put bumps
# along the rim and the streaks.
write_png('terrain_glass_n.png', grid(lambda x, y: (128, 128, 255, 255)))
# LabPBR specular: R smoothness 255 (polished), G F0 10 (0.04, ior ~1.5),
# B 0 (no porosity or SSS), A 255 (no emission).
write_png('terrain_glass_s.png', grid(lambda x, y: (255, 10, 0, 255)))

# --- Stained glass: four panes in lead ---
# Each pane's colour is what light through it becomes - in the view and in the
# shadow it casts (gpu.js reads the texel a ray enters). The lead lines are a
# near-black tint: almost nothing passes, so they draw dark lines in the light.
LEAD = (38, 38, 44, 255)
PANES = {(0, 0): (214, 58, 52, 255), (1, 0): (66, 108, 226, 255),
         (0, 1): (236, 200, 64, 255), (1, 1): (72, 190, 104, 255)}
def stained(x, y):
    if x in (0, 5, 6, 11) or y in (0, 5, 6, 11):
        return LEAD
    return PANES[(x // 6, y // 6)]
write_png('terrain_stainedGlass.png', grid(stained))
write_png('terrain_stainedGlass_n.png', grid(lambda x, y: (128, 128, 255, 255)))
write_png('terrain_stainedGlass_s.png', grid(lambda x, y: (255, 10, 0, 255)))

# --- Frosted glass ---
# Pale and even. The frost IS the normal map: a random tilt per texel, so the
# rays bending through and off it scatter into a stable grain (gpu.js reads it
# at the locked texel, as every normal map is read). Authored here as noise;
# draw it instead for a pattern. Smoothness 155 is LabPBR alpha
# (1 - 155/255)^2 = 0.15 - spread highlights, still smooth enough to trace.
import random
rnd = random.Random(11)
FROST_TILT = 40            # of 127: about 0.3 either way, in x and y
def frosted(x, y):
    v = 236 + ((x * 7 + y * 13) % 5) - 2
    return (v, v + 2, v + 3, 255)
frost_n = [[(128 + rnd.randint(-FROST_TILT, FROST_TILT),
             128 + rnd.randint(-FROST_TILT, FROST_TILT), 255, 255) for x in range(N)]
           for y in range(N)]
write_png('terrain_frostedGlass.png', grid(frosted))
write_png('terrain_frostedGlass_n.png', frost_n)
write_png('terrain_frostedGlass_s.png', grid(lambda x, y: (155, 10, 0, 255)))

print('wrote terrain_glass, terrain_stainedGlass and terrain_frostedGlass (albedo, _n, _s)')
