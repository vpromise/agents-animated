"""Writes assets/backdrop.png for "Agents, Animated": warm charcoal, a soft vignette, a fine grid."""
import sys
import numpy as np
from PIL import Image
W, H = 1920, 1080
out = sys.argv[1]
y, x = np.mgrid[0:H, 0:W].astype(np.float32)
base = np.array([20, 18, 15], np.float32)            # #14120F
glow = np.array([46, 34, 26], np.float32)            # warm centre lift
r = np.sqrt(((x - W * 0.5) / (W * 0.62)) ** 2 + ((y - H * 0.45) / (H * 0.70)) ** 2)
t = np.clip(1 - r, 0, 1) ** 2
img = base + (glow - base) * t[..., None] * 0.55
edge = np.clip(r - 0.75, 0, 1)                         # darker corners
img *= (1 - 0.35 * edge)[..., None]
step = 1920 / 14.222 * 0.5                             # a line every half unit
minor = ((np.abs((x + 0.5) % step) < 1.0) | (np.abs((y + 0.5) % step) < 1.0))
major = ((np.abs((x + 0.5) % (step * 4)) < 1.2) | (np.abs((y + 0.5) % (step * 4)) < 1.2))
img += minor[..., None] * np.array([7, 6, 5], np.float32) * (0.6 + 0.4 * t[..., None])
img += major[..., None] * np.array([6, 5, 4], np.float32) * (0.6 + 0.4 * t[..., None])
Image.fromarray(np.clip(img, 0, 255).astype(np.uint8)).save(out)
print("wrote", out)
