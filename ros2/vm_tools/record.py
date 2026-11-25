"""Screenshots of display :0 at a fixed rate.  python3 record.py OUT_DIR SECONDS FPS  (one file: OUT_DIR given as *.png)"""
import os
import shutil
import sys
import time

from PIL import ImageGrab

out = sys.argv[1]
if out.endswith(".png"):
    ImageGrab.grab(xdisplay=":0").save(out)
    print("saved", out)
    sys.exit()
secs, fps = float(sys.argv[2]), float(sys.argv[3])
shutil.rmtree(out, ignore_errors=True)
os.makedirs(out)
t0, k = time.time(), 0
while time.time() - t0 < secs:
    ImageGrab.grab(xdisplay=":0").save(os.path.join(out, f"f{k:03d}.png"))
    k += 1
    time.sleep(max(0.0, t0 + k / fps - time.time()))
print("frames", k)
