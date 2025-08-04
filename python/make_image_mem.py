import torch
from torchvision import transforms
import numpy as np

# load sample batch
d = torch.load("./output/sample_batch.pt")
img = d['images'][0]  # first image, shape [1,28,28]
# flatten and quantise with 7 fractional bits
frac_bits = 7
flat = img.view(-1).numpy()
fixed = []
for v in flat:
    q = int(round(v * (1 << frac_bits)))
    if q < 0:
        q = (1 << 16) + q
    fixed.append(q)
with open("weights/image.mem", "w") as f:
    for v in fixed:
        f.write(f"{v:04X}\n")  # 16-bit hex
print("Written weights/image.mem")