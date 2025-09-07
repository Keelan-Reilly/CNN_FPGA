import torch
import numpy as np

# load sample batch
d = torch.load("./output/sample_batch.pt")
img = d["images"][0].squeeze(0).numpy()  # [28,28] in [0,1]
bytes_ = np.clip(np.rint(img * 255), 0, 255).astype(np.uint8)

# save as .mem file (hex format)
with open("weights/input_image.mem", "w") as f:
    for b in bytes_.flatten():
        f.write(f"{b:02X}\n")

# raw bytes (cleaner for a C++ TB)
open("weights/input_image.bin", "wb").write(bytes_.tobytes())
print("Wrote input_image.mem / input_image.bin")
