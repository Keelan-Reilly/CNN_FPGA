import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import argparse
import os
from pathlib import Path

# Same model definition
class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(8 * 14 * 14, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.fc1(x)

def load_image_mem_hex_bytes(path: str):
    """
    Load 28x28 image from a .mem text file with ONE 2-hex-digit byte per line (00..FF).
    Returns a torch tensor [1,1,28,28] in float32 scaled to [0,1].
    """
    with open(path) as f:
        lines = [l.strip() for l in f if l.strip()]
    if len(lines) != 28 * 28:
        raise ValueError(f"Expected 784 lines, got {len(lines)} in {path}")
    arr = np.array([int(h, 16) for h in lines], dtype=np.uint8).reshape(1, 1, 28, 28)
    return torch.tensor(arr, dtype=torch.float32) / 255.0

def load_image_bin_bytes(path: str):
    """
    Load 28x28 image from a raw .bin file containing exactly 784 bytes.
    Returns a torch tensor [1,1,28,28] in float32 scaled to [0,1].
    """
    data = np.fromfile(path, dtype=np.uint8)
    if data.size != 28 * 28:
        raise ValueError(f"Expected 784 bytes, got {data.size} in {path}")
    arr = data.reshape(1, 1, 28, 28)
    return torch.tensor(arr, dtype=torch.float32) / 255.0

def load_image_auto(path: str):
    """
    Auto-detect loader:
      - *.bin  -> raw 784 bytes
      - otherwise assume .mem with 2-hex-digit lines
    """
    ext = Path(path).suffix.lower()
    if ext == ".bin":
        return load_image_bin_bytes(path)
    else:
        return load_image_mem_hex_bytes(path)

def evaluate(model, image_tensor):
    model.eval()
    with torch.no_grad():
        out = model(image_tensor)
        probs = torch.softmax(out, dim=1)
        pred = out.argmax(dim=1).item()
        confidence = probs[0, pred].item()
    return pred, confidence

def main(args):
    # Load model
    model = SmallCNN()
    state = torch.load(os.path.join(args.model_dir, "small_cnn.pth"), map_location="cpu")
    model.load_state_dict(state)

    # Load image (bytes -> [0,1] float)
    img = load_image_auto(args.image_file)

    # Run float inference
    pred, conf = evaluate(model, img)
    print(f"Float-model prediction on input image: {pred} (confidence {conf:.3f})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run float inference on the same image sent to hardware")
    parser.add_argument("--image-file", type=str, default="./weights/input_image.mem",
                        help="Path to input image (input_image.mem with 2-hex-digit bytes per line, or input_image.bin)")
    parser.add_argument("--model-dir", type=str, default="./output", help="Directory with small_cnn.pth")
    args = parser.parse_args()
    main(args)