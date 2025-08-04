import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import argparse
import os

# same model definition
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

def fixed_hex_to_float(hex_str, frac_bits=7):
    """Convert 16-bit two's complement hex string to float given fractional bits."""
    val = int(hex_str, 16)
    if val & (1 << 15):  # negative
        val = val - (1 << 16)
    return val / (1 << frac_bits)

def load_uart_image(path, frac_bits=7):
    with open(path) as f:
        lines = [l.strip() for l in f if l.strip()]
    if len(lines) != 28 * 28:
        raise ValueError(f"Expected 784 pixels, got {len(lines)}")
    floats = [fixed_hex_to_float(h, frac_bits) for h in lines]
    img = torch.tensor(floats, dtype=torch.float32).view(1, 1, 28, 28)  # batch=1
    return img

def evaluate(model, image_tensor):
    model.eval()
    with torch.no_grad():
        out = model(image_tensor)
        probs = torch.softmax(out, dim=1)
        pred = out.argmax(dim=1).item()
        confidence = probs[0, pred].item()
    return pred, confidence

def main(args):
    # load model
    model = SmallCNN()
    state = torch.load(os.path.join(args.model_dir, "small_cnn.pth"), map_location="cpu")
    model.load_state_dict(state)

    # load image from uart_out.txt
    img = load_uart_image(args.uart_file, frac_bits=args.frac_bits)
    pred, conf = evaluate(model, img)
    print(f"Prediction from float model on received image: {pred} (confidence {conf:.3f})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Consume uart_out.txt and run inference")
    parser.add_argument("--uart-file", type=str, default="uart_out.txt", help="Path to uart_out.txt")
    parser.add_argument("--model-dir", type=str, default="./output", help="Directory with saved model")
    parser.add_argument("--frac-bits", type=int, default=7, help="Fractional bits used in fixed-point")
    args = parser.parse_args()
    main(args)