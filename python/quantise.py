import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import os
import numpy as np

# Fixed-point conversion helpers
def to_fixed(x, frac_bits):
    """Convert float to signed fixed-point integer representation."""
    return int(round(x * (1 << frac_bits)))

def from_fixed(x, frac_bits):
    """Convert fixed-point integer back to float."""
    return x / (1 << frac_bits)

def quantise_tensor(tensor, frac_bits, bit_width=16):
    """Quantise a PyTorch tensor to fixed-point integers with optional saturation."""
    flat = tensor.cpu().numpy().flatten()
    q = []
    for val in flat:
        f = to_fixed(val, frac_bits)
        # Saturate into bit width (two's complement)
        max_val = (1 << (bit_width - 1)) - 1
        min_val = - (1 << (bit_width - 1))
        if f > max_val:
            f = max_val
        if f < min_val:
            f = min_val
        if f < 0:
            f = (1 << bit_width) + f  # two's complement representation for negative
        q.append(f)
    return np.array(q, dtype=np.int32).reshape(tensor.shape)

# Same architecture to load weights
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

def save_mem(arr, path, bit_width=16):
    """Write fixed-point array to .mem file in hex (one value per line)."""
    with open(path, "w") as f:
        flat = arr.flatten()
        for v in flat:
            if v < 0:
                # Should already be converted to two's complement above, but safe guard
                v = (1 << bit_width) + v
            f.write(f"{v:0{bit_width//4}X}\n")  # Hex with leading zeros

def main(args):
    # Load model structure and weights
    model = SmallCNN()
    state_dict = torch.load(os.path.join(args.model_dir, "small_cnn.pth"), map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()

    # Quantisation parameter
    weight_frac = args.weight_frac  # fractional bits for weights

    # Quantise all relevant parameters
    q_conv_w = quantise_tensor(model.conv1.weight.data, weight_frac)
    q_conv_b = quantise_tensor(model.conv1.bias.data, weight_frac)
    q_fc_w = quantise_tensor(model.fc1.weight.data, weight_frac)
    q_fc_b = quantise_tensor(model.fc1.bias.data, weight_frac)

    # Export to .mem files
    os.makedirs(args.out_dir, exist_ok=True)
    save_mem(q_conv_w, os.path.join(args.out_dir, "conv1_weights.mem"))
    save_mem(q_conv_b, os.path.join(args.out_dir, "conv1_biases.mem"))
    save_mem(q_fc_w, os.path.join(args.out_dir, "fc1_weights.mem"))
    save_mem(q_fc_b, os.path.join(args.out_dir, "fc1_biases.mem"))

    print(f"Exported quantised .mem files to {args.out_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quantise trained CNN and export fixed-point weights to .mem")
    parser.add_argument('--model-dir', type=str, default='./output', help="Directory containing small_cnn.pth")
    parser.add_argument('--out-dir', type=str, default='./weights', help="Output directory for .mem files")
    parser.add_argument('--weight-frac', type=int, default=7, help="Fractional bits used for weights (e.g., 7 means Q X.7)")
    parser.add_argument('--act-frac', type=int, default=7, help="Fractional bits for activations (reserved for future use)")
    args = parser.parse_args()
    main(args)