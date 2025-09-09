# compare_sw_hw.py
import argparse, time, pathlib
import serial
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np, os

# --- Model must match training/export ---
class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(8 * 14 * 14, 10)
    def forward(self, x):
        x = F.relu(self.conv1(x)); x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.fc1(x)

def load_mem(path):
    lines = [l.strip() for l in open(path) if l.strip()]
    if len(lines) != 784:
        raise ValueError(f"{path}: expected 784 lines, got {len(lines)}")
    return bytes(int(h, 16) for h in lines)

def load_bin(path):
    b = pathlib.Path(path).read_bytes()
    if len(b) != 784:
        raise ValueError(f"{path}: expected 784 bytes, got {len(b)}")
    return b

def load_image_bytes(path):
    return load_bin(path) if pathlib.Path(path).suffix.lower() == ".bin" else load_mem(path)

def bytes_to_tensor_0_1(img_bytes):
    arr = np.frombuffer(img_bytes, dtype=np.uint8).reshape(1,1,28,28)
    return torch.tensor(arr, dtype=torch.float32) / 255.0

def sw_predict(model_dir, img_bytes):
    model = SmallCNN()
    state = torch.load(os.path.join(model_dir, "small_cnn.pth"), map_location="cpu")
    model.load_state_dict(state); model.eval()
    x = bytes_to_tensor_0_1(img_bytes)
    with torch.no_grad():
        logits = model(x)
        pred = logits.argmax(dim=1).item()
        conf = torch.softmax(logits, dim=1)[0, pred].item()
    return pred, conf

def hw_predict(port, baud, timeout, img_bytes, reset_wait):
    with serial.Serial(port, baud, bytesize=8, parity="N", stopbits=1, timeout=timeout) as ser:
        print(f"Opened {ser.name} @ {baud} 8N1")
        print("If needed, press board RESET…")
        time.sleep(reset_wait)
        n = ser.write(img_bytes); ser.flush()
        if n != 784: raise RuntimeError(f"Wrote {n} bytes (expected 784)")
        rx = ser.read(1)
        if len(rx) != 1: raise TimeoutError("Timed out waiting for prediction byte")
        ch = rx.decode(errors="ignore")
        if not ch.isdigit():
            print(f"Got non-digit: 0x{rx[0]:02X}")
            return None
        return ord(ch) - ord("0")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default="weights/input_image.mem")
    ap.add_argument("--model-dir", default="./output")
    ap.add_argument("--port", default="COM4")      # set your default here
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--timeout", type=float, default=3.0)
    ap.add_argument("--reset-wait", type=float, default=0.6)
    ap.add_argument("--trials", type=int, default=5, help="repeat HW to check stability")
    args = ap.parse_args()

    img_bytes = load_image_bytes(args.image)

    sw_pred, sw_conf = sw_predict(args.model_dir, img_bytes)
    print(f"[SW] predicted {sw_pred}  (conf {sw_conf:.3f})")

    hw_preds = []
    for t in range(args.trials):
        try:
            hw = hw_predict(args.port, args.baud, args.timeout, img_bytes, args.reset_wait)
            print(f"[HW] trial {t+1}: {hw}")
            hw_preds.append(hw)
        except Exception as e:
            print(f"[HW] trial {t+1} ERROR: {e}")
            hw_preds.append(None)

    agree = sum(1 for p in hw_preds if p == sw_pred)
    print(f"\nAgreement: {agree}/{len(hw_preds)} trials equal to SW ({sw_pred}).")
    if agree != len(hw_preds):
        print("⚠️  Mismatch/instability detected — likely timing-related on FPGA.")

if __name__ == "__main__":
    main()
