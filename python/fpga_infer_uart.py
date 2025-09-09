# fpga_infer_uart.py
import argparse
import time
import serial
import pathlib


def load_mem(path):
    lines = [line.strip() for line in open(path) if line.strip()]
    if len(lines) != 784:
        raise ValueError(f"{path}: expected 784 lines, got {len(lines)}")
    return bytes(int(h, 16) for h in lines)


def load_bin(path):
    b = pathlib.Path(path).read_bytes()
    if len(b) != 784:
        raise ValueError(f"{path}: expected 784 bytes, got {len(b)}")
    return b


def load_image(path):
    ext = pathlib.Path(path).suffix.lower()
    return load_bin(path) if ext == ".bin" else load_mem(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default="COM3", help="COM port (e.g. COM7)")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument(
        "--image",
        default="weights/input_image.mem",
        help=".mem (784 hex lines) or .bin (784 bytes)",
    )
    ap.add_argument(
        "--reset-wait",
        type=float,
        default=0.5,
        help="Seconds to wait after pressing board reset (manual)",
    )
    ap.add_argument("--timeout", type=float, default=3.0, help="UART read timeout (s)")
    args = ap.parse_args()

    img_bytes = load_image(args.image)

    # Open serial
    with serial.Serial(
        args.port, args.baud, bytesize=8, parity="N", stopbits=1, timeout=args.timeout
    ) as ser:
        print(f"Opened {ser.name} @ {args.baud} 8N1")

        # Recommended: press the board's CPU RESET/PROG now to clear the pipeline
        print("If needed, press board RESET…")
        time.sleep(args.reset_wait)

        # Send all 784 bytes
        n = ser.write(img_bytes)
        ser.flush()  # push out TX FIFO
        if n != 784:
            raise RuntimeError(f"Wrote {n} bytes (expected 784)")

        # Read single ASCII digit
        rx = ser.read(1)
        if len(rx) != 1:
            raise TimeoutError("Timed out waiting for prediction byte")
        pred_chr = rx.decode(errors="ignore")
        if not pred_chr.isdigit():
            print(f"Got non-digit byte: 0x{rx[0]:02X}")
        pred = ord(pred_chr) - ord("0")
        print(f"FPGA predicted: {pred} (ASCII '{pred_chr}')")


if __name__ == "__main__":
    main()
