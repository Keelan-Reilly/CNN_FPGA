import argparse
import time

def send_image_file(image_mem_path, out_path, delay_s=0.001):
    """
    Simulate sending each line (byte) from a .mem image file over UART by
    writing it with a configurable inter-byte delay to a destination file.
    """
    # Read the fixed-point image bytes
    with open(image_mem_path) as f:
        lines = [line.strip() for line in f if line.strip()]

    # Write each byte to output with delay to emulate serial timing
    with open(out_path, "w") as out:
        for byte_hex in lines:
            out.write(byte_hex + "\n")
            out.flush()  # ensure written immediately
            time.sleep(delay_s)  # simulate transmit time per byte
    print(f"Simulated UART send complete; output written to {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simulate sending a .mem image over UART")
    parser.add_argument("--image-mem", type=str, required=True, help="Path to input .mem file containing image")
    parser.add_argument("--out", type=str, default="./uart_out.txt", help="Destination for simulated serial output")
    parser.add_argument("--delay", type=float, default=0.001, help="Delay between bytes in seconds (simulated baud)")
    args = parser.parse_args()
    send_image_file(args.image_mem, args.out, delay_s=args.delay)