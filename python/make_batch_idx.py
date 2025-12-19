#!/usr/bin/env python3
import argparse
import os
import struct


def read_idx_images(path: str):
    with open(path, "rb") as f:
        magic, n, rows, cols = struct.unpack(">IIII", f.read(16))
        assert magic == 2051, f"Bad magic for images: {magic}"
        data = f.read()
    assert rows == 28 and cols == 28, f"Expected 28x28, got {rows}x{cols}"
    assert len(data) == n * rows * cols, (
        f"Size mismatch: {len(data)} vs {n * rows * cols}"
    )
    return n, data  # data is already uint8 bytes, row-major per image


def read_idx_labels(path: str):
    with open(path, "rb") as f:
        magic, n = struct.unpack(">II", f.read(8))
        assert magic == 2049, f"Bad magic for labels: {magic}"
        data = f.read()
    assert len(data) == n, f"Size mismatch: {len(data)} vs {n}"
    return n, data  # uint8 bytes, one per label


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images-idx", required=True, help="Path to *-images-idx3-ubyte")
    ap.add_argument("--labels-idx", required=True, help="Path to *-labels-idx1-ubyte")
    ap.add_argument("--out-dir", default="weights", help="Output directory")
    ap.add_argument("--n", type=int, default=1000, help="How many samples to export")
    args = ap.parse_args()

    n_img, img_bytes = read_idx_images(args.images_idx)
    n_lab, lab_bytes = read_idx_labels(args.labels_idx)
    assert n_img == n_lab, f"Count mismatch images={n_img} labels={n_lab}"

    N = min(args.n, n_img)
    os.makedirs(args.out_dir, exist_ok=True)

    images_bin_path = os.path.join(args.out_dir, "images.bin")
    labels_bin_path = os.path.join(args.out_dir, "labels.bin")

    # Each image is 28*28 bytes
    one = 28 * 28
    out_imgs = img_bytes[: N * one]
    out_labs = lab_bytes[:N]

    with open(images_bin_path, "wb") as f:
        f.write(out_imgs)

    with open(labels_bin_path, "wb") as f:
        f.write(out_labs)

    # Optional: single-image file for quick sanity runs
    with open(os.path.join(args.out_dir, "input_image.bin"), "wb") as f:
        f.write(out_imgs[:one])

    print(f"Wrote {images_bin_path} ({N} images, {len(out_imgs)} bytes)")
    print(f"Wrote {labels_bin_path} ({N} labels, {len(out_labs)} bytes)")
    print(f"Wrote {os.path.join(args.out_dir, 'input_image.bin')} (first image)")


if __name__ == "__main__":
    main()
