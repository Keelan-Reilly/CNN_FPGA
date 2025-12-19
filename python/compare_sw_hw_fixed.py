# compare_sw_hw_fixed.py
#
# Bit-accurate(ish) software golden model that matches the RTL assumptions:
#   - Input byte -> Q0.FRAC_BITS via the same LUT formula used in top_level
#   - conv2d: SAME padding, int MAC, bias pre-shifted by FRAC_BITS, then >>> FRAC_BITS + sat
#   - ReLU: clamp negatives to 0 (in-place)
#   - maxpool: 2x2 max, stride 2
#   - dense: int MAC, bias pre-shifted by FRAC_BITS, then >>> (FRAC_BITS+POST_SHIFT) + sat
#   - argmax on signed logits
#
# It also optionally talks to the FPGA over UART and compares HW vs golden.
#
# Usage examples:
#   python python/compare_sw_hw_fixed.py --image weights/input_image.bin --weights-dir weights --no-hw
#   python python/compare_sw_hw_fixed.py --image weights/input_image.bin --weights-dir weights --port COM5 --trials 5
#   python python/compare_sw_hw_fixed.py --debug-hw-logits ...   (expects HW to emit 'L'+20 bytes then digit)

import argparse
import pathlib
import time
from typing import List, Tuple, Optional

try:
    import serial  # pyserial
except ImportError:
    serial = None


# --------------------------- helpers ---------------------------


def read_image_bytes(path: str) -> bytes:
    p = pathlib.Path(path)
    if p.suffix.lower() == ".bin":
        b = p.read_bytes()
        if len(b) != 784:
            raise ValueError(f"{path}: expected 784 bytes, got {len(b)}")
        return b
    elif p.suffix.lower() == ".mem":
        lines = [ln.strip() for ln in p.read_text().splitlines() if ln.strip()]
        if len(lines) != 784:
            raise ValueError(f"{path}: expected 784 lines, got {len(lines)}")
        return bytes(int(h, 16) & 0xFF for h in lines)
    else:
        raise ValueError(f"{path}: expected .bin or .mem")


def sign_extend(value: int, bits: int) -> int:
    mask = (1 << bits) - 1
    value &= mask
    sign = 1 << (bits - 1)
    return (value ^ sign) - sign


def read_mem_signed(path: str, bits: int) -> List[int]:
    lines = [ln.strip() for ln in open(path, "r") if ln.strip()]
    out = []
    for ln in lines:
        v = int(ln, 16)
        out.append(sign_extend(v, bits))
    return out


def sat_signed(x: int, bits: int) -> int:
    mx = (1 << (bits - 1)) - 1
    mn = -(1 << (bits - 1))
    if x > mx:
        return mx
    if x < mn:
        return mn
    return x


# --------------------------- golden model ---------------------------


def bytes_to_q(img_bytes: bytes, frac_bits: int, data_width: int) -> List[int]:
    # Matches RTL LUT:
    #   v = (k * (1<<FRAC_BITS) + 127) / 255
    # stored as signed DATA_WIDTH (but values are 0..2^FRAC_BITS)
    scale = 1 << frac_bits
    out = []
    for k in img_bytes:
        v = (k * scale + 127) // 255
        out.append(sat_signed(v, data_width))
    return out  # length 784


def conv2d_same_int(
    ifmap: List[int],  # length Cin*H*W
    W: List[int],  # flattened oc*(Cin*K*K)+ic*(K*K)+kr*K+kc
    B: List[int],  # length Cout
    H: int,
    Wd: int,
    Cin: int,
    Cout: int,
    K: int,
    frac_bits: int,
    data_width: int,
) -> List[int]:
    pad = (K - 1) // 2
    ofmap = [0] * (Cout * H * Wd)

    k2 = K * K
    oc_stride = Cin * k2

    for oc in range(Cout):
        b = B[oc]
        for r in range(H):
            for c in range(Wd):
                acc = int(b) << frac_bits  # bias_ext
                base_w = oc * oc_stride
                for ic in range(Cin):
                    base_ic = base_w + ic * k2
                    for kr in range(K):
                        ir = r + kr - pad
                        for kc in range(K):
                            icc = c + kc - pad
                            wv = W[base_ic + kr * K + kc]
                            if 0 <= ir < H and 0 <= icc < Wd:
                                pix = ifmap[(ic * H + ir) * Wd + icc]
                            else:
                                pix = 0
                            acc += int(pix) * int(wv)

                shifted = acc >> frac_bits  # arithmetic in Python for negative ints
                outv = sat_signed(shifted, data_width)
                ofmap[(oc * H + r) * Wd + c] = outv

    return ofmap


def relu_inplace(x: List[int]) -> None:
    for i, v in enumerate(x):
        if v < 0:
            x[i] = 0


def maxpool2x2_stride2(
    ifmap: List[int],  # length C*H*W
    C: int,
    H: int,
    Wd: int,
) -> List[int]:
    assert H % 2 == 0 and Wd % 2 == 0
    OH, OW = H // 2, Wd // 2
    out = [0] * (C * OH * OW)

    for ch in range(C):
        for r in range(OH):
            for c in range(OW):
                r0 = 2 * r
                c0 = 2 * c
                a0 = ifmap[(ch * H + r0) * Wd + (c0)]
                a1 = ifmap[(ch * H + r0) * Wd + (c0 + 1)]
                a2 = ifmap[(ch * H + (r0 + 1)) * Wd + (c0)]
                a3 = ifmap[(ch * H + (r0 + 1)) * Wd + (c0 + 1)]
                out[(ch * OH + r) * OW + c] = max(a0, a1, a2, a3)

    return out


def dense_int(
    x: List[int],  # length IN_DIM (int16-ish)
    W: List[int],  # flattened o*IN_DIM + i
    B: List[int],  # length OUT_DIM
    IN_DIM: int,
    OUT_DIM: int,
    frac_bits: int,
    data_width: int,
    post_shift: int = 0,
) -> List[int]:
    out = [0] * OUT_DIM
    sh = frac_bits + post_shift

    for o in range(OUT_DIM):
        acc = int(B[o]) << frac_bits
        row_base = o * IN_DIM
        for i in range(IN_DIM):
            acc += int(x[i]) * int(W[row_base + i])

        shifted = acc >> sh
        out[o] = sat_signed(shifted, data_width)

    return out


def argmax_signed(vec: List[int]) -> int:
    besti = 0
    bestv = vec[0]
    for i in range(1, len(vec)):
        if vec[i] > bestv:
            bestv = vec[i]
            besti = i
    return besti


def sw_golden_predict(
    img_bytes: bytes,
    weights_dir: str,
    data_width: int = 16,
    frac_bits: int = 7,
    img_size: int = 28,
    in_channels: int = 1,
    out_channels: int = 8,
    kernel: int = 3,
    num_classes: int = 10,
    post_shift: int = 0,
) -> Tuple[int, List[int]]:
    wdir = pathlib.Path(weights_dir)

    conv_w = read_mem_signed(str(wdir / "conv1_weights.mem"), data_width)
    conv_b = read_mem_signed(str(wdir / "conv1_biases.mem"), data_width)
    fc_w = read_mem_signed(str(wdir / "fc1_weights.mem"), data_width)
    fc_b = read_mem_signed(str(wdir / "fc1_biases.mem"), data_width)

    # basic sanity checks to catch “wrong file” early
    exp_conv_w = out_channels * in_channels * kernel * kernel
    exp_conv_b = out_channels
    exp_fc_w = num_classes * (out_channels * (img_size // 2) * (img_size // 2))
    exp_fc_b = num_classes

    if len(conv_w) != exp_conv_w:
        raise ValueError(
            f"conv1_weights.mem: expected {exp_conv_w} vals, got {len(conv_w)}"
        )
    if len(conv_b) != exp_conv_b:
        raise ValueError(
            f"conv1_biases.mem: expected {exp_conv_b} vals, got {len(conv_b)}"
        )
    if len(fc_w) != exp_fc_w:
        raise ValueError(f"fc1_weights.mem: expected {exp_fc_w} vals, got {len(fc_w)}")
    if len(fc_b) != exp_fc_b:
        raise ValueError(f"fc1_biases.mem: expected {exp_fc_b} vals, got {len(fc_b)}")

    # 1) input bytes -> fixed-point activations (matches RTL LUT)
    ifmap_q = bytes_to_q(img_bytes, frac_bits=frac_bits, data_width=data_width)

    # 2) conv
    conv = conv2d_same_int(
        ifmap=ifmap_q,
        W=conv_w,
        B=conv_b,
        H=img_size,
        Wd=img_size,
        Cin=in_channels,
        Cout=out_channels,
        K=kernel,
        frac_bits=frac_bits,
        data_width=data_width,
    )

    # 3) relu
    relu_inplace(conv)

    # 4) maxpool
    pool = maxpool2x2_stride2(conv, C=out_channels, H=img_size, Wd=img_size)

    # 5) dense -> logits
    in_dim = out_channels * (img_size // 2) * (img_size // 2)
    logits = dense_int(
        x=pool,
        W=fc_w,
        B=fc_b,
        IN_DIM=in_dim,
        OUT_DIM=num_classes,
        frac_bits=frac_bits,
        data_width=data_width,
        post_shift=post_shift,
    )

    # 6) argmax
    pred = argmax_signed(logits)
    return pred, logits


# --------------------------- UART HW compare (optional) ---------------------------


def _read_exact(ser, n, timeout):
    end = time.time() + timeout
    out = bytearray()
    while len(out) < n:
        chunk = ser.read(n - len(out))
        if chunk:
            out += chunk
        elif time.time() > end:
            raise TimeoutError(f"Timed out reading {n} bytes (got {len(out)})")
    return bytes(out)


def _read_until(ser, target_byte, timeout):
    end = time.time() + timeout
    while True:
        b = ser.read(1)
        if b == target_byte:
            return
        if not b and time.time() > end:
            raise TimeoutError(f"Timed out waiting for {target_byte!r}")


def hw_predict_uart(
    port: str,
    baud: int,
    timeout: float,
    img_bytes: bytes,
    reset_wait: float,
    debug_logits: bool = False,
) -> Tuple[Optional[int], Optional[List[int]]]:
    if serial is None:
        raise RuntimeError("pyserial not installed. pip install pyserial")

    with serial.Serial(
        port, baud, bytesize=8, parity="N", stopbits=1, timeout=timeout
    ) as ser:
        ser.reset_input_buffer()
        ser.reset_output_buffer()

        time.sleep(reset_wait)

        n = ser.write(img_bytes)
        ser.flush()
        if n != 784:
            raise RuntimeError(f"Wrote {n} bytes (expected 784)")

        hw_logits = None
        if debug_logits:
            # Expect 'L' then 10 int16 big-endian
            _read_until(ser, b"L", timeout)
            raw = _read_exact(ser, 20, timeout)
            hw_logits = [
                int.from_bytes(raw[i : i + 2], "big", signed=True)
                for i in range(0, 20, 2)
            ]

        # Read first ASCII digit, skipping junk
        end = time.time() + timeout
        while True:
            b = ser.read(1)
            if b and b.isdigit():
                return (b[0] - ord("0"), hw_logits)
            if not b and time.time() > end:
                raise TimeoutError("Timed out waiting for prediction byte")


# --------------------------- main ---------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default="weights/input_image.bin")
    ap.add_argument("--weights-dir", default="weights")
    ap.add_argument("--data-width", type=int, default=16)
    ap.add_argument("--frac-bits", type=int, default=7)
    ap.add_argument("--post-shift", type=int, default=0)

    ap.add_argument(
        "--no-hw", action="store_true", help="Only compute golden, don't talk to FPGA"
    )
    ap.add_argument("--port", default="")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--timeout", type=float, default=3.0)
    ap.add_argument("--reset-wait", type=float, default=0.6)
    ap.add_argument("--trials", type=int, default=5)
    ap.add_argument(
        "--debug-hw-logits",
        action="store_true",
        help="Expect 'L'+20 logit bytes before the digit",
    )
    ap.add_argument("--print-golden-logits", action="store_true")
    args = ap.parse_args()

    img_bytes = read_image_bytes(args.image)

    sw_pred, sw_logits = sw_golden_predict(
        img_bytes=img_bytes,
        weights_dir=args.weights_dir,
        data_width=args.data_width,
        frac_bits=args.frac_bits,
        post_shift=args.post_shift,
    )

    print(f"[GOLDEN fixed] predicted {sw_pred}")
    if args.print_golden_logits:
        print("[GOLDEN logits]", sw_logits)

    if args.no_hw:
        return

    if not args.port:
        raise ValueError("Provide --port (or use --no-hw).")

    hw_preds = []
    for t in range(args.trials):
        try:
            hw_pred, hw_logits = hw_predict_uart(
                port=args.port,
                baud=args.baud,
                timeout=args.timeout,
                img_bytes=img_bytes,
                reset_wait=args.reset_wait,
                debug_logits=args.debug_hw_logits,
            )
            hw_preds.append(hw_pred)
            ok = hw_pred == sw_pred
            print(f"[HW] trial {t + 1}: {hw_pred}  {'OK' if ok else 'MISMATCH'}")

            if hw_logits is not None:
                print("[HW logits]", hw_logits)
                # Optional: compare logits elementwise if your HW debug logits are the same scaling
                if len(hw_logits) == len(sw_logits):
                    diffs = [hw_logits[i] - sw_logits[i] for i in range(len(sw_logits))]
                    max_abs = max(abs(d) for d in diffs)
                    print(f"[logit diff] max |HW-SW| = {max_abs}")

        except Exception as e:
            hw_preds.append(None)
            print(f"[HW] trial {t + 1} ERROR: {e}")

    agree = sum(1 for p in hw_preds if p == sw_pred)
    print(f"\nAgreement vs golden: {agree}/{len(hw_preds)} (golden={sw_pred})")
    if agree != len(hw_preds):
        print(
            "⚠️  Mismatch/instability: now it's almost certainly RTL timing/handshake or BRAM semantics, not the software model."
        )


if __name__ == "__main__":
    main()
