# python/uart_ping.py
import serial, time

PORT = "COM5"
BAUD = 115200

with serial.Serial(PORT, BAUD, timeout=0.2) as ser:
    ser.reset_input_buffer()
    ser.reset_output_buffer()
    time.sleep(0.05)

    ser.write(b"\x55")
    ser.flush()

    # give FPGA time even if it’s slow/booting
    time.sleep(0.05)

    data = ser.read(1)
    print("RX:", data)
