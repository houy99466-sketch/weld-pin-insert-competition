# -*- coding: utf-8 -*-
"""Passive sniff a serial port at several baud rates (read-only, sends nothing)."""
import sys, time
sys.stdout.reconfigure(encoding='utf-8')
import serial
from serial.tools import list_ports

PORT = sys.argv[1] if len(sys.argv) > 1 else 'COM22'
BAUDS = [921600, 1000000, 2000000, 3000000, 460800, 115200]
b = int(sys.argv[2]) if len(sys.argv) > 2 else None
if b:
    BAUDS = [b]
SEC = float(sys.argv[3]) if len(sys.argv) > 3 else 1.2

print('ports:')
for p in list_ports.comports():
    print('  %-8s %-42s %s vid=%s pid=%s sn=%s' % (
        p.device, (p.description or '')[:42], (p.manufacturer or '')[:18],
        hex(p.vid) if p.vid else '-', hex(p.pid) if p.pid else '-', p.serial_number))
print()

for baud in BAUDS:
    try:
        s = serial.Serial(PORT, baud, timeout=0.05)
    except Exception as e:
        print('%s @%d  OPEN FAIL: %s' % (PORT, baud, e))
        continue
    buf = bytearray()
    t0 = time.time()
    while time.time() - t0 < SEC:
        try:
            d = s.read(4096)
        except Exception as e:
            print('  read err', e); break
        if d:
            buf += d
    s.close()
    print('%s @%-8d bytes=%-6d %s' % (
        PORT, baud, len(buf), buf[:48].hex(' ') if buf else '(no data)'))
    if len(buf) > 48:
        print('        head64: %s' % buf[:64].hex(' '))
