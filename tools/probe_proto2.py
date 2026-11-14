# -*- coding: utf-8 -*-
"""更宽的只读探测：波特率 × 帧格式 × 候选查询帧。只发查询，不发运动指令。
用法: python probe_proto2.py COM22 [bauds...]
"""
import sys, time
sys.stdout.reconfigure(encoding='utf-8')
import serial

PORT = sys.argv[1] if len(sys.argv) > 1 else 'COM22'
BAUDS = [int(x) for x in sys.argv[2:]] or [
    9600, 19200, 38400, 57600, 115200, 230400, 460800, 500000,
    921600, 1000000, 1500000, 2000000, 2500000, 3000000, 4000000]
FRAMINGS = [(8, 'N', 1), (8, 'E', 1), (8, 'N', 2)]


def scs_ping(sid):
    body = [sid, 0x02, 0x01]
    return bytes([0xFF, 0xFF] + body + [(~(sum(body))) & 0xFF])


TESTS = [
    ('SCS-ping1', scs_ping(0x01)),
    ('SCS-bcast', scs_ping(0xFE)),
    ('SLCAN-V', b'V\r'),
    ('SLCAN-C', b'C\r'),
    ('ASCII-AT', b'AT\r\n'),
    ('ASCII-VER', b'VER\r\n'),
    ('ASCII-ENQ', b'\r\n'),
    ('MODBUS-9600q', bytes([0x01, 0x03, 0x00, 0x00, 0x00, 0x01, 0x84, 0x0A])),
    ('DJI-AA55', bytes([0xAA, 0x55, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00])),
    ('PING-55AA', bytes([0x55, 0xAA, 0x01, 0x01, 0x00, 0x01])),
    ('BUS-7B', bytes([0x7B, 0x7B, 0x01, 0x00])),
]

hits = 0
for baud in BAUDS:
    for (bits, par, stop) in FRAMINGS:
        try:
            s = serial.Serial(PORT, baud, bytesize=bits, parity=par,
                              stopbits=stop, timeout=0.2)
        except Exception as e:
            print(f'{baud} {bits}{par}{stop}: open failed {e}')
            continue
        got_any = False
        for name, frame in TESTS:
            s.reset_input_buffer()
            try:
                s.write(frame); s.flush()
            except Exception as e:
                print(f'   write err {e}'); break
            t0 = time.time(); buf = bytearray()
            while time.time() - t0 < 0.3:
                d = s.read(128)
                if d:
                    buf += d
            if buf:
                got_any = True; hits += 1
                print(f'{baud:>8} {bits}{par}{stop}  {name:<12} rx={buf[:24].hex(" ")}')
        s.close()
        if got_any:
            print(f'{baud:>8} {bits}{par}{stop}  <<< 有响应 >>>')
print('total replies:', hits)
