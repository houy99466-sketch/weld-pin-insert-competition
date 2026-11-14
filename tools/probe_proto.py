# -*- coding: utf-8 -*-
"""只读协议扫描：对串口逐个长度/波特率发送候选查询帧，记录任何回码。
不发送使能/运动指令。用法: python probe_proto.py COM22
"""
import sys, time
sys.stdout.reconfigure(encoding='utf-8')
import serial

PORT = sys.argv[1] if len(sys.argv) > 1 else 'COM22'
BAUDS = [int(x) for x in sys.argv[2:]] or [115200, 921600, 1000000, 2000000, 3000000]

def scs_ping(sid=1):
    """Feetech/SCServo 协议 0 PING: FF FF ID LEN=02 INST=01 CHK"""
    body = [sid, 0x02, 0x01]
    chk = (~(sid + 0x02 + 0x01)) & 0xFF
    return bytes([0xFF, 0xFF] + body + [chk])

def scs_ping_v1(sid=1):
    """STS 协议 1 PING"""
    hdr = [sid, 0x02, 0x01]
    return bytes([0xFF, 0xFF] + hdr + [((~(sum(hdr))) & 0xFF)])

TESTS = [
    ('SLCAN-C', b'C\r'),
    ('SLCAN-V', b'V\r'),
    ('SLCAN-S6', b'S6\r'),
    ('SLCAN-O', b'O\r'),
    ('SCS-PING', scs_ping(1)),
    ('SCS-PING-id6', scs_ping(6)),
    ('MODBUS-03', bytes([0x01, 0x03, 0x00, 0x00, 0x00, 0x01, 0x84, 0x0A])),
    ('ASCII-VER', b'VER\r\n'),
    ('ASCII-QRY', b'#GETJPOS\r\n'),
    ('ASCII-ENQ', b'\r\n'),
    ('LF-LF', b'\n'),
    ('NULL', b'\x00'),
]

for baud in BAUDS:
    print('=' * 64)
    print(f'{PORT} @ {baud}')
    try:
        s = serial.Serial(PORT, baud, timeout=0.25)
    except Exception as e:
        print('  open failed:', e)
        continue
    try:
        for name, frame in TESTS:
            s.reset_input_buffer()
            s.write(frame)
            s.flush()
            t0 = time.time()
            buf = bytearray()
            while time.time() - t0 < 0.35:
                d = s.read(256)
                if d:
                    buf += d
            print('  %-14s tx=%-26s rx=%-3d %s' % (
                name, frame.hex(' ')[:26], len(buf), buf[:32].hex(' ') if buf else '-'))
    finally:
        s.close()
