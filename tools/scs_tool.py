# -*- coding: utf-8 -*-
"""
Feetech SCS/STS 总线舵机 只读枚举工具（新机械臂识别用）。

协议: FF FF <ID> <LEN> <INSTR> <PARAM...> <CHK>
      CHK = ~(ID+LEN+INSTR+PARAM) & 0xFF
      PING=0x01, READ=0x02, WRITE=0x03
      应答: FF FF <ID> <LEN> <ERR> <DATA...> <CHK>

用法:
  python scs_tool.py COM22 1000000 scan          # 扫描 1..24 号舵机
  python scs_tool.py COM22 1000000 read 1 56 2   # 读寄存器
  python scs_tool.py COM22 1000000 monitor 1-6   # 连续读位置
"""
import sys, time
sys.stdout.reconfigure(encoding='utf-8')
import serial

PORT = sys.argv[1] if len(sys.argv) > 1 else 'COM22'
BAUD = int(sys.argv[2]) if len(sys.argv) > 2 else 1000000
CMD = sys.argv[3] if len(sys.argv) > 3 else 'scan'

REG = {
    'model': (3, 2), 'id': (5, 1), 'baud': (6, 1),
    'torque_en': (40, 1), 'goal_pos': (42, 2),
    'pos': (56, 2), 'speed': (58, 2), 'load': (60, 2),
    'volt': (62, 1), 'temp': (63, 1), 'moving': (66, 1), 'current': (69, 2),
}

MODEL_NAME = {
    777: 'SCS/STS series (?)',
    2825: 'SCS225',
    1: 'SCS0009', 2: 'SCS0009',
    500: 'SCS15', 777: 'SCS17/SCS series',
    3000: 'STS3215', 3001: 'STS3215',
    0x0C: '?', 
}


class Bus:
    def __init__(self, port, baud):
        self.s = serial.Serial(port, baud, timeout=0.08)

    def close(self):
        self.s.close()

    @staticmethod
    def frame(sid, instr, params=()):
        body = [sid, len(params) + 2, instr] + list(params)
        return bytes([0xFF, 0xFF] + body + [(~(sum(body))) & 0xFF])

    def talk(self, sid, instr, params=(), expect=None, tries=2):
        f = self.frame(sid, instr, params)
        for _ in range(tries):
            self.s.reset_input_buffer()
            self.s.write(f)
            self.s.flush()
            buf = bytearray()
            t0 = time.time()
            while time.time() - t0 < 0.12:
                d = self.s.read(64)
                if d:
                    buf += d
                    if len(buf) >= 6:
                        # 找一帧完整应答
                        i = buf.find(b'\xff\xff')
                        if i >= 0 and len(buf) >= i + 4:
                            ln = buf[i + 3]
                            if len(buf) >= i + 4 + ln:
                                return bytes(buf[i:i + 4 + ln])
        return None

    def ping(self, sid, tries=2):
        return self.talk(sid, 0x01, (), tries=tries)

    def read(self, sid, addr, length, tries=2):
        r = self.talk(sid, 0x02, [addr, length], tries=tries)
        if not r or len(r) < 6:
            return None
        err = r[4]
        data = r[5:-1]
        if err != 0 or len(data) < length:
            return None
        return data

    def read_int(self, sid, addr, length):
        d = self.read(sid, addr, length)
        if d is None:
            return None
        if length == 1:
            return d[0]
        return d[0] | (d[1] << 8)


def scan(bus, lo=1, hi=24):
    found = []
    for sid in range(lo, hi + 1):
        if bus.ping(sid):
            found.append(sid)
    print('responding ids:', found)
    for sid in found:
        model = bus.read_int(sid, *REG['model'])
        pos = bus.read_int(sid, *REG['pos'])
        spd = bus.read_int(sid, *REG['speed'])
        load = bus.read_int(sid, *REG['load'])
        volt = bus.read_int(sid, *REG['volt'])
        temp = bus.read_int(sid, *REG['temp'])
        deg = None if pos is None else pos * 360.0 / 4096.0
        spd_rpm = None if spd is None else (spd & 0x7FFF) * 0.0146 * (-1 if spd & 0x8000 else 1)
        ld = None if load is None else (load & 0x3FF) * 0.1 * (-1 if load & 0x400 else 1)
        print(f'  id={sid:<3} model={model!s:<6} pos={pos!s:<6} '
              f'{"" if deg is None else f"{deg:7.2f}°"}  speed={spd_rpm!s:<8} load={ld!s:<7} '
              f'volt={volt!s:<4} temp={temp}')
    return found


def main():
    bus = Bus(PORT, BAUD)
    try:
        if CMD == 'scan':
            scan(bus)
        elif CMD == 'read':
            sid, addr, length = int(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6])
            print(bus.read(sid, addr, length))
        elif CMD == 'monitor':
            rng = sys.argv[4]
            a, b = (rng.split('-') + [rng])[:2]
            ids = list(range(int(a), int(b) + 1))
            t0 = time.time()
            while time.time() - t0 < float(sys.argv[5]) if len(sys.argv) > 5 else 5.0:
                row = []
                for sid in ids:
                    p = bus.read_int(sid, *REG['pos'])
                    row.append('-' if p is None else f'{p*360/4096:7.2f}')
                print('  ' + ' '.join(row))
                time.sleep(0.1)
    finally:
        bus.close()


if __name__ == '__main__':
    main()
