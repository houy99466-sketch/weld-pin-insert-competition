# -*- coding: utf-8 -*-
"""测量主臂批量读的吞吐：比较逐条读 vs 一次性发 6 条读请求再统一解析。"""
import sys, time
sys.stdout.reconfigure(encoding='utf-8')
import serial

PORT, BAUD = 'COM22', 1_000_000
IDS = [1, 2, 3, 4, 5, 6]


def frame(sid, instr, params=()):
    body = [sid, len(params) + 2, instr] + list(params)
    return bytes([0xFF, 0xFF] + body + [(~(sum(body))) & 0xFF])


def parse(buf):
    out = {}
    i = 0
    while True:
        i = buf.find(b"\xff\xff", i)
        if i < 0 or len(buf) < i + 4:
            break
        ln = buf[i + 3]
        if len(buf) < i + 4 + ln:
            break
        sid = buf[i + 2]
        err = buf[i + 4]
        data = buf[i + 5:i + 4 + ln]
        if err == 0 and len(data) >= 2:
            out[sid] = data[0] | (data[1] << 8)
        i += 4 + ln
    return out


s = serial.Serial(PORT, BAUD, timeout=0.02)

# --- A: 逐条 ---
t0 = time.time(); rounds = 0; miss = 0
while time.time() - t0 < 1.5:
    got = 0
    for sid in IDS:
        s.reset_input_buffer()
        s.write(frame(sid, 0x02, [56, 2]))
        buf = b''
        dl = time.time() + 0.03
        while time.time() < dl:
            d = s.read(32)
            if d:
                buf += d
                if len(buf) >= 8:
                    break
        if parse(buf):
            got += 1
    rounds += 1
    miss += (6 - got)
A = rounds / (time.time() - t0)
print(f"A 逐条读: {A:.1f} Hz (缺 {miss})")

# --- B: 批量发 ---
t0 = time.time(); rounds = 0; miss = 0; last = {}
while time.time() - t0 < 1.5:
    s.reset_input_buffer()
    s.write(b''.join(frame(sid, 0x02, [56, 2]) for sid in IDS))
    buf = b''
    dl = time.time() + 0.03
    while time.time() < dl:
        d = s.read(256)
        if d:
            buf += d
            if len(parse(buf)) >= 6:
                break
    got = parse(buf)
    last = got
    rounds += 1
    miss += (6 - len(got))
B = rounds / (time.time() - t0)
print(f"B 批量发: {B:.1f} Hz (缺 {miss})  sample={last}")

# --- C: 批量 + 固定小等待 ---
t0 = time.time(); rounds = 0; miss = 0
while time.time() - t0 < 1.5:
    s.reset_input_buffer()
    s.write(b''.join(frame(sid, 0x02, [56, 2]) for sid in IDS))
    time.sleep(0.004)
    buf = s.read(256)
    got = parse(buf)
    rounds += 1
    miss += (6 - len(got))
C = rounds / (time.time() - t0)
print(f"C 批量+4ms: {C:.1f} Hz (缺 {miss})")
s.close()
