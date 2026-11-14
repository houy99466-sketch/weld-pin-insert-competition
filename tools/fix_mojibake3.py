# -*- coding: utf-8 -*-
"""
按行把残余乱码反演回中文：把非 ASCII 串按 cp936 编码回字节，再按 UTF-8 解码。
'?' 当作缺口保留（那些是被 PowerShell 吃掉的字节），留给人工/字典补。
"""
import ctypes
import sys
from pathlib import Path

k32 = ctypes.WinDLL("kernel32", use_last_error=True)


def char_to_cp936(ch: str) -> bytes | None:
    buf = ctypes.create_string_buffer(8)
    n = k32.WideCharToMultiByte(936, 0, ch, 1, buf, len(buf), None, None)
    if n <= 0:
        return None
    return buf.raw[:n]


def is_ascii(ch):
    return ord(ch) < 0x80


def invert_run(run: str) -> str | None:
    """一段连续的非 ASCII 文本 -> 原文（失败返回 None）"""
    b = bytearray()
    for ch in run:
        cb = char_to_cp936(ch)
        if cb is None:
            return None
        b += cb
    try:
        return bytes(b).decode("utf-8")
    except UnicodeDecodeError:
        return None


def invert_line(line: str) -> str:
    out = []
    i = 0
    n = len(line)
    while i < n:
        ch = line[i]
        if is_ascii(ch):
            out.append(ch)
            i += 1
            continue
        j = i
        while j < n and not is_ascii(line[j]):
            j += 1
        run = line[i:j]
        inv = invert_run(run)
        out.append(inv if inv is not None else run)
        i = j
    return "".join(out)


src = Path(sys.argv[1])
dst = Path(sys.argv[2]) if len(sys.argv) > 2 else src.with_suffix(".zh.py")
text = src.read_text(encoding="utf-8")
lines = text.split("\n")
res = [invert_line(l) for l in lines]
dst.write_text("\n".join(res), encoding="utf-8")
left = [(i + 1, l) for i, l in enumerate(res) if "?" in l]
print(f"写出 {dst}；仍含 '?' 的行 {len(left)}")
for i, l in left:
    print(f"  {i}: {l.strip()[:120]}")
