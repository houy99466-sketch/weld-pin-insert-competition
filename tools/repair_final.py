# -*- coding: utf-8 -*-
"""
server.py 一次性复原（合并三步）：
  1. 读损坏文件（PowerShell 曾把 UTF-8 当 GBK 读、再以 UTF-8 写回）
  2. 用 __pycache__ 里损坏前的编译产物当字典，把字符串/文档串按原文逐字替换
  3. 剩余乱码按行反演回中文（注释不在 pyc 里，只能这样救）
  4. 统一 LF/CRLF，不用文本模式写（否则 \r\r\n）
"""
import ctypes
import marshal
import sys
from pathlib import Path

k32 = ctypes.WinDLL("kernel32", use_last_error=True)


def mb2wc(b: bytes, cp: int = 936) -> str:
    n = k32.MultiByteToWideChar(cp, 0, b, len(b), None, 0)
    buf = ctypes.create_unicode_buffer(n)
    k32.MultiByteToWideChar(cp, 0, b, len(b), buf, n)
    return buf[:n]


def wc2mb(ch: str, cp: int = 936) -> bytes | None:
    buf = ctypes.create_string_buffer(8)
    n = k32.WideCharToMultiByte(cp, 0, ch, 1, buf, len(buf), None, None)
    return buf.raw[:n] if n > 0 else None


def mangle(s: str) -> str:
    return mb2wc(s.encode("utf-8")).replace("\ufffd", "?")


def invert_run(run: str) -> str | None:
    b = bytearray()
    for ch in run:
        cb = wc2mb(ch)
        if cb is None:
            return None
        b += cb
    try:
        return bytes(b).decode("utf-8")
    except UnicodeDecodeError:
        return None


def invert_line(line: str) -> str:
    out, i, n = [], 0, len(line)
    while i < n:
        ch = line[i]
        if ord(ch) < 0x80:
            out.append(ch)
            i += 1
            continue
        j = i
        while j < n and ord(line[j]) >= 0x80:
            j += 1
        run = line[i:j]
        inv = invert_run(run)
        out.append(inv if inv is not None else run)
        i = j
    return "".join(out)


src, pyc, dst = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
raw = src.read_bytes()
if raw[:3] == b"\xef\xbb\xbf":
    raw = raw[3:]
text = raw.decode("utf-8").replace("\r\n", "\n")

code = marshal.loads(pyc.read_bytes()[16:])
strs = []


def walk(c):
    for k in c.co_consts:
        if isinstance(k, str):
            if any(ord(ch) > 127 for ch in k):
                strs.append(k)
        elif isinstance(k, dict):
            for v in k.values():
                if isinstance(v, str) and any(ord(ch) > 127 for ch in v):
                    strs.append(v)
        elif hasattr(k, "co_consts"):
            walk(k)


walk(code)
strs = sorted(set(strs), key=len, reverse=True)

fixed = 0
for s in strs:
    m = mangle(s)
    if not m or m == s:
        continue
    if m in text:
        fixed += text.count(m)
        text = text.replace(m, s)

text = "\n".join(invert_line(l) for l in text.split("\n"))
text = text.replace("\r", "")
out = text.replace("\n", "\r\n")
dst.write_bytes(out.encode("utf-8"))

left = [(i + 1, l) for i, l in enumerate(text.split("\n")) if "?" in l]
print(f"字典替换 {fixed} 处 -> 写出 {dst}（{len(text.splitlines())} 行）")
print(f"仍含 '?' 的行 {len(left)}")
for i, l in left:
    print(f"  {i}: {l.strip()[:120]}")
