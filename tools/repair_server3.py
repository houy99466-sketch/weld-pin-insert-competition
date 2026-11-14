# -*- coding: utf-8 -*-
"""
最终版 server.py 乱码修复：
用 Windows 的 MultiByteToWideChar(936) 精确复现「PowerShell 把 UTF-8 当 GBK 读」产生的乱码，
再拿 __pycache__ 里损坏前的编译产物当字典，逐个常量做字面替换。
"""
import ctypes
import marshal
import sys
import types
from pathlib import Path

k32 = ctypes.WinDLL("kernel32", use_last_error=True)


def mb2wc(b: bytes, cp: int = 936) -> str:
    n = k32.MultiByteToWideChar(cp, 0, b, len(b), None, 0)
    buf = ctypes.create_unicode_buffer(n)
    k32.MultiByteToWideChar(cp, 0, b, len(b), buf, n)
    return buf[:n]


def mangle(s: str) -> str:
    return mb2wc(s.encode("utf-8")).replace("\ufffd", "?")


src, pyc, dst = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
raw = src.read_bytes()
if raw[:3] == b"\xef\xbb\xbf":
    raw = raw[3:]
text = raw.decode("utf-8")

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
miss = []
for s in strs:
    m = mangle(s)
    if not m or m == s:
        continue
    done = False
    for cand in (m, m.replace("\n", "\r\n")):
        if cand in text:
            n = text.count(cand)
            text = text.replace(cand, s.replace("\n", "\r\n") if "\r\n" in cand else s)
            fixed += n
            done = True
    if not done and s not in text:
        miss.append(s)

dst.write_text(text, encoding="utf-8")
print(f"还原 {fixed} 处；未能匹配 {len(miss)} 个常量")
for s in miss:
    print("  未匹配:", s[:50].replace("\n", "|"))
left = [(i + 1, l) for i, l in enumerate(text.splitlines()) if "?" in l]
print(f"仍含 '?': {len(left)} 行")
for i, l in left:
    print(f"  {i}: {l.strip()[:120]}")
