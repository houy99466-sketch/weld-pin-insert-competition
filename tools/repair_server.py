# -*- coding: utf-8 -*-
"""
server.py 乱码修复：用 __pycache__ 里「损坏前」的编译产物当字典，
把源码里被 GBK 误解码 + 丢字节的字符串/文档串按原文逐个还原。
（注释不在 pyc 里，剩下的 '?' 会列出来手工补。）
"""
import marshal
import re
import sys
import types
from pathlib import Path

src = Path(sys.argv[1])
pyc = Path(sys.argv[2])
dst = Path(sys.argv[3]) if len(sys.argv) > 3 else src.with_suffix(".fixed.py")

raw = src.read_bytes()
if raw[:3] == b"\xef\xbb\xbf":
    raw = raw[3:]
text = raw.decode("utf-8")

code = marshal.loads(pyc.read_bytes()[16:])
strs = []


def walk(c):
    for k in c.co_consts:
        if isinstance(k, str) and any(ord(ch) > 127 for ch in k):
            strs.append(k)
        elif isinstance(k, types.CodeType):
            walk(k)


walk(code)
strs = sorted(set(strs), key=len, reverse=True)
print(f"pyc 中带中文的字符串常量: {len(strs)} 个，最长 {len(strs[0]) if strs else 0} 字")


def mangle(s: str) -> str:
    return s.encode("utf-8").decode("gbk", errors="replace").replace("\ufffd", "?")


def flexible_regex(m: str) -> re.Pattern:
    segs = [s for s in m.split("?") if s]
    if not segs:
        return None
    pat = "[?]{0,6}".join(re.escape(s) for s in segs)
    return re.compile(pat)


fixed = 0
for s in strs:
    if s in text:          # 已经完好
        continue
    m = mangle(s)
    rx = flexible_regex(m)
    if rx is None:
        continue
    mm = rx.search(text)
    if not mm:
        continue
    new, n = rx.subn(lambda _m: s, text)
    if n:
        text = new
        fixed += n

print(f"还原字符串/文档串: {fixed} 处")

dst.write_text(text, encoding="utf-8")
print(f"写出 {dst}")

left = [(i + 1, l) for i, l in enumerate(text.splitlines()) if "?" in l]
print(f"仍含 '?' 的行: {len(left)}")
for i, l in left[:40]:
    print(f"  {i}: {l.strip()[:110]}")
