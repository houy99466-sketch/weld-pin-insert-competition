# -*- coding: utf-8 -*-
"""
第二代修复：用 difflib 把 pyc 里的原文对齐到乱码源码上，替换整段。
比第一代（严格锚点正则）宽容，能吃掉被 GBK 吃掉的字节造成的错位。
"""
import difflib
import marshal
import re
import sys
import types
from pathlib import Path

src = Path(sys.argv[1])
pyc = Path(sys.argv[2])
dst = Path(sys.argv[3]) if len(sys.argv) > 3 else src.with_suffix(".fixed2.py")

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
        elif isinstance(c, types.CodeType):
            walk(c)


def walk(c):
    for k in c.co_consts:
        if isinstance(k, str) and any(ord(ch) > 127 for ch in k):
            strs.append(k)
        elif isinstance(k, types.CodeType):
            walk(k)


walk(code)
strs = sorted(set(strs), key=len, reverse=True)


def mangle(s: str) -> str:
    return s.encode("utf-8").decode("gbk", errors="replace").replace("\ufffd", "?")


def squash(t: str) -> str:
    """比对用：把 '?' 全部去掉，避免丢字节造成的错位干扰。"""
    return t.replace("?", "")


lines = text.splitlines(keepends=True)
fixed = 0
skipped = []

for s in strs:
    if not s.strip():
        continue
    m = mangle(s)
    target = squash(m)
    if not target:
        continue
    # 在源码里找“去掉 ? 后最像”的行区间（常量可能跨行，最多 30 行）
    best = None
    for i in range(len(lines)):
        acc = ""
        for j in range(i, min(len(lines), i + 30)):
            acc += lines[j]
            if len(acc) > len(m) + 40:
                break
            cand = squash(acc)
            if target in cand:
                best = (i, j)
                break
        if best:
            break
    if best is None:
        skipped.append(s)
        continue
    i, j = best
    block = "".join(lines[i:j + 1])
    sm = difflib.SequenceMatcher(None, squash(block), target, autojunk=False)
    if sm.ratio() < 0.75 and len(target) > 20:
        skipped.append(s)
        continue
    # 用 difflib 在原块上做替换：把 block 中与 target 对应的片段换成 s
    cb = squash(block)
    ops = difflib.SequenceMatcher(None, cb, target, autojunk=False).get_opcodes()
    # 把 cb 的下标映射回 block 的下标
    idx_map = []
    for pos, ch in enumerate(block):
        if ch != "?":
            idx_map.append(pos)
    out = []
    cursor = 0
    for tag, a1, a2, b1, b2 in ops:
        p1 = idx_map[a1] if a1 < len(idx_map) else len(block)
        p2 = idx_map[a2] if a2 < len(idx_map) else len(block)
        out.append(block[cursor:p1])
        if tag in ("replace", "insert"):
            out.append(s[b1:b2])
        cursor = p2
    out.append(block[cursor:])
    newblock = "".join(out)
    lines[i:j + 1] = [newblock]
    fixed += 1

text2 = "".join(lines)
dst.write_text(text2, encoding="utf-8")
print(f"替换 {fixed} 处，跳过 {len(skipped)} 个常量")
for s in skipped[:15]:
    print("  跳过:", s[:40].replace("\n", "|"))
left = [(k + 1, l) for k, l in enumerate(text2.splitlines()) if "?" in l]
print(f"仍含 '?': {len(left)} 行")
for k, l in left[:50]:
    print(f"  {k}: {l.strip()[:110]}")
