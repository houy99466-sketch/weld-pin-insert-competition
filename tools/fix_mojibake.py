# -*- coding: utf-8 -*-
"""
修复被 PowerShell 误当 GBK 读、再以 UTF-8 写回的文件（乱码还原）。

原理：原文件是 UTF-8 字节；Get-Content 用 cp936 解码成乱码文本；Set-Content 又以
UTF-8 写出。所以现在文件里每个字符都是「原 UTF-8 某字节对按 GBK 解码的结果」。
只要按 GBK 的字符→字节 反查表逐字还原，就能拿回原始字节，再按 UTF-8 解码。
"""
import sys
from pathlib import Path

src = Path(sys.argv[1])
dst = Path(sys.argv[2]) if len(sys.argv) > 2 else src.with_suffix(".recovered")

# 建 GBK 反查表（含 PUA，Python 自带 codec 编码不了的那些也能查回来）
char2bytes = {}
for lead in range(0x81, 0xFF):
    for trail in list(range(0x40, 0x7F)) + list(range(0x80, 0xFF)):
        pair = bytes([lead, trail])
        try:
            ch = pair.decode("gbk")
        except Exception:
            continue
        if len(ch) == 1:
            char2bytes.setdefault(ch, pair)
# 单字节 0x80（GBK 里是欧元符）
try:
    char2bytes.setdefault(b"\x80".decode("gbk"), b"\x80")
except Exception:
    pass

raw = src.read_bytes()
if raw[:3] == b"\xef\xbb\xbf":
    raw = raw[3:]
text = raw.decode("utf-8")

out = bytearray()
missing = {}
for ch in text:
    o = ord(ch)
    if o < 0x80:
        out.append(o)
        continue
    b = char2bytes.get(ch)
    if b is None:
        missing[ch] = missing.get(ch, 0) + 1
        out += ch.encode("utf-8")      # 兜底：原样保留
    else:
        out += b

try:
    fixed = bytes(out).decode("utf-8")
except UnicodeDecodeError as e:
    print("还原后不是合法 UTF-8：", e)
    print("缺失字符：", {k: v for k, v in list(missing.items())[:10]})
    sys.exit(1)

dst.write_text(fixed, encoding="utf-8")
print(f"还原 OK -> {dst}")
print(f"  字节 {len(raw)} -> {len(out)}   缺失字符种类 {len(missing)}")
print("  抽样：", fixed.splitlines()[101][:70] if len(fixed.splitlines()) > 101 else "")
