# -*- coding: utf-8 -*-
"""
还原「UTF-8 源文件被 Windows PowerShell 按 cp936 读出、再以 UTF-8 写回」造成的乱码。

关键点：用 ctypes 调 WideCharToMultiByte(936) 做逐字反查，
这样连 GBK 的 PUA 字符也能精确还原成原来的字节对（Python 自带 codec 反而编不回去）。
"""
import ctypes
import sys
from pathlib import Path

k32 = ctypes.WinDLL("kernel32", use_last_error=True)
CP936 = 936


def char_to_cp936(ch: str) -> bytes:
    buf = ctypes.create_string_buffer(8)
    n = k32.WideCharToMultiByte(CP936, 0, ch, 1, buf, len(buf), None, None)
    if n <= 0:
        raise OSError(ctypes.get_last_error(), "WideCharToMultiByte failed")
    return buf.raw[:n]


def main():
    src = Path(sys.argv[1])
    dst = Path(sys.argv[2]) if len(sys.argv) > 2 else src.with_suffix(".fixed")
    raw = src.read_bytes()
    if raw[:3] == b"\xef\xbb\xbf":
        raw = raw[3:]
    text = raw.decode("utf-8")

    out = bytearray()
    for ch in text:
        if ord(ch) < 0x80:
            out.append(ord(ch))
        else:
            out += char_to_cp936(ch)

    try:
        fixed = bytes(out).decode("utf-8")
    except UnicodeDecodeError as e:
        print("还原后仍不是合法 UTF-8：", e)
        print(bytes(out)[max(0, e.start - 40):e.start + 40].hex(" "))
        return 1

    dst.write_text(fixed, encoding="utf-8")
    print(f"还原 OK -> {dst}")
    print(f"  字符 {len(text)}  字节 {len(out)}")
    print("  抽样:", fixed.splitlines()[100][:60] if len(fixed.splitlines()) > 100 else "")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
