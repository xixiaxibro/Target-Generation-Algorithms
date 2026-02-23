"""
IPv6 address format conversions.

Ported from 6tree function1_T (C++ by Zhizhu Liu, 2019).

Supported formats
-----------------
std  Standard notation  2001:0db8:85a3:0000:0000:8a2e:0370:7334
b4   32-char hex        20010db885a300000000 8a2e03707334
b1   128-char binary    00100000000000010000110110111000...
b2   64-char base-4     20010db885a3000000008a2e03707334 (pairs of bits)
b3   42-char octal      (126 significant bits; first 2 bits dropped)
b5   25-char base-32    (125 significant bits; first 3 bits dropped)

All internal processing in this project uses b4 (32 lowercase hex chars).
"""

import ipaddress

_HEX = "0123456789abcdef"
_BASE32_CHARS = "0123456789abcdefghijklmnopqrstuv"  # 0–31


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def normalize_to_b4(addr: str) -> str | None:
    """
    Convert any valid IPv6 address (any format) to b4.

    Tries standard IPv6 notation first, then falls back to the custom
    6tree formats (b1, b2, b3, b4, b5).

    Returns None if the input cannot be parsed.
    """
    addr = addr.strip()

    # Try standard IPv6 (handles :: compression etc.)
    try:
        ip = ipaddress.ip_address(addr)
        if ip.version == 6:
            return ip.exploded.replace(":", "").lower()
    except ValueError:
        pass

    # Already b4?
    if _is_b4(addr):
        return addr.lower()

    # b1 → b4
    if _is_b1(addr):
        return _b1_to_b4(addr)

    # b2 → b4
    if _is_b2(addr):
        return _b2_to_b4(addr)

    # b3 → b4
    if _is_b3(addr):
        return _b3_to_b4(addr)

    # b5 → b4
    if _is_b5(addr):
        return _b5_to_b4(addr)

    return None


def b4_to_std(b4: str) -> str:
    """Convert 32-char hex string to full standard IPv6 notation."""
    if len(b4) != 32:
        raise ValueError(f"Expected 32-char hex string, got {len(b4)}")
    groups = [b4[i : i + 4] for i in range(0, 32, 4)]
    return ":".join(groups)


def std_to_b4(addr: str) -> str | None:
    """Convert standard IPv6 address to 32-char hex (b4)."""
    return normalize_to_b4(addr)


def b4_to_b1(b4: str) -> str:
    """32-char hex → 128-char binary string."""
    return bin(int(b4, 16))[2:].zfill(128)


def b4_to_b2(b4: str) -> str:
    """32-char hex → 64-char base-4 string."""
    b1 = b4_to_b1(b4)
    return "".join(str((int(b1[i]) << 1) | int(b1[i + 1])) for i in range(0, 128, 2))


def b4_to_b3(b4: str) -> str:
    """32-char hex → 42-char octal string (first 2 bits dropped)."""
    b1 = b4_to_b1(b4)[2:]  # drop first 2 bits → 126 bits
    return "".join(
        str(int(b1[i : i + 3], 2)) for i in range(0, 126, 3)
    )


def b4_to_b5(b4: str) -> str:
    """32-char hex → 25-char base-32 string (first 3 bits dropped)."""
    b1 = b4_to_b1(b4)[3:]  # drop first 3 bits → 125 bits
    return "".join(
        _BASE32_CHARS[int(b1[i : i + 5], 2)] for i in range(0, 125, 5)
    )


# ---------------------------------------------------------------------------
# Format detection
# ---------------------------------------------------------------------------

def _is_b4(s: str) -> bool:
    return len(s) == 32 and all(c in _HEX for c in s.lower())


def _is_b1(s: str) -> bool:
    return len(s) == 128 and all(c in "01" for c in s)


def _is_b2(s: str) -> bool:
    return len(s) == 64 and all(c in "0123" for c in s)


def _is_b3(s: str) -> bool:
    return len(s) == 42 and all(c in "01234567" for c in s)


def _is_b5(s: str) -> bool:
    valid = set(_BASE32_CHARS)
    return len(s) == 25 and all(c.lower() in valid for c in s)


# ---------------------------------------------------------------------------
# Conversions to b4
# ---------------------------------------------------------------------------

def _b1_to_b4(b1: str) -> str:
    """128-char binary → 32-char hex."""
    return hex(int(b1, 2))[2:].zfill(32)


def _b2_to_b4(b2: str) -> str:
    """64-char base-4 → 32-char hex (via binary)."""
    b1 = "".join(bin(int(c))[2:].zfill(2) for c in b2)
    return _b1_to_b4(b1)


def _b3_to_b4(b3: str) -> str:
    """42-char octal → 32-char hex (via binary, prepend 2 zero bits)."""
    b1_core = "".join(bin(int(c))[2:].zfill(3) for c in b3)  # 126 bits
    b1 = "00" + b1_core  # restore 128 bits
    return _b1_to_b4(b1)


def _b5_to_b4(b5: str) -> str:
    """25-char base-32 → 32-char hex (via binary, prepend 3 zero bits)."""
    b1_core = "".join(
        bin(_BASE32_CHARS.index(c.lower()))[2:].zfill(5) for c in b5
    )  # 125 bits
    b1 = "000" + b1_core  # restore 128 bits
    return _b1_to_b4(b1)
