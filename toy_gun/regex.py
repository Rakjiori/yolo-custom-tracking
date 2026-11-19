import re as _re
from re import *

IGNORECASE = _re.IGNORECASE

_MAP = [
    (r"[^\s\p{L}\p{N}]+", r"(?:[^\s\w]|_)+"),
    (r"[\p{L}]+", r"[^\W\d_]+"),
    (r"[\p{N}]", r"\d"),
    (r"\p{L}", r"[^\W\d_]"),
    (r"\p{N}", r"\d"),
]


def _tr(pattern):
    if isinstance(pattern, str):
        for a, b in _MAP:
            pattern = pattern.replace(a, b)
    return pattern


def compile(pattern, flags=0):
    return _re.compile(_tr(pattern), flags)


def findall(pattern, string, flags=0):
    return _re.findall(_tr(pattern), string, flags)


def sub(pattern, repl, string, count=0, flags=0):
    return _re.sub(_tr(pattern), repl, string, count=count, flags=flags)
