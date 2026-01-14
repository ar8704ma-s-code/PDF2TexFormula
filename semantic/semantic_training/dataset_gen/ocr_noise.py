import random
import re

def random_insert(text):
    pos = random.randint(0, len(text))
    return text[:pos] + random.choice([" ", "{", "}", "^", "_"]) + text[pos:]

def random_delete(text):
    if not text:
        return text
    pos = random.randint(0, len(text) - 1)
    return text[:pos] + text[pos+1:]

def random_replace(text):
    if not text:
        return text
    pos = random.randint(0, len(text) - 1)
    return text[:pos] + random.choice("abcdefghijklmnopqrstuvwxyz") + text[pos+1:]

def apply_ocr_noise(text):
    ops = [random_insert, random_delete, random_replace]
    for _ in range(random.randint(1, 3)):
        text = random.choice(ops)(text)
    return text

def corrupt_formula(f):
    ops = [
        lambda s: s.replace("{ ", "{").replace(" }", "}"),
        lambda s: re.sub(r"\\([a-zA-Z]+)", r" \1", s),  # lose backslash
        lambda s: s.replace("  ", " "),
        lambda s: re.sub(r"([_^\"]) ", r"\1", s),
        lambda s: s.replace("{", " { ").replace("}", " } "),
        lambda s: re.sub(r"\\left\s+", r"\\left ", s),
    ]
    op = random.choice(ops)
    return op(f)
