import re

URL_RE = re.compile(r"https?://\S+|www\.\S+")
HANDLE_RE = re.compile(r"[@#]\w+")
NON_ALNUM_RE = re.compile(r"[^a-zA-Z0-9\s.,:/@#+-]")
MULTISPACE_RE = re.compile(r"\s+")
CONTRACTIONS = {
    "can't": "cannot", "won't": "will not", "i'm": "i am",
    "don't": "do not", "it's": "it is", "that's": "that is",
}

def expand_contractions(text: str) -> str:
    for k, v in CONTRACTIONS.items():
        text = re.sub(rf"\b{k}\b", v, text, flags=re.IGNORECASE)
    return text

def normalize_text(
    text: str,
    lower_case: bool = True,
    remove_urls: bool = True,
    remove_handles: bool = True,
    remove_specials: bool = True,
    expand_contr: bool = True,
) -> str:
    if not text:
        return ""
    t = text
    if lower_case:
        t = t.lower()
    if expand_contr:
        t = expand_contractions(t)
    if remove_urls:
        t = URL_RE.sub(" ", t)
    if remove_handles:
        t = HANDLE_RE.sub(" ", t)
    if remove_specials:
        t = NON_ALNUM_RE.sub(" ", t)
    t = MULTISPACE_RE.sub(" ", t).strip()
    return t
