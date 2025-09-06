import re

EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
PHONE_RE = re.compile(r"(\+?\d{1,3}[-.\s]?)?(\(?\d{2,4}\)?[-.\s]?)?\d{3,4}[-.\s]?\d{4}")
URL_RE = re.compile(r"https?://\S+|www\.\S+")

def redact_pii(text: str) -> str:
    if not text:
        return ""
    t = EMAIL_RE.sub("[EMAIL]", text)
    t = PHONE_RE.sub("[PHONE]", t)
    t = URL_RE.sub("[URL]", t)
    return t
