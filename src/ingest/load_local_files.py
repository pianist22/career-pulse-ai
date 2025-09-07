# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

from pathlib import Path
import re
import cv2
import pytesseract
import fitz  # PyMuPDF
import docx
import pandas as pd
from typing import List, Dict
from src.config import load_config
from src.utils.io_helpers import save_table

def extract_pdf(path: Path) -> str:
    text_parts = []
    with fitz.open(path.as_posix()) as doc:
        for page in doc:
            text_parts.append(page.get_text())
    return "\n".join(text_parts)

def extract_docx(path: Path) -> str:
    d = docx.Document(path.as_posix())
    return "\n".join([p.text for p in d.paragraphs])

def extract_image_ocr(path: Path, lang="eng") -> str:
    img = cv2.imread(path.as_posix())
    if img is None:
        return ""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Simple binarization can boost OCR on scanned resumes
    gray = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY)[12]
    text = pytesseract.image_to_string(gray, lang=lang)
    return text

def extract_text_any(path: Path, ocr_lang="eng") -> str:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return extract_pdf(path)
    if suffix in (".docx",):
        return extract_docx(path)
    if suffix in (".png", ".jpg", ".jpeg", ".tiff"):
        return extract_image_ocr(path, ocr_lang)
    # Fallback: try OCR for unknown formats
    return extract_image_ocr(path, ocr_lang)

def crawl_local_dir(root: Path, ocr_lang="eng") -> List[Dict]:
    records = []
    for p in root.rglob("*"):
        if p.is_file():
            try:
                text = extract_text_any(p, ocr_lang)
                if text and text.strip():
                    records.append({"filepath": p.as_posix(), "text": text})
            except Exception as e:
                # Keep going; log minimal error info for now
                records.append({"filepath": p.as_posix(), "text": ""})
    return records

def main():
    cfg = load_config()
    root = Path(cfg["data"]["local_data_dir"])
    out_dir = Path(cfg["data"]["interim_dir"])
    out_path = out_dir / f"local_raw.{cfg['data']['export_format']}"
    rows = crawl_local_dir(root, cfg["ingestion"]["ocr_lang"])
    df = pd.DataFrame(rows)
    save_table(df, out_path.as_posix())
    print(f"Saved: {out_path}")

if __name__ == "__main__":
    main()
