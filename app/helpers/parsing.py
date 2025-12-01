import os
import re
from pathlib import Path
from typing import List
from pdfminer.high_level import extract_text as pdf_extract
from docx import Document
from unstructured.partition.auto import partition
from app.models.models import DocRecord
import logging
logging.getLogger("pdfminer").setLevel(logging.ERROR)

def read_txt(p: Path) -> str:
    return p.read_text(errors="ignore")

def read_docx(p: Path) -> str:
    doc = Document(str(p))
    return "\n".join([p.text for p in doc.paragraphs])

def read_pdf(p: Path) -> str:
    try:
        return pdf_extract(str(p))
    except Exception:
        # fallback to unstructured
        elems = partition(filename=str(p))
        return "\n".join([e.text for e in elems if hasattr(e, "text") and e.text])

def clean_text(x: str) -> str:
    x = re.sub(r'\s+', ' ', x).strip()
    return x

def load_folder(folder: str) -> List[DocRecord]:
    out = []
    for root, _, files in os.walk(folder):
        for f in files:
            p = Path(root) / f
            ext = p.suffix.lower()
            if ext in [".txt"]:
                t = read_txt(p)
            elif ext in [".pdf"]:
                t = read_pdf(p)
            elif ext in [".docx"]:
                t = read_docx(p)
            else:
                continue
            out.append(DocRecord(id=p.stem, path=str(p), text=clean_text(t)))
    return out
