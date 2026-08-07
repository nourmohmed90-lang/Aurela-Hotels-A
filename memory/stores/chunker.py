from pathlib import Path
from typing import List, Dict

from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

from memory.stores.config import (
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    SUPPORTED_EXTENSIONS,
)

splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=["\n\n", "\n", ". ", " ", ""]
)


def read_text_file(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def read_pdf(path: Path) -> str:
    reader = PdfReader(str(path))
    text = []
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text.append(page_text)
    return "\n".join(text)


def load_document(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in [".md", ".txt"]:
        return read_text_file(path)
    if suffix == ".pdf":
        return read_pdf(path)
    raise ValueError(f"Unsupported file format: {path}")


def clean_text(text: str) -> str:
    text = text.replace("\t", " ")
    while "  " in text:
        text = text.replace("  ", " ")
    return text.strip()


def split_document(path: Path) -> List[Dict]:
    text = clean_text(load_document(path))
    chunks = splitter.split_text(text)

    documents = []
    for index, chunk in enumerate(chunks):
        documents.append({
            "id": f"{path.stem}_{index}",
            "text": chunk,
            "metadata": {
                "source": path.name,
                "chunk": index,
                "file_type": path.suffix.lower()
            }
        })
    return documents


def load_documents(folder: Path) -> List[Dict]:
    documents = []
    if not folder.exists():
        return documents

    for file in folder.iterdir():
        if file.is_file() and file.suffix.lower() in SUPPORTED_EXTENSIONS:
            documents.extend(split_document(file))

    return documents