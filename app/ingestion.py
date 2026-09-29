from pathlib import Path

from pypdf import PdfReader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import CHUNK_SIZE, CHUNK_OVERLAP


PDF_PATH = Path("data/Agentic-AI.pdf")


def load_and_split_pdf(
    pdf_path: str | Path = PDF_PATH,
) -> list[Document]:
    """
    Extract text from a PDF and split it into chunks.

    Each chunk retains its source file and page number.
    """

    pdf_path = Path(pdf_path)

    if not pdf_path.exists():
        raise FileNotFoundError(
            f"PDF file not found: {pdf_path}"
        )

    reader = PdfReader(str(pdf_path))

    documents = []

    # Extract text page by page.
    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""

        if not text.strip():
            continue

        documents.append(
            Document(
                page_content=text,
                metadata={
                    "source": pdf_path.name,
                    "page": page_number,
                },
            )
        )

    if not documents:
        raise ValueError(
            "No extractable text found in the PDF."
        )

    # Split page text into smaller chunks.
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""],
    )

    chunks = splitter.split_documents(documents)

    return chunks