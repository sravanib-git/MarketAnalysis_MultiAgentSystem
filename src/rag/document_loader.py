from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader


DATA_DIR = Path("data/microsoft")


def load_documents():
    documents = []

    pdf_files = list(DATA_DIR.glob("*.pdf"))

    if not pdf_files:
        raise FileNotFoundError(
            f"No PDF files found in {DATA_DIR}"
        )

    for pdf_file in pdf_files:
        loader = PyPDFLoader(str(pdf_file))
        docs = loader.load()

        documents.extend(docs)

        print(
            f"Loaded {pdf_file.name}: "
            f"{len(docs)} pages"
        )

    print(f"\nTotal pages loaded: {len(documents)}")

    return documents