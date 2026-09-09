from src.rag.document_loader import load_documents
from src.rag.text_splitter import split_documents


def main():

    print("=" * 80)
    print("HEADING-AWARE CHUNKING TEST")
    print("=" * 80)

    print("\nLoading documents...")

    documents = load_documents()

    print(f"Loaded pages: {len(documents)}")

    print("\nSplitting documents...")

    chunks = split_documents(documents)

    print(f"Created chunks: {len(chunks)}")

    print("\nInspecting first 15 chunks:")

    for index, chunk in enumerate(chunks[:15], start=1):

        source = chunk.metadata.get(
            "source",
            "Unknown",
        )

        page = chunk.metadata.get(
            "page",
            "Unknown",
        )

        section_heading = chunk.metadata.get(
            "section_heading",
            "Unknown",
        )

        start_index = chunk.metadata.get(
            "start_index",
            "Unknown",
        )

        print("\n" + "-" * 80)
        print(f"CHUNK {index}")
        print("-" * 80)

        print(f"Source: {source}")
        print(f"Page: {page}")
        print(f"Section: {section_heading}")
        print(f"Start index: {start_index}")
        print(f"Characters: {len(chunk.page_content)}")

        print("\nContent:")
        print(chunk.page_content[:1000])


if __name__ == "__main__":
    main()