from src.rag.document_loader import load_documents


def main():
    print("Loading Microsoft documents...\n")

    documents = load_documents()

    print("\nFirst document:")
    print(documents[0].page_content[:1000])

    print("\nMetadata:")
    print(documents[0].metadata)


if __name__ == "__main__":
    main()