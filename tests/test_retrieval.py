from src.rag.retriever import retrieve_documents


def main():
    query = "What was Microsoft's revenue in fiscal year 2025?"

    results = retrieve_documents(
        query=query,
        k=5
    )

    print("\n========================================")
    print("RETRIEVAL TEST")
    print("========================================")

    print(f"Query: {query}")
    print(f"Documents retrieved: {len(results)}")

    for index, document in enumerate(results, start=1):
        print("\n----------------------------------------")
        print(f"Result {index}")
        print("----------------------------------------")

        print("Source:", document.metadata.get("source"))
        print("Page:", document.metadata.get("page"))
        print("\nContent:")
        print(document.page_content[:1000])

    print("\n========================================")


if __name__ == "__main__":
    main()