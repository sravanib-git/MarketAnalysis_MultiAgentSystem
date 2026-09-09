from pathlib import Path

from src.rag.retriever import (
    get_vector_store,
    retrieve_documents,
)


QUESTIONS = [
    "What was Microsoft's total revenue in fiscal year 2025?",
    "What was Microsoft's operating income in fiscal year 2025?",
    "What was Microsoft's net income in fiscal year 2025?",
    "What drove Microsoft's revenue growth in fiscal year 2025?",
    "What was Microsoft Cloud revenue in fiscal year 2025?",
    "What are Microsoft's major business segments?",
]


def display_documents(documents):
    """
    Display the source and page for each retrieved document.
    """

    for rank, document in enumerate(documents, start=1):
        source = Path(
            document.metadata.get(
                "source",
                "Unknown source"
            )
        ).name

        page = document.metadata.get(
            "page",
            "Unknown page"
        )

        preview = document.page_content[:180]
        preview = preview.replace("\n", " ")

        print(f"\n  Rank {rank}")
        print(f"  Source: {source}")
        print(f"  Page: {page}")
        print(f"  Preview: {preview}...")


def evaluate_question(query: str):
    """
    Compare normal similarity search with MMR retrieval.
    """

    vector_store = get_vector_store()

    print("\n" + "=" * 80)
    print(f"QUESTION: {query}")
    print("=" * 80)

    print("\nNORMAL SIMILARITY SEARCH")
    print("-" * 80)

    similarity_results = vector_store.similarity_search(
        query=query,
        k=5,
    )

    display_documents(similarity_results)

    print("\nMMR RETRIEVAL")
    print("-" * 80)

    mmr_results = retrieve_documents(
        query=query,
        k=5,
        fetch_k=10,
        lambda_mult=0.7,
    )

    display_documents(mmr_results)


def main():
    for query in QUESTIONS:
        evaluate_question(query)

    print("\n" + "=" * 80)
    print("RETRIEVAL EVALUATION COMPLETED")
    print("=" * 80)


if __name__ == "__main__":
    main()