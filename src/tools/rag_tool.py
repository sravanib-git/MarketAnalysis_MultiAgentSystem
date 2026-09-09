from src.rag.retriever import retrieve_documents


def rag_tool(
    query: str,
    k: int = 5,
    company: str | None = None,
):
    """
    Retrieve relevant documents from the internal RAG system.

    Parameters
    ----------
    query : str
        User's research query.

    k : int
        Number of final documents to retrieve.

    company : str | None
        Company currently being researched.
        This prevents cross-company retrieval.
    """

    documents = retrieve_documents(
        query=query,
        k=k,
        company=company,
    )

    return documents


if __name__ == "__main__":

    question = "What was Microsoft's revenue in fiscal year 2025?"

    documents = rag_tool(
        query=question,
        k=5,
        company="Microsoft",
    )

    print("\n========================================")
    print("RAG TOOL TEST")
    print("========================================")

    print(f"Question: {question}")
    print(f"Documents retrieved: {len(documents)}")

    for rank, document in enumerate(documents, start=1):

        print(f"\n--- Result {rank} ---")

        print(document.page_content[:500])

        print("\nMetadata:")
        print(document.metadata)