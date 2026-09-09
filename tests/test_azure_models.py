from src.config.azure_models import (
    get_chat_model,
    get_embedding_model,
)


def main():
    print("Testing Azure Chat Model...")

    llm = get_chat_model()

    response = llm.invoke(
        "Explain what an AI agent is in one simple sentence."
    )

    print("\nChat model response:")
    print(response.content)

    print("\nTesting Azure Embedding Model...")

    embeddings = get_embedding_model()

    vector = embeddings.embed_query(
        "Microsoft cloud computing"
    )

    print("\nEmbedding test successful!")
    print("Number of dimensions:", len(vector))
    print("First five values:", vector[:5])


if __name__ == "__main__":
    main()