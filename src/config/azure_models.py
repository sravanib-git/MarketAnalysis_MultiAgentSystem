from langchain_openai import (
    AzureChatOpenAI,
    AzureOpenAIEmbeddings,
)

from src.config.settings import (
    AZURE_OPENAI_API_KEY,
    AZURE_OPENAI_ENDPOINT,
    AZURE_OPENAI_API_VERSION,
    AZURE_OPENAI_CHAT_DEPLOYMENT,
    AZURE_OPENAI_EMBEDDING_DEPLOYMENT,
)


def get_chat_model():
    """
    Creates and returns the Azure chat model.

    Temperature is intentionally not provided because
    some newer Azure/OpenAI reasoning models only support
    their default temperature value.
    """

    llm = AzureChatOpenAI(
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        api_key=AZURE_OPENAI_API_KEY,
        azure_deployment=AZURE_OPENAI_CHAT_DEPLOYMENT,
        api_version=AZURE_OPENAI_API_VERSION,
    )

    return llm


def get_embedding_model():
    """
    Creates and returns the Azure embedding model.
    """

    embeddings = AzureOpenAIEmbeddings(
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        api_key=AZURE_OPENAI_API_KEY,
        azure_deployment=AZURE_OPENAI_EMBEDDING_DEPLOYMENT,
        api_version=AZURE_OPENAI_API_VERSION,
    )

    return embeddings