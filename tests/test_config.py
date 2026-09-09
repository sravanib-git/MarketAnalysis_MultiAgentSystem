from src.config.settings import (
    AZURE_OPENAI_API_KEY,
    AZURE_OPENAI_ENDPOINT,
)


def test_configuration_variables_exist():
    assert AZURE_OPENAI_API_KEY is not None
    assert AZURE_OPENAI_ENDPOINT is not None