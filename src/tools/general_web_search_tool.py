import os
import re
from typing import Any

from dotenv import load_dotenv
from langchain_tavily import TavilySearch


load_dotenv()


MAX_RESULTS = 10


# Official and generally reliable sources
PREFERRED_SOURCE_KEYWORDS = [
    "microsoft.com",
    "azure.microsoft.com",
    "learn.microsoft.com",
    "amazon.com",
    "aws.amazon.com",
    "nvidia.com",
    "investor.nvidia.com",
    "apple.com",
    "abc.xyz",
    "aboutamazon.com",
    "about.fb.com",
    "tesla.com",
    "oracle.com",
    "ibm.com",
    "sec.gov",
    "reuters.com",
    "bloomberg.com",
    "ft.com",
    "wsj.com",
    "techcrunch.com",
    "theverge.com",
    "cnbc.com",
]

# Sources and content that are usually unsuitable
EXCLUDED_SOURCE_KEYWORDS = [
    "pinterest.com",
    "facebook.com",
    "instagram.com",
    "tiktok.com",
    "youtube.com",
    "reddit.com",
    "quora.com",
]

EXCLUDED_CONTENT_TERMS = [
    "buy now",
    "discount",
    "coupon",
    "casino",
    "gambling",
    "porn",
    "adult content",
]


def validate_tavily_api_key() -> None:
    """
    Validate that the Tavily API key is available.
    """

    api_key = os.getenv("TAVILY_API_KEY")

    if not api_key:
        raise ValueError(
            "TAVILY_API_KEY is missing. "
            "Add it to your .env file."
        )


def web_search_tool(
    query: str,
    max_results: int = MAX_RESULTS,
) -> dict[str, Any]:
    """
    Execute a general web search using Tavily.

    Returns the raw Tavily response as a dictionary.
    """

    validate_tavily_api_key()

    search = TavilySearch(
        max_results=max_results,
        topic="general",
    )

    response = search.invoke(
        {
            "query": query,
        }
    )

    if not isinstance(response, dict):
        return {
            "query": query,
            "results": [],
            "raw_response": response,
        }

    return response


def get_result_text(result: dict[str, Any]) -> str:
    """
    Combine useful text fields from one search result.
    """

    title = result.get("title", "")
    content = result.get("content", "")
    url = result.get("url", "")

    return f"{title} {content} {url}".lower()


def normalize_text(text: str) -> str:
    """
    Normalize text for duplicate detection.
    """

    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^a-z0-9 ]", "", text)

    return text.strip()


def is_preferred_source(result: dict[str, Any]) -> bool:
    """
    Check whether a result comes from a preferred source.
    """

    url = str(result.get("url", "")).lower()

    return any(
        keyword in url
        for keyword in PREFERRED_SOURCE_KEYWORDS
    )


def is_excluded_source(result: dict[str, Any]) -> bool:
    """
    Check whether a result comes from an excluded source.
    """

    url = str(result.get("url", "")).lower()

    return any(
        keyword in url
        for keyword in EXCLUDED_SOURCE_KEYWORDS
    )


def contains_excluded_content(result: dict[str, Any]) -> bool:
    """
    Remove clearly unsuitable content.
    """

    text = get_result_text(result)

    return any(
        term in text
        for term in EXCLUDED_CONTENT_TERMS
    )


def remove_duplicate_results(
    results: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Remove duplicate results using URL and title/content.
    """

    unique_results = []
    seen_urls = set()
    seen_text = set()

    for result in results:
        url = str(result.get("url", "")).strip().lower()

        title = str(result.get("title", ""))
        content = str(result.get("content", ""))

        normalized_result_text = normalize_text(
            f"{title} {content[:300]}"
        )

        if url and url in seen_urls:
            continue

        if normalized_result_text in seen_text:
            continue

        if url:
            seen_urls.add(url)

        seen_text.add(normalized_result_text)
        unique_results.append(result)

    return unique_results


def filter_general_results(
    results: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Filter and rank general web-search results.

    Preferred sources are placed first.
    """

    filtered_results = []

    for result in results:
        if not isinstance(result, dict):
            continue

        if is_excluded_source(result):
            continue

        if contains_excluded_content(result):
            continue

        title = result.get("title", "")
        content = result.get("content", "")
        url = result.get("url", "")

        if not title and not content and not url:
            continue

        filtered_results.append(result)

    filtered_results = remove_duplicate_results(
        filtered_results
    )

    # Put preferred sources first
    filtered_results.sort(
        key=lambda result: (
            is_preferred_source(result),
            result.get("score", 0),
        ),
        reverse=True,
    )

    return filtered_results


def search_general_web(
    query: str,
    max_results: int = MAX_RESULTS,
) -> dict[str, Any]:
    """
    Perform a general web search and return filtered results.
    """

    raw_response = web_search_tool(
        query=query,
        max_results=max_results,
    )

    original_results = raw_response.get("results", [])

    filtered_results = filter_general_results(
        original_results
    )

    return {
        "query": query,
        "results": filtered_results,
        "original_results": original_results,
        "result_count": len(filtered_results),
    }


def format_general_search_results(
    search_response: dict[str, Any],
) -> str:
    """
    Convert general web-search results into readable text
    for the Analysis Agent.
    """

    results = search_response.get("results", [])

    if not results:
        return "No relevant general web-search results found."

    formatted_results = []

    for index, result in enumerate(results, start=1):
        title = result.get("title", "Untitled")
        url = result.get("url", "URL unavailable")
        content = result.get(
            "content",
            "No content available.",
        )

        formatted_results.append(
            f"""
--- General Web Result {index} ---
Title: {title}
URL: {url}
Content:
{content}
""".strip()
        )

    return "\n\n".join(formatted_results)


if __name__ == "__main__":
    test_query = (
        "Compare Microsoft's cloud business "
        "with Amazon's cloud business"
    )

    print("\n" + "=" * 60)
    print("GENERAL WEB SEARCH TEST")
    print("=" * 60)

    print(f"Query: {test_query}")

    response = search_general_web(
        query=test_query,
        max_results=10,
    )

    print(
        f"Filtered results: "
        f"{response['result_count']}"
    )

    print("\n" + "-" * 60)

    for index, result in enumerate(
        response["results"],
        start=1,
    ):
        print(f"\nResult {index}")
        print(f"Title: {result.get('title')}")
        print(f"URL: {result.get('url')}")
        print(
            f"Content: "
            f"{result.get('content', '')[:500]}"
        )