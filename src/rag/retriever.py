import re
from typing import Any

from rank_bm25 import BM25Okapi
from langchain_chroma import Chroma

from src.config.azure_models import get_embedding_model
from src.rag.document_loader import load_documents
from src.rag.text_splitter import split_documents


# ============================================================
# CONFIGURATION
# ============================================================

PERSIST_DIRECTORY = "data/vector_store"

COLLECTION_NAME = "microsoft_documents"

DEFAULT_FINAL_K = 5

DEFAULT_FETCH_K = 20

# Weight configuration for reranking
KEYWORD_WEIGHT = 0.25
METRIC_WEIGHT = 0.30
YEAR_WEIGHT = 0.15
PHRASE_WEIGHT = 0.10
SEMANTIC_WEIGHT = 0.20


# ============================================================
# BM25 CACHE
# ============================================================

_BM25_RETRIEVER = None
_BM25_DOCUMENTS = None


# ============================================================
# EMBEDDING CACHE
# ============================================================

_EMBEDDING_MODEL = None


# ============================================================
# COMPANY ALIASES
# ============================================================

COMPANY_ALIASES = {
    "Microsoft": [
        "microsoft",
        "msft",
    ],
    "Amazon": [
        "amazon",
        "amzn",
        "aws",
    ],
    "Nvidia": [
        "nvidia",
        "nvda",
    ],
    "Apple": [
        "apple",
        "aapl",
    ],
    "Alphabet": [
        "alphabet",
        "google",
        "googl",
        "goog",
    ],
    "Meta": [
        "meta",
        "facebook",
    ],
    "Tesla": [
        "tesla",
        "tsla",
    ],
}


# ============================================================
# FINANCIAL METRIC ALIASES
# ============================================================

METRIC_ALIASES = {
    "revenue": [
        "revenue",
        "revenues",
        "sales",
        "total revenue",
        "total revenues",
    ],
    "operating income": [
        "operating income",
        "income from operations",
        "operating profit",
    ],
    "net income": [
        "net income",
        "net earnings",
        "net profit",
        "profit for the year",
        "net income attributable",
    ],
    "diluted earnings per share": [
        "diluted earnings per share",
        "diluted eps",
        "diluted earnings per common share",
        "earnings per share diluted",
    ],
    "earnings per share": [
        "earnings per share",
        "eps",
        "basic earnings per share",
        "diluted earnings per share",
    ],
    "gross margin": [
        "gross margin",
        "gross profit margin",
    ],
    "operating margin": [
        "operating margin",
    ],
    "net margin": [
        "net margin",
        "net profit margin",
    ],
    "cloud revenue": [
        "cloud revenue",
        "microsoft cloud",
        "cloud business",
    ],
    "revenue growth": [
        "revenue growth",
        "revenue increased",
        "revenue increase",
        "growth in revenue",
    ],
}


# ============================================================
# TEXT NORMALIZATION
# ============================================================

def normalize_text(text: str) -> str:
    """
    Normalize text for matching and BM25 tokenization.
    """

    if not text:
        return ""

    text = str(text).lower()

    text = text.replace("’", "'")

    text = re.sub(
        r"[^a-z0-9\s]",
        " ",
        text,
    )

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip()


# ============================================================
# COMPANY NAME NORMALIZATION
# ============================================================

def normalize_company_name(company: str) -> str | None:
    """
    Convert company aliases into canonical company names.
    """

    if not company:
        return None

    normalized_company = normalize_text(company)

    for canonical_name, aliases in COMPANY_ALIASES.items():

        for alias in aliases:

            if normalized_company == normalize_text(alias):

                return canonical_name

    return None


# ============================================================
# DETECT COMPANY FROM QUERY
# ============================================================

def detect_company_from_query(query: str) -> str | None:
    """
    Detect the first company mentioned in a query.
    """

    if not query:
        return None

    normalized_query = normalize_text(query)

    detected_companies = []

    for canonical_name, aliases in COMPANY_ALIASES.items():

        for alias in aliases:

            normalized_alias = normalize_text(alias)

            if re.search(
                rf"\b{re.escape(normalized_alias)}\b",
                normalized_query,
            ):

                detected_companies.append(
                    canonical_name
                )

                break

    if not detected_companies:

        return None

    return detected_companies[0]


# ============================================================
# DETECT COMPANY FROM DOCUMENT
# ============================================================

def detect_company_from_document(document) -> str | None:
    """
    Detect the company associated with a document.

    Source filename is checked first because it is
    generally more reliable than document content.
    """

    metadata = document.metadata or {}

    source = normalize_text(
        str(
            metadata.get(
                "source",
                "",
            )
        )
    )

    page_content = normalize_text(
        document.page_content or ""
    )

    source_company_priority = [
        "microsoft",
        "msft",
        "amazon",
        "amzn",
        "nvidia",
        "nvda",
        "apple",
        "aapl",
        "alphabet",
        "google",
        "googl",
        "goog",
        "meta",
        "facebook",
        "tesla",
        "tsla",
    ]

    for company_alias in source_company_priority:

        if re.search(
            rf"\b{re.escape(company_alias)}\b",
            source,
        ):

            return normalize_company_name(
                company_alias
            )

    combined_text = normalize_text(
        f"{source} {page_content}"
    )

    all_aliases = []

    for canonical_name, aliases in COMPANY_ALIASES.items():

        for alias in aliases:

            all_aliases.append(
                (
                    canonical_name,
                    normalize_text(alias),
                )
            )

    all_aliases.sort(
        key=lambda item: len(item[1]),
        reverse=True,
    )

    for canonical_name, normalized_alias in all_aliases:

        if re.search(
            rf"\b{re.escape(normalized_alias)}\b",
            combined_text,
        ):

            return canonical_name

    return None


# ============================================================
# FILTER DOCUMENTS BY COMPANY
# ============================================================

def filter_documents_by_company(
    documents,
    company: str | None = None,
):
    """
    Filter documents belonging to a specific company.
    """

    if not company:

        return documents

    normalized_company = normalize_company_name(
        company
    )

    if not normalized_company:

        return documents

    filtered_documents = []

    for document in documents:

        document_company = detect_company_from_document(
            document
        )

        if document_company == normalized_company:

            filtered_documents.append(
                document
            )

    return filtered_documents


# ============================================================
# GET VECTOR STORE
# ============================================================

def get_vector_store():
    """
    Load the existing Chroma vector store.
    """

    embeddings = get_embedding_model()

    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=PERSIST_DIRECTORY,
    )

    return vector_store


# ============================================================
# VECTOR SIMILARITY SEARCH
# ============================================================

def retrieve_vector_documents(
    query: str,
    k: int = 5,
    company: str | None = None,
):
    """
    Retrieve documents using Chroma vector similarity.
    """

    print("\nRunning vector similarity search...")

    vector_store = get_vector_store()

    documents = vector_store.similarity_search(
        query=query,
        k=k,
    )

    print(
        f"Vector search returned "
        f"{len(documents)} documents."
    )

    if company:

        documents = filter_documents_by_company(
            documents,
            company,
        )

        print(
            f"After company filtering: "
            f"{len(documents)} documents."
        )

    return documents


# ============================================================
# VECTOR MMR SEARCH
# ============================================================

def retrieve_mmr_documents(
    query: str,
    k: int = 5,
    fetch_k: int = 20,
    lambda_mult: float = 0.5,
    company: str | None = None,
):
    """
    Retrieve documents using Maximal Marginal Relevance.
    """

    print("\nRunning MMR retrieval...")

    vector_store = get_vector_store()

    documents = vector_store.max_marginal_relevance_search(
        query=query,
        k=k,
        fetch_k=fetch_k,
        lambda_mult=lambda_mult,
    )

    print(
        f"MMR returned "
        f"{len(documents)} documents."
    )

    if company:

        documents = filter_documents_by_company(
            documents,
            company,
        )

        print(
            f"After company filtering: "
            f"{len(documents)} documents."
        )

    return documents


# ============================================================
# BUILD BM25 RETRIEVER
# ============================================================

def get_bm25_retriever():
    """
    Build and cache the BM25 index.
    """

    global _BM25_RETRIEVER
    global _BM25_DOCUMENTS

    if _BM25_RETRIEVER is not None:

        return (
            _BM25_RETRIEVER,
            _BM25_DOCUMENTS,
        )

    print("\nBuilding BM25 index...")

    documents = load_documents()

    print(
        f"Loaded {len(documents)} pages for BM25."
    )

    chunks = split_documents(
        documents
    )

    print(
        f"Created {len(chunks)} chunks for BM25."
    )

    tokenized_documents = [

        normalize_text(
            document.page_content
        ).split()

        for document in chunks

    ]

    bm25 = BM25Okapi(
        tokenized_documents
    )

    _BM25_RETRIEVER = bm25
    _BM25_DOCUMENTS = chunks

    print(
        "BM25 index created successfully."
    )

    return (
        _BM25_RETRIEVER,
        _BM25_DOCUMENTS,
    )


# ============================================================
# BM25 SEARCH
# ============================================================

def retrieve_bm25_documents(
    query: str,
    k: int = 5,
    company: str | None = None,
):
    """
    Retrieve documents using BM25 keyword search.
    """

    print("\nRunning BM25 keyword search...")

    bm25, documents = get_bm25_retriever()

    tokenized_query = normalize_text(
        query
    ).split()

    scores = bm25.get_scores(
        tokenized_query
    )

    ranked_indexes = sorted(
        range(len(scores)),
        key=lambda index: scores[index],
        reverse=True,
    )

    selected_documents = []

    normalized_company = normalize_company_name(
        company
    ) if company else None

    for index in ranked_indexes:

        document = documents[index]

        if normalized_company:

            document_company = (
                detect_company_from_document(
                    document
                )
            )

            if document_company != normalized_company:

                continue

        selected_documents.append(
            document
        )

        if len(selected_documents) >= k:

            break

    print(
        f"BM25 returned "
        f"{len(selected_documents)} documents."
    )

    return selected_documents


# ============================================================
# QUERY METRIC DETECTION
# ============================================================

def detect_query_metrics(
    query: str,
) -> list[str]:
    """
    Detect financial metrics requested by the query.
    """

    normalized_query = normalize_text(
        query
    )

    detected_metrics = []

    aliases_sorted = []

    for canonical_metric, aliases in METRIC_ALIASES.items():

        for alias in aliases:

            aliases_sorted.append(
                (
                    canonical_metric,
                    normalize_text(alias),
                )
            )

    aliases_sorted.sort(
        key=lambda item: len(item[1]),
        reverse=True,
    )

    for canonical_metric, alias in aliases_sorted:

        if re.search(
            rf"\b{re.escape(alias)}\b",
            normalized_query,
        ):

            if canonical_metric not in detected_metrics:

                detected_metrics.append(
                    canonical_metric
                )

    return detected_metrics


# ============================================================
# FINANCIAL YEAR DETECTION
# ============================================================

def detect_query_years(
    query: str,
) -> list[str]:
    """
    Detect years and fiscal-year expressions.
    """

    normalized_query = normalize_text(
        query
    )

    years = re.findall(
        r"\b(?:fy\s*)?(20\d{2})\b",
        normalized_query,
    )

    fiscal_years = []

    for year in years:

        fiscal_years.append(
            year
        )

    return list(
        dict.fromkeys(
            fiscal_years
        )
    )


# ============================================================
# KEYWORD OVERLAP
# ============================================================

def calculate_keyword_overlap(
    query: str,
    document,
) -> float:
    """
    Calculate query-token overlap with a document.
    """

    query_tokens = set(
        normalize_text(
            query
        ).split()
    )

    document_tokens = set(
        normalize_text(
            document.page_content
        ).split()
    )

    if not query_tokens:

        return 0.0

    overlap = (
        query_tokens
        & document_tokens
    )

    return len(overlap) / len(query_tokens)


# ============================================================
# METRIC MATCH SCORE
# ============================================================

def calculate_metric_match_score(
    query: str,
    document,
) -> float:
    """
    Score whether the document contains the exact
    financial metric requested by the query.
    """

    query_metrics = detect_query_metrics(
        query
    )

    if not query_metrics:

        return 0.0

    document_text = normalize_text(
        document.page_content
    )

    matched_metrics = 0

    for metric in query_metrics:

        aliases = METRIC_ALIASES.get(
            metric,
            [],
        )

        metric_found = False

        for alias in aliases:

            normalized_alias = normalize_text(
                alias
            )

            if re.search(
                rf"\b{re.escape(normalized_alias)}\b",
                document_text,
            ):

                metric_found = True

                break

        if metric_found:

            matched_metrics += 1

    return matched_metrics / len(query_metrics)


# ============================================================
# YEAR MATCH SCORE
# ============================================================

def calculate_year_match_score(
    query: str,
    document,
) -> float:
    """
    Score whether the document contains the requested year.
    """

    query_years = detect_query_years(
        query
    )

    if not query_years:

        return 0.0

    document_text = normalize_text(
        document.page_content
    )

    matched_years = 0

    for year in query_years:

        if year in document_text:

            matched_years += 1

    return matched_years / len(query_years)


# ============================================================
# PHRASE MATCH SCORE
# ============================================================

def calculate_phrase_match_score(
    query: str,
    document,
) -> float:
    """
    Score exact multi-word phrase matches.
    """

    normalized_query = normalize_text(
        query
    )

    normalized_document = normalize_text(
        document.page_content
    )

    query_tokens = normalized_query.split()

    if len(query_tokens) < 2:

        return 0.0

    phrases = []

    for phrase_length in [4, 3, 2]:

        if len(query_tokens) >= phrase_length:

            for index in range(
                len(query_tokens) - phrase_length + 1
            ):

                phrase = " ".join(
                    query_tokens[
                        index:index + phrase_length
                    ]
                )

                phrases.append(
                    phrase
                )

    if not phrases:

        return 0.0

    matched_phrases = 0

    for phrase in phrases:

        if phrase in normalized_document:

            matched_phrases += 1

    return min(
        matched_phrases / len(phrases),
        1.0,
    )


# ============================================================
# SEMANTIC SIMILARITY SCORE
# ============================================================

def calculate_semantic_similarity(
    query: str,
    document,
) -> float:
    """
    Calculate semantic similarity using the same embedding
    model used by the vector store.

    Chroma's similarity score is not directly available after
    combining BM25 and vector results, so we calculate a
    normalized cosine-like similarity here.
    """

    global _EMBEDDING_MODEL

    if _EMBEDDING_MODEL is None:

        _EMBEDDING_MODEL = get_embedding_model()

    try:

        query_vector = _EMBEDDING_MODEL.embed_query(
            query
        )

        document_vector = _EMBEDDING_MODEL.embed_query(
            document.page_content
        )

        query_norm = sum(
            value * value
            for value in query_vector
        ) ** 0.5

        document_norm = sum(
            value * value
            for value in document_vector
        ) ** 0.5

        if query_norm == 0 or document_norm == 0:

            return 0.0

        dot_product = sum(
            query_value * document_value
            for query_value, document_value
            in zip(
                query_vector,
                document_vector,
            )
        )

        similarity = (
            dot_product
            / (query_norm * document_norm)
        )

        return max(
            0.0,
            min(
                float(similarity),
                1.0,
            ),
        )

    except Exception as error:

        print(
            f"[SEMANTIC SCORE ERROR] {error}"
        )

        return 0.0


# ============================================================
# DOCUMENT RELEVANCE SCORE
# ============================================================

def calculate_document_relevance_score(
    query: str,
    document,
) -> dict[str, float]:
    """
    Calculate multiple relevance signals.
    """

    keyword_score = calculate_keyword_overlap(
        query,
        document,
    )

    metric_score = calculate_metric_match_score(
        query,
        document,
    )

    year_score = calculate_year_match_score(
        query,
        document,
    )

    phrase_score = calculate_phrase_match_score(
        query,
        document,
    )

    semantic_score = calculate_semantic_similarity(
        query,
        document,
    )

    final_score = (

        KEYWORD_WEIGHT
        * keyword_score

        +

        METRIC_WEIGHT
        * metric_score

        +

        YEAR_WEIGHT
        * year_score

        +

        PHRASE_WEIGHT
        * phrase_score

        +

        SEMANTIC_WEIGHT
        * semantic_score

    )

    # Strong penalty when a financial query explicitly asks
    # for a metric that the document does not contain.
    query_metrics = detect_query_metrics(
        query
    )

    if query_metrics and metric_score == 0.0:

        final_score *= 0.35

    return {
        "keyword": keyword_score,
        "metric": metric_score,
        "year": year_score,
        "phrase": phrase_score,
        "semantic": semantic_score,
        "final": final_score,
    }


# ============================================================
# RERANK DOCUMENTS
# ============================================================

def rerank_documents(
    query: str,
    documents,
    k: int = 5,
):
    """
    Rerank documents using:

        1. Keyword overlap
        2. Exact financial metric match
        3. Fiscal-year match
        4. Phrase match
        5. Semantic similarity
    """

    if not documents:

        return []

    scored_documents = []

    for document in documents:

        scores = calculate_document_relevance_score(
            query,
            document,
        )

        scored_documents.append(
            (
                scores["final"],
                scores,
                document,
            )
        )

    scored_documents.sort(
        key=lambda item: item[0],
        reverse=True,
    )

    print("\nRERANKING SCORES")

    for rank, (
        final_score,
        scores,
        document,
    ) in enumerate(
        scored_documents,
        start=1,
    ):

        metadata = document.metadata or {}

        print(
            f"Candidate {rank}: "
            f"final={final_score:.4f}, "
            f"keyword={scores['keyword']:.2f}, "
            f"metric={scores['metric']:.2f}, "
            f"year={scores['year']:.2f}, "
            f"phrase={scores['phrase']:.2f}, "
            f"semantic={scores['semantic']:.2f}, "
            f"page={metadata.get('page', 'Unknown')}"
        )

    return [
        document
        for final_score, scores, document
        in scored_documents[:k]
    ]


# ============================================================
# DOCUMENT DEDUPLICATION
# ============================================================

def deduplicate_documents(
    documents,
):
    """
    Remove duplicate chunks using normalized content.
    """

    unique_documents = []

    seen_content = set()

    for document in documents:

        content = (
            document.page_content
            or ""
        )

        content_key = normalize_text(
            content
        )

        if not content_key:

            continue

        if content_key in seen_content:

            continue

        seen_content.add(
            content_key
        )

        unique_documents.append(
            document
        )

    return unique_documents


# ============================================================
# HYBRID RETRIEVAL
# ============================================================

def retrieve_hybrid_documents(
    query: str,
    k: int = 5,
    vector_k: int = 20,
    bm25_k: int = 20,
    company: str | None = None,
):
    """
    Hybrid retrieval combining:

        1. Chroma vector search
        2. BM25 keyword search
        3. Company filtering
        4. Deduplication
        5. Metric-aware reranking
        6. Semantic reranking
    """

    print("\n========================================")
    print("HYBRID RETRIEVAL")
    print("========================================")

    # ========================================================
    # COMPANY DETECTION
    # ========================================================

    if company:

        company = normalize_company_name(
            company
        )

        print(
            f"Company filter: {company}"
        )

    else:

        detected_company = detect_company_from_query(
            query
        )

        if detected_company:

            company = detected_company

            print(
                f"Detected company: {company}"
            )

    # ========================================================
    # METRIC DETECTION
    # ========================================================

    detected_metrics = detect_query_metrics(
        query
    )

    if detected_metrics:

        print(
            f"Detected metrics: {detected_metrics}"
        )

    detected_years = detect_query_years(
        query
    )

    if detected_years:

        print(
            f"Detected years: {detected_years}"
        )

    # ========================================================
    # VECTOR SEARCH
    # ========================================================

    print("\n[1] VECTOR SEARCH")

    vector_store = get_vector_store()

    vector_documents = vector_store.similarity_search(
        query=query,
        k=vector_k,
    )

    print(
        f"Vector candidates: "
        f"{len(vector_documents)}"
    )

    # ========================================================
    # BM25 SEARCH
    # ========================================================

    print("\n[2] BM25 SEARCH")

    bm25_documents = retrieve_bm25_documents(
        query=query,
        k=bm25_k,
        company=company,
    )

    print(
        f"BM25 candidates: "
        f"{len(bm25_documents)}"
    )

    # ========================================================
    # COMPANY FILTER VECTOR RESULTS
    # ========================================================

    if company:

        vector_documents = filter_documents_by_company(
            vector_documents,
            company,
        )

        print(
            f"Vector candidates after company filtering: "
            f"{len(vector_documents)}"
        )

    # ========================================================
    # COMBINE RESULTS
    # ========================================================

    combined_documents = (
        vector_documents
        + bm25_documents
    )

    print(
        f"\nCombined candidates: "
        f"{len(combined_documents)}"
    )

    # ========================================================
    # DEDUPLICATION
    # ========================================================

    unique_documents = deduplicate_documents(
        combined_documents
    )

    print(
        f"After deduplication: "
        f"{len(unique_documents)}"
    )

    # ========================================================
    # FINAL COMPANY FILTER
    # ========================================================

    if company:

        unique_documents = filter_documents_by_company(
            unique_documents,
            company,
        )

        print(
            f"After final company filtering: "
            f"{len(unique_documents)}"
        )

    # ========================================================
    # RERANK
    # ========================================================

    print("\n[3] METRIC-AWARE RERANKING")

    final_documents = rerank_documents(
        query=query,
        documents=unique_documents,
        k=k,
    )

    print(
        f"Final documents: "
        f"{len(final_documents)}"
    )

    # ========================================================
    # INSPECT RESULTS
    # ========================================================

    inspect_retrieval_results(
        query=query,
        documents=final_documents,
    )

    return final_documents


# ============================================================
# MAIN RETRIEVAL FUNCTION
# ============================================================

def retrieve_documents(
    query: str,
    k: int = 5,
    fetch_k: int = 20,
    lambda_mult: float = 0.5,
    company: str | None = None,
):
    """
    Main retrieval entry point.
    """

    return retrieve_hybrid_documents(
        query=query,
        k=k,
        vector_k=fetch_k,
        bm25_k=fetch_k,
        company=company,
    )


# ============================================================
# SIMILARITY INSPECTION
# ============================================================

def inspect_retrieval_results(
    query: str,
    documents,
):
    """
    Print retrieved documents and relevance signals.
    """

    print("\n========================================")
    print("RETRIEVAL RESULTS")
    print("========================================")

    if not documents:

        print(
            "No documents retrieved."
        )

        return

    for index, document in enumerate(
        documents,
        start=1,
    ):

        company = detect_company_from_document(
            document
        )

        metadata = document.metadata or {}

        source = metadata.get(
            "source",
            "Unknown",
        )

        page = metadata.get(
            "page",
            "Unknown",
        )

        scores = calculate_document_relevance_score(
            query,
            document,
        )

        print("\n----------------------------------------")

        print(
            f"Rank       : {index}"
        )

        print(
            f"Company    : {company}"
        )

        print(
            f"Source     : {source}"
        )

        print(
            f"Page       : {page}"
        )

        print(
            f"Final Score: {scores['final']:.4f}"
        )

        print(
            f"Keyword    : {scores['keyword']:.2f}"
        )

        print(
            f"Metric     : {scores['metric']:.2f}"
        )

        print(
            f"Year       : {scores['year']:.2f}"
        )

        print(
            f"Phrase     : {scores['phrase']:.2f}"
        )

        print(
            f"Semantic   : {scores['semantic']:.2f}"
        )

        print(
            "Content:"
        )

        print(
            document.page_content[:500]
        )


# ============================================================
# MAIN TEST
# ============================================================

if __name__ == "__main__":

    print(
        "\n########################################"
    )

    print(
        "# RETRIEVER TEST"
    )

    print(
        "########################################"
    )

    test_queries = [
        "Microsoft revenue in fiscal year 2025",
        "Microsoft operating income in fiscal year 2025",
        "Microsoft net income in fiscal year 2025",
        "Microsoft diluted earnings per share in fiscal year 2025",
        "Microsoft revenue growth from fiscal year 2024 to fiscal year 2025",
    ]

    for test_query in test_queries:

        print(
            "\n\n========================================"
        )

        print(
            f"Query: {test_query}"
        )

        print(
            "========================================"
        )

        documents = retrieve_documents(
            query=test_query,
            k=5,
            fetch_k=20,
            company="Microsoft",
        )

        print(
            f"\nDocuments retrieved: "
            f"{len(documents)}"
        )

    print(
        "\n========================================"
    )

    print(
        "TEST COMPLETED"
    )

    print(
        "========================================"
    )