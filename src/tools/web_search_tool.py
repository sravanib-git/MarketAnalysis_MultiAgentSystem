import json
import os
import re
from typing import Any
from urllib.parse import urlparse

from dotenv import load_dotenv
from langchain_tavily import TavilySearch


# ============================================================
# ENVIRONMENT
# ============================================================

load_dotenv()


# ============================================================
# CONFIGURATION
# ============================================================

MAX_RESULTS = 10


# ============================================================
# COMPANY ALIASES
# ============================================================

COMPANY_ALIASES = {
    "nvidia": [
        "nvidia",
        "nvidia corporation",
        "nvda",
    ],
    "microsoft": [
        "microsoft",
        "microsoft corporation",
        "msft",
    ],
    "apple": [
        "apple",
        "apple inc",
        "aapl",
    ],
    "alphabet": [
        "alphabet",
        "alphabet inc",
        "google",
        "googl",
        "goog",
    ],
    "amazon": [
        "amazon",
        "amazon.com",
        "amazon.com inc",
        "amzn",
    ],
    "meta": [
        "meta",
        "meta platforms",
        "meta platforms inc",
        "facebook",
        "fb",
    ],
    "tesla": [
        "tesla",
        "tesla inc",
        "tsla",
    ],
    "oracle": [
        "oracle",
        "oracle corporation",
        "orcl",
    ],
    "ibm": [
        "ibm",
        "international business machines",
    ],
}


# ============================================================
# OFFICIAL / AUTHORITATIVE DOMAINS
# ============================================================

OFFICIAL_DOMAINS = {
    "nvidia": [
        "nvidia.com",
        "nvidianews.nvidia.com",
        "investor.nvidia.com",
    ],
    "microsoft": [
        "microsoft.com",
        "news.microsoft.com",
        "investor.microsoft.com",
    ],
    "apple": [
        "apple.com",
        "investor.apple.com",
    ],
    "alphabet": [
        "abc.xyz",
        "google.com",
        "abc.xyz/investor",
    ],
    "amazon": [
        "amazon.com",
        "ir.aboutamazon.com",
    ],
    "meta": [
        "meta.com",
        "investor.atmeta.com",
    ],
    "tesla": [
        "tesla.com",
        "ir.tesla.com",
    ],
    "oracle": [
        "oracle.com",
        "investor.oracle.com",
    ],
    "ibm": [
        "ibm.com",
        "newsroom.ibm.com",
    ],
}


AUTHORITATIVE_DOMAINS = [
    "sec.gov",
    "investor.",
    "ir.",
    "annualreports.com",
]


# ============================================================
# TERMS TO AVOID
# ============================================================

EXCLUDED_TERMS = [
    "price target",
    "stock forecast",
    "share price forecast",
    "analyst estimate",
    "analyst estimates",
    "earnings estimate",
    "earnings estimates",
    "revenue estimate",
    "revenue estimates",
    "profit estimate",
    "profit estimates",
    "consensus estimate",
    "consensus estimates",
    "price prediction",
    "stock prediction",
]


# ============================================================
# FINANCIAL TERMS
# ============================================================

FINANCIAL_KEYWORDS = [
    "revenue",
    "sales",
    "net income",
    "operating income",
    "gross profit",
    "gross margin",
    "operating margin",
    "profit",
    "earnings",
    "eps",
    "cash flow",
    "free cash flow",
    "financial results",
    "annual results",
    "annual report",
    "10-k",
    "10-q",
    "fiscal year",
    "year ended",
    "quarter ended",
]


# ============================================================
# SOURCE QUALITY
# ============================================================

SOURCE_QUALITY_KEYWORDS = {
    "sec.gov": 20,
    "annualreports.com": 10,
    "investor.": 15,
    "ir.": 15,
    "nvidianews.nvidia.com": 20,
    "news.microsoft.com": 20,
    "press release": 8,
    "financial results": 8,
    "annual report": 8,
    "10-k": 8,
}


# ============================================================
# METRIC ALIASES
# ============================================================

METRIC_ALIASES = {
    "revenue": [
        "revenue",
        "total revenue",
        "net revenue",
        "net sales",
        "sales",
    ],

    "profit": [
        "profit",
        "net income",
        "operating income",
        "net profit",
        "earnings",
    ],

    "net income": [
        "net income",
        "net profit",
        "profit",
    ],

    "operating income": [
        "operating income",
        "operating profit",
    ],

    "year-over-year revenue growth": [
        "revenue growth",
        "year-over-year",
        "year over year",
        "year-on-year",
        "year on year",
        "yoy",
        "revenue increased",
        "revenue grew",
        "revenue up",
        "revenue more than doubled",
    ],

    "quarterly revenue growth": [
        "quarterly revenue",
        "quarterly results",
        "quarter",
        "q1",
        "q2",
        "q3",
        "q4",
        "sequential",
        "quarter-over-quarter",
        "quarter over quarter",
        "q/q",
        "revenue growth",
        "revenue increased",
        "revenue up",
    ],

    "compound annual growth rate (cagr)": [
        "cagr",
        "compound annual growth rate",
        "compound annual growth",
        "annualized growth",
        "annualised growth",
        "multi-year growth",
        "five-year growth",
        "three-year growth",
    ],

    "growth": [
        "growth",
        "grew",
        "increased",
        "increase",
        "up",
        "year-over-year",
        "year over year",
        "year-on-year",
        "yoy",
    ],

    "margin": [
        "margin",
        "gross margin",
        "operating margin",
        "profit margin",
    ],

    "eps": [
        "eps",
        "earnings per share",
        "diluted earnings per share",
    ],

    "cash flow": [
        "cash flow",
        "operating cash flow",
        "free cash flow",
    ],

    "market share": [
        "market share",
        "share of market",
    ],

    "capex": [
        "capex",
        "capital expenditure",
        "capital expenditures",
    ],
}


# ============================================================
# NORMALIZATION HELPERS
# ============================================================

def normalize_text(value: Any) -> str:
    """
    Normalize text for matching.
    """
    if value is None:
        return ""

    return re.sub(
        r"\s+",
        " ",
        str(value).lower().strip(),
    )


def normalize_company(company: str) -> str:
    """
    Convert a company name into the canonical key.
    """
    normalized = normalize_text(company)

    for canonical, aliases in COMPANY_ALIASES.items():
        if normalized == canonical:
            return canonical

        if normalized in aliases:
            return canonical

    return normalized


def get_company_aliases(company: str) -> list[str]:
    """
    Return aliases for a company.
    """
    canonical = normalize_company(company)

    return COMPANY_ALIASES.get(
        canonical,
        [canonical],
    )


def contains_term(text: str, term: str) -> bool:
    """
    Word/phrase aware matching.
    """
    text = normalize_text(text)
    term = normalize_text(term)

    if not text or not term:
        return False

    pattern = r"\b" + re.escape(term) + r"\b"

    return bool(re.search(pattern, text))


def extract_years(text: str) -> list[int]:
    """
    Extract years such as 2024, 2025, 2026.
    """
    return [
        int(year)
        for year in re.findall(
            r"\b20\d{2}\b",
            text,
        )
    ]


def normalize_fiscal_year(
    fiscal_year: str | int | None,
) -> str | None:
    """
    Normalize fiscal year input.
    """
    if fiscal_year is None:
        return None

    value = str(fiscal_year).strip()

    match = re.search(
        r"(20\d{2})",
        value,
    )

    if not match:
        return None

    return match.group(1)


# ============================================================
# URL / DOMAIN HELPERS
# ============================================================

def get_domain(url: str) -> str:
    """
    Extract normalized hostname.
    """
    if not url:
        return ""

    try:
        hostname = urlparse(url).netloc.lower()

        if hostname.startswith("www."):
            hostname = hostname[4:]

        return hostname

    except Exception:
        return ""


def is_authoritative_domain(
    url: str,
    company: str,
) -> bool:
    """
    Check whether a URL belongs to an authoritative source.
    """
    domain = get_domain(url)

    if not domain:
        return False

    canonical = normalize_company(company)

    official_domains = OFFICIAL_DOMAINS.get(
        canonical,
        [],
    )

    for official in official_domains:
        official = official.lower()

        if domain == official or domain.endswith(
            "." + official
        ):
            return True

    for authoritative in AUTHORITATIVE_DOMAINS:
        if authoritative.endswith("."):
            if authoritative in domain:
                return True
        elif domain == authoritative or domain.endswith(
            "." + authoritative
        ):
            return True

    return False


def is_official_company_domain(
    url: str,
    company: str,
) -> bool:
    """
    Check if URL is specifically an official company domain.
    """
    domain = get_domain(url)

    canonical = normalize_company(company)

    for official in OFFICIAL_DOMAINS.get(
        canonical,
        [],
    ):
        official = official.lower()

        if domain == official or domain.endswith(
            "." + official
        ):
            return True

    return False


# ============================================================
# COMPANY MATCHING
# ============================================================

def calculate_company_match_score(
    result: dict[str, Any],
    company: str,
) -> int:
    """
    Calculate how strongly the result matches
    the requested company.
    """

    canonical = normalize_company(company)
    aliases = get_company_aliases(company)

    title = normalize_text(
        result.get("title", "")
    )

    url = normalize_text(
        result.get("url", "")
    )

    content = normalize_text(
        result.get("content", "")
    )

    score = 0

    # --------------------------------------------------------
    # Official company domain
    # --------------------------------------------------------

    if is_official_company_domain(
        result.get("url", ""),
        company,
    ):
        score += 30

    # --------------------------------------------------------
    # Title matching
    # --------------------------------------------------------

    if contains_term(title, canonical):
        score += 30

    else:
        for alias in aliases:
            if contains_term(title, alias):
                score += 25
                break

    # --------------------------------------------------------
    # URL matching
    # --------------------------------------------------------

    for alias in aliases:
        if contains_term(url, alias):
            score += 25
            break

    # --------------------------------------------------------
    # Content matching
    # --------------------------------------------------------

    canonical_count = content.count(
        canonical
    )

    if canonical_count >= 3:
        score += 20

    elif canonical_count >= 2:
        score += 15

    elif canonical_count >= 1:
        score += 10

    # --------------------------------------------------------
    # Alias matching
    # --------------------------------------------------------

    alias_found = False

    for alias in aliases:
        if contains_term(content, alias):
            alias_found = True
            break

    if alias_found:
        score += 10

    return score


def matches_requested_company(
    result: dict[str, Any],
    company: str,
) -> bool:
    """
    Determine whether a result belongs to
    the requested company.
    """

    canonical = normalize_company(company)

    title = normalize_text(
        result.get("title", "")
    )

    url = normalize_text(
        result.get("url", "")
    )

    content = normalize_text(
        result.get("content", "")
    )

    domain = get_domain(
        result.get("url", "")
    )

    aliases = get_company_aliases(company)

    # --------------------------------------------------------
    # Official domain = trusted company match
    # --------------------------------------------------------

    if is_official_company_domain(
        result.get("url", ""),
        company,
    ):
        return True

    # --------------------------------------------------------
    # Strong title / URL match
    # --------------------------------------------------------

    if contains_term(title, canonical):
        return True

    if any(
        contains_term(title, alias)
        for alias in aliases
    ):
        return True

    if any(
        contains_term(url, alias)
        for alias in aliases
    ):
        return True

    # --------------------------------------------------------
    # SEC results
    #
    # SEC pages often have accession-number URLs,
    # so the company name may only exist in the content.
    # --------------------------------------------------------

    if domain == "sec.gov" or domain.endswith(
        ".sec.gov"
    ):

        canonical_count = content.count(
            canonical
        )

        alias_found = any(
            contains_term(content, alias)
            for alias in aliases
        )

        if canonical_count >= 1 or alias_found:
            return True

    # --------------------------------------------------------
    # Strong content match
    # --------------------------------------------------------

    if content.count(canonical) >= 2:
        return True

    if any(
        contains_term(content, alias)
        for alias in aliases
    ):
        return True

    # --------------------------------------------------------
    # Final score-based decision
    # --------------------------------------------------------

    score = calculate_company_match_score(
        result,
        company,
    )

    return score >= 25


# ============================================================
# FINANCIAL CONTENT
# ============================================================

def is_financial_content(
    result: dict[str, Any],
) -> bool:
    """
    Determine whether a result contains
    meaningful financial information.
    """

    title = normalize_text(
        result.get("title", "")
    )

    content = normalize_text(
        result.get("content", "")
    )

    combined = f"{title} {content}"

    matches = 0

    for keyword in FINANCIAL_KEYWORDS:
        if contains_term(combined, keyword):
            matches += 1

    return matches >= 1


# ============================================================
# EXCLUDED CONTENT
# ============================================================

def contains_excluded_terms(
    result: dict[str, Any],
) -> bool:
    """
    Reject obvious forecasting/speculation pages.

    IMPORTANT:
    We do NOT reject a financial filing merely because
    its body happens to mention analyst estimates.
    """

    title = normalize_text(
        result.get("title", "")
    )

    content = normalize_text(
        result.get("content", "")
    )

    url = normalize_text(
        result.get("url", "")
    )

    # --------------------------------------------------------
    # Strong rejection if title itself is forecast related
    # --------------------------------------------------------

    for term in EXCLUDED_TERMS:
        if contains_term(title, term):
            return True

    # --------------------------------------------------------
    # Reject obvious forecast URLs
    # --------------------------------------------------------

    forecast_url_terms = [
        "forecast",
        "prediction",
        "price-target",
        "price_target",
        "stock-prediction",
    ]

    if any(
        term in url
        for term in forecast_url_terms
    ):
        return True

    # --------------------------------------------------------
    # Body-only mentions are NOT enough to reject.
    #
    # SEC filings frequently mention estimates,
    # analyst expectations, etc.
    # --------------------------------------------------------

    body_excluded_count = sum(
        1
        for term in EXCLUDED_TERMS
        if contains_term(content, term)
    )

    # Only reject if multiple forecast terms appear
    # AND the source is not authoritative.
    if body_excluded_count >= 2:
        if not is_authoritative_domain(
            result.get("url", ""),
            "",
        ):
            return True

    return False


# ============================================================
# SOURCE QUALITY
# ============================================================

def calculate_source_quality(
    result: dict[str, Any],
) -> int:
    """
    Calculate source authority score.
    """

    title = normalize_text(
        result.get("title", "")
    )

    url = normalize_text(
        result.get("url", "")
    )

    combined = f"{title} {url}"

    score = 0

    for keyword, points in SOURCE_QUALITY_KEYWORDS.items():

        if keyword in combined:
            score += points

    return score


# ============================================================
# OFFICIAL SOURCE BONUS
# ============================================================

def calculate_official_bonus(
    result: dict[str, Any],
    company: str,
) -> int:
    """
    Give official company domains an additional bonus.
    """

    if is_official_company_domain(
        result.get("url", ""),
        company,
    ):
        return 15

    return 0


# ============================================================
# YEAR SCORE
# ============================================================

def calculate_year_score(
    result: dict[str, Any],
    fiscal_year: str | None,
) -> int:
    """
    Score fiscal-year relevance.
    """

    if not fiscal_year:
        return 0

    title = normalize_text(
        result.get("title", "")
    )

    content = normalize_text(
        result.get("content", "")
    )

    url = normalize_text(
        result.get("url", "")
    )

    requested_year = str(fiscal_year)

    combined = (
        f"{title} "
        f"{content} "
        f"{url}"
    )

    # --------------------------------------------------------
    # Strong exact fiscal-year indicators
    # --------------------------------------------------------

    strong_patterns = [
        f"fiscal year {requested_year}",
        f"fy{requested_year}",
        f"fy {requested_year}",
        f"fiscal {requested_year}",
    ]

    if any(
        pattern in combined
        for pattern in strong_patterns
    ):
        return 30

    # --------------------------------------------------------
    # Plain year match
    # --------------------------------------------------------

    if requested_year in combined:
        return 20

    # --------------------------------------------------------
    # Penalize clearly wrong years only when title
    # strongly identifies another year.
    # --------------------------------------------------------

    title_years = extract_years(title)

    if title_years:
        if all(
            year != int(requested_year)
            for year in title_years
        ):
            return -15

    return 0


# ============================================================
# METRIC SCORE
# ============================================================

def calculate_metric_score(
    result: dict[str, Any],
    metric: str,
) -> int:
    """
    Calculate metric relevance.

    This is intentionally flexible because financial
    documents often phrase the same fact differently.
    """

    title = normalize_text(
        result.get("title", "")
    )

    content = normalize_text(
        result.get("content", "")
    )

    combined = f"{title} {content}"

    metric_normalized = normalize_text(
        metric
    )

    terms = METRIC_ALIASES.get(
        metric_normalized,
        [metric_normalized],
    )

    score = 0

    # --------------------------------------------------------
    # Exact metric phrase
    # --------------------------------------------------------

    if contains_term(
        title,
        metric_normalized,
    ):
        score += 20

    elif contains_term(
        content,
        metric_normalized,
    ):
        score += 10

    # --------------------------------------------------------
    # Alias matches
    # --------------------------------------------------------

    alias_matches = 0

    for term in terms:
        if contains_term(
            combined,
            term,
        ):
            alias_matches += 1

    score += min(
        alias_matches * 5,
        20,
    )

    # --------------------------------------------------------
    # Special handling: YoY revenue growth
    # --------------------------------------------------------

    if metric_normalized == "year-over-year revenue growth":

        revenue_present = (
            "revenue" in combined
        )

        growth_present = any(
            phrase in combined
            for phrase in [
                "year-over-year",
                "year over year",
                "year-on-year",
                "year on year",
                "yoy",
                "revenue grew",
                "revenue increased",
                "revenue up",
                "revenue more than doubled",
            ]
        )

        if revenue_present and growth_present:
            score += 15

    # --------------------------------------------------------
    # Special handling: quarterly revenue growth
    # --------------------------------------------------------

    elif metric_normalized == "quarterly revenue growth":

        revenue_present = (
            "revenue" in combined
        )

        quarter_present = any(
            phrase in combined
            for phrase in [
                "quarter",
                "q1",
                "q2",
                "q3",
                "q4",
                "sequential",
                "q/q",
                "quarter-over-quarter",
                "quarter over quarter",
            ]
        )

        growth_present = any(
            phrase in combined
            for phrase in [
                "growth",
                "grew",
                "increased",
                "up",
            ]
        )

        if (
            revenue_present
            and quarter_present
            and growth_present
        ):
            score += 20

    # --------------------------------------------------------
    # Special handling: CAGR
    # --------------------------------------------------------

    elif (
        metric_normalized
        == "compound annual growth rate (cagr)"
    ):

        if (
            "cagr" in combined
            or "compound annual growth rate" in combined
            or "compound annual growth" in combined
        ):
            score += 20

    return min(score, 50)


# ============================================================
# EVIDENCE BONUS
# ============================================================

def calculate_evidence_bonus(
    result: dict[str, Any],
    metric: str,
) -> int:
    """
    Reward results that contain actual numerical evidence.
    """

    content = normalize_text(
        result.get("content", "")
    )

    metric_normalized = normalize_text(
        metric
    )

    bonus = 0

    # --------------------------------------------------------
    # Currency / financial numbers
    # --------------------------------------------------------

    has_money = bool(
        re.search(
            r"[$€£]\s?\d[\d,.]*",
            content,
        )
    )

    has_percentage = bool(
        re.search(
            r"\d+(?:\.\d+)?\s?%",
            content,
        )
    )

    if has_money:
        bonus += 5

    if has_percentage:
        bonus += 5

    # --------------------------------------------------------
    # Revenue + percentage = strong growth evidence
    # --------------------------------------------------------

    if (
        "revenue" in content
        and has_percentage
    ):
        bonus += 5

    # --------------------------------------------------------
    # YoY-specific evidence
    # --------------------------------------------------------

    if metric_normalized == "year-over-year revenue growth":

        if (
            "revenue" in content
            and (
                "year-over-year" in content
                or "year over year" in content
                or "year-on-year" in content
                or "year on year" in content
                or "yoy" in content
                or "revenue up" in content
                or "revenue grew" in content
                or "revenue increased" in content
            )
            and has_percentage
        ):
            bonus += 10

    # --------------------------------------------------------
    # Quarterly evidence
    # --------------------------------------------------------

    if metric_normalized == "quarterly revenue growth":

        if (
            "revenue" in content
            and "quarter" in content
            and has_percentage
        ):
            bonus += 10

    return min(bonus, 20)


# ============================================================
# FINAL RELEVANCE SCORE
# ============================================================

def calculate_relevance_score(
    result: dict[str, Any],
    company: str,
    metric: str,
    fiscal_year: str | None,
) -> dict[str, Any]:
    """
    Calculate all relevance components.
    """

    company_score = calculate_company_match_score(
        result,
        company,
    )

    year_score = calculate_year_score(
        result,
        fiscal_year,
    )

    metric_score = calculate_metric_score(
        result,
        metric,
    )

    source_quality = calculate_source_quality(
        result,
    )

    official_bonus = calculate_official_bonus(
        result,
        company,
    )

    evidence_bonus = calculate_evidence_bonus(
        result,
        metric,
    )

    financial_bonus = (
        10
        if is_financial_content(result)
        else 0
    )

    total_score = (
        company_score
        + year_score
        + metric_score
        + source_quality
        + official_bonus
        + evidence_bonus
        + financial_bonus
    )

    enriched = dict(result)

    enriched["relevance_score"] = total_score
    enriched["company_match"] = company_score
    enriched["year_score"] = year_score
    enriched["metric_score"] = metric_score
    enriched["source_quality"] = source_quality
    enriched["official_bonus"] = official_bonus
    enriched["evidence_bonus"] = evidence_bonus

    return enriched


# ============================================================
# FILTER FINANCIAL RESULTS
# ============================================================

def filter_financial_results(
    results: list[dict[str, Any]],
    company: str,
    metric: str,
    fiscal_year: str | None = None,
) -> list[dict[str, Any]]:
    """
    Filter and rank financial search results.

    Important design principle:
    Do not throw away valid financial evidence simply
    because the page does not contain the exact phrase
    used in the query.
    """

    filtered: list[dict[str, Any]] = []

    for result in results:

        # ----------------------------------------------------
        # Company check
        # ----------------------------------------------------

        if not matches_requested_company(
            result,
            company,
        ):
            continue

        # ----------------------------------------------------
        # Forecast/speculation check
        # ----------------------------------------------------

        if contains_excluded_terms(result):
            continue

        # ----------------------------------------------------
        # Financial relevance
        #
        # Official/SEC sources can survive even if the
        # Tavily snippet is sparse.
        # ----------------------------------------------------

        financial = is_financial_content(
            result
        )

        authoritative = is_authoritative_domain(
            result.get("url", ""),
            company,
        )

        metric_score = calculate_metric_score(
            result,
            metric,
        )

        if not financial:
            if not authoritative and metric_score < 10:
                continue

        # ----------------------------------------------------
        # Calculate ranking score
        # ----------------------------------------------------

        enriched = calculate_relevance_score(
            result=result,
            company=company,
            metric=metric,
            fiscal_year=fiscal_year,
        )

        # ----------------------------------------------------
        # Minimum quality threshold
        #
        # Company match itself is already meaningful.
        # We intentionally keep this threshold moderate.
        # ----------------------------------------------------

        if enriched["company_match"] < 20:
            continue

        filtered.append(
            enriched
        )

    # --------------------------------------------------------
    # Sort strongest evidence first
    # --------------------------------------------------------

    filtered.sort(
        key=lambda item: item.get(
            "relevance_score",
            0,
        ),
        reverse=True,
    )

    return filtered


# ============================================================
# DEDUPLICATION
# ============================================================

def deduplicate_results(
    results: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Remove duplicate URLs/titles.
    """

    seen_urls: set[str] = set()
    seen_titles: set[str] = set()

    unique_results: list[dict[str, Any]] = []

    for result in results:

        url = normalize_text(
            result.get("url", "")
        ).rstrip("/")

        title = normalize_text(
            result.get("title", "")
        )

        if url and url in seen_urls:
            continue

        if title and title in seen_titles:
            continue

        if url:
            seen_urls.add(url)

        if title:
            seen_titles.add(title)

        unique_results.append(
            result
        )

    return unique_results


# ============================================================
# TAVILY SEARCH
# ============================================================

def tavily_web_search(
    query: str,
    max_results: int = MAX_RESULTS,
) -> list[dict[str, Any]]:
    """
    Execute Tavily search.
    """

    tavily_api_key = os.getenv(
        "TAVILY_API_KEY"
    )

    if not tavily_api_key:
        raise ValueError(
            "TAVILY_API_KEY is not configured."
        )

    search = TavilySearch(
        max_results=max_results,
        topic="general",
        tavily_api_key=tavily_api_key,
    )

    response = search.invoke(
        {
            "query": query,
        }
    )

    # --------------------------------------------------------
    # Tavily may return a dict or JSON string depending
    # on package/version.
    # --------------------------------------------------------

    if isinstance(response, str):

        try:
            response = json.loads(
                response
            )

        except json.JSONDecodeError:
            return []

    if not isinstance(response, dict):
        return []

    results = response.get(
        "results",
        [],
    )

    if not isinstance(results, list):
        return []

    return results


# ============================================================
# METRIC QUERY TERMS
# ============================================================

def get_metric_query(metric: str) -> str:
    """
    Convert the analyzer's metric into a focused search phrase.

    This is NOT query expansion.
    We generate one focused query for the requested metric.
    """

    metric_normalized = normalize_text(
        metric
    )

    if metric_normalized == "revenue":
        return "revenue"

    if metric_normalized == "year-over-year revenue growth":
        return "revenue growth year-over-year"

    if metric_normalized == "quarterly revenue growth":
        return "quarterly revenue growth"

    if (
        metric_normalized
        == "compound annual growth rate (cagr)"
    ):
        return "CAGR revenue growth"

    if metric_normalized == "profit":
        return "net income operating income profit"

    if metric_normalized == "net income":
        return "net income"

    if metric_normalized == "operating income":
        return "operating income"

    if metric_normalized == "margin":
        return "gross margin operating margin"

    if metric_normalized == "eps":
        return "EPS earnings per share"

    if metric_normalized == "cash flow":
        return "cash flow free cash flow"

    if metric_normalized == "market share":
        return "market share"

    if metric_normalized == "capex":
        return "capital expenditure capex"

    if metric_normalized == "growth":
        return "revenue growth"

    return metric


# ============================================================
# BUILD FINANCIAL QUERY
# ============================================================

def build_financial_query(
    company: str,
    metric: str,
    fiscal_year: str | None = None,
) -> str:
    """
    Build a focused financial search query.

    IMPORTANT:
    The previous implementation used several large OR groups.
    That made the query unnecessarily restrictive.

    This version uses:
        company
        metric
        fiscal period
        authoritative-source hints

    No hypothetical questions.
    No query expansion.
    No information compression.
    """

    canonical = normalize_company(
        company
    )

    aliases = get_company_aliases(
        company
    )

    # Prefer the normal company name.
    display_company = aliases[0]

    metric_query = get_metric_query(
        metric
    )

    query_parts = [
        f'"{display_company}"',
        metric_query,
    ]

    # --------------------------------------------------------
    # Fiscal year
    # --------------------------------------------------------

    if fiscal_year:

        query_parts.append(
            f'"fiscal year {fiscal_year}"'
        )

        query_parts.append(
            f'FY{fiscal_year}'
        )

    # --------------------------------------------------------
    # Source intent
    #
    # These are plain terms, NOT a giant OR group.
    # --------------------------------------------------------

    query_parts.extend(
        [
            "financial results",
            "annual report",
            "10-K",
            "investor relations",
        ]
    )

    return " ".join(
        query_parts
    )


# ============================================================
# SEARCH ONE FINANCIAL METRIC
# ============================================================

def search_financial_metric(
    company: str,
    metric: str,
    fiscal_year: str | None = None,
    max_results: int = MAX_RESULTS,
) -> dict[str, Any]:
    """
    Search for one financial metric.
    """

    normalized_year = normalize_fiscal_year(
        fiscal_year
    )

    query = build_financial_query(
        company=company,
        metric=metric,
        fiscal_year=normalized_year,
    )

    print(
        "\nFinancial search query:",
        query,
    )

    # --------------------------------------------------------
    # Tavily
    # --------------------------------------------------------

    raw_results = tavily_web_search(
        query=query,
        max_results=max_results,
    )

    print(
        "Raw Tavily results:",
        len(raw_results),
    )

    # --------------------------------------------------------
    # Filtering
    # --------------------------------------------------------

    filtered_results = filter_financial_results(
        results=raw_results,
        company=company,
        metric=metric,
        fiscal_year=normalized_year,
    )

    print(
        "After relevance filtering:",
        len(filtered_results),
    )

    # --------------------------------------------------------
    # Deduplicate
    # --------------------------------------------------------

    unique_results = deduplicate_results(
        filtered_results
    )

    print(
        "After deduplication:",
        len(unique_results),
    )

    # --------------------------------------------------------
    # Limit
    # --------------------------------------------------------

    final_results = unique_results[
        :max_results
    ]

    return {
        "company": company,
        "metric": metric,
        "fiscal_year": normalized_year,
        "query": query,
        "results": final_results,
    }


# ============================================================
# SEARCH ALL FINANCIAL METRICS
# ============================================================

def search_company_financials(
    company: str,
    metrics: list[str],
    fiscal_year: str | None = None,
    max_results: int = MAX_RESULTS,
) -> dict[str, Any]:
    """
    Search multiple financial metrics for one company.
    """

    normalized_year = normalize_fiscal_year(
        fiscal_year
    )

    results_by_metric: dict[str, Any] = {}

    print(
        "\n"
        + "=" * 60
    )

    print(
        f"FINANCIAL SEARCH: {company}"
    )

    print(
        "=" * 60
    )

    for metric in metrics:

        print(
            "\n"
            + "-" * 60
        )

        print(
            f"METRIC: {metric}"
        )

        print(
            "-" * 60
        )

        response = search_financial_metric(
            company=company,
            metric=metric,
            fiscal_year=normalized_year,
            max_results=max_results,
        )

        results_by_metric[
            metric
        ] = response

        print(
            f"\nQuery: {response['query']}"
        )

        print(
            f"Results found: "
            f"{len(response['results'])}"
        )

    return {
        "company": company,
        "fiscal_year": normalized_year,
        "metrics": results_by_metric,
    }


# ============================================================
# FORMAT RESULTS
# ============================================================

def format_financial_results(
    response: dict[str, Any],
) -> str:
    """
    Convert financial search results into readable text.
    """

    company = response.get(
        "company",
        "",
    )

    metric = response.get(
        "metric",
        "",
    )

    fiscal_year = response.get(
        "fiscal_year",
        "",
    )

    query = response.get(
        "query",
        "",
    )

    results = response.get(
        "results",
        [],
    )

    lines: list[str] = []

    lines.append(
        "=" * 70
    )

    lines.append(
        f"Company: {company}"
    )

    lines.append(
        f"Metric: {metric}"
    )

    if fiscal_year:
        lines.append(
            f"Fiscal Year: {fiscal_year}"
        )

    lines.append(
        f"Query: {query}"
    )

    lines.append(
        "=" * 70
    )

    if not results:

        lines.append(
            "No reliable financial results found."
        )

        return "\n".join(lines)

    for index, result in enumerate(
        results,
        start=1,
    ):

        title = result.get(
            "title",
            "Untitled",
        )

        url = result.get(
            "url",
            "",
        )

        content = result.get(
            "content",
            "",
        )

        relevance_score = result.get(
            "relevance_score",
            0,
        )

        company_match = result.get(
            "company_match",
            0,
        )

        year_score = result.get(
            "year_score",
            0,
        )

        metric_score = result.get(
            "metric_score",
            0,
        )

        source_quality = result.get(
            "source_quality",
            0,
        )

        evidence_bonus = result.get(
            "evidence_bonus",
            0,
        )

        lines.append(
            f"\n[{index}] {title}"
        )

        lines.append(
            f"URL: {url}"
        )

        lines.append(
            f"Relevance Score: {relevance_score}"
        )

        lines.append(
            f"Company Match: {company_match}"
        )

        lines.append(
            f"Year Score: {year_score}"
        )

        lines.append(
            f"Metric Score: {metric_score}"
        )

        lines.append(
            f"Source Quality: {source_quality}"
        )

        lines.append(
            f"Evidence Bonus: {evidence_bonus}"
        )

        lines.append(
            f"Content: {content}"
        )

    return "\n".join(lines)


# ============================================================
# STANDALONE TEST
# ============================================================

if __name__ == "__main__":

    print(
        "\n"
        + "#" * 70
    )

    print(
        "# FINANCIAL WEB SEARCH TEST"
    )

    print(
        "#" * 70
    )

    company = "NVIDIA"

    fiscal_year = "2025"

    metrics = [
        "revenue",
        "year-over-year revenue growth",
        "quarterly revenue growth",
        "compound annual growth rate (cagr)",
    ]

    response = search_company_financials(
        company=company,
        metrics=metrics,
        fiscal_year=fiscal_year,
    )

    print(
        "\n"
        + "#" * 70
    )

    print(
        "# FINAL TEST OUTPUT"
    )

    print(
        "#" * 70
    )

    for metric, metric_response in response[
        "metrics"
    ].items():

        print(
            "\n"
            + "-" * 70
        )

        print(
            f"METRIC: {metric}"
        )

        print(
            "-" * 70
        )

        print(
            format_financial_results(
                metric_response
            )
        )