from typing import TypedDict, List, Optional

import json
import re

from src.config.azure_models import get_chat_model


# ============================================================
# QUERY ANALYSIS RESULT
# ============================================================

class QueryAnalysis(TypedDict):
    companies: List[str]
    intent: str
    context: str
    metrics: List[str]
    fiscal_year: Optional[str]
    research_targets: List[str]


# ============================================================
# INITIALIZE LLM
# ============================================================

llm = get_chat_model()


# ============================================================
# QUERY ANALYZER PROMPT
# ============================================================

QUERY_ANALYZER_PROMPT = """
You are a Query Analyzer for a Market and Competitive Intelligence System.

Your job is to analyze the user's question and convert it into structured
information that downstream research agents can use.

Extract the following information.

1. companies

Identify every company explicitly mentioned in the question.

Return the commonly recognized company name.

Examples:

MSFT -> Microsoft
Microsoft -> Microsoft
NVIDIA -> NVIDIA
NVDA -> NVIDIA
Apple -> Apple
Amazon -> Amazon
Google -> Google
Alphabet -> Alphabet
Meta -> Meta
Tesla -> Tesla
Oracle -> Oracle
IBM -> IBM

Do not invent companies.

2. intent

Choose exactly ONE of:

- financial_analysis
- comparison
- business_analysis
- strategy_analysis
- general_information

Use "comparison" when the user:

- compares companies
- asks which company is better
- asks which company is more profitable
- asks about differences between companies
- asks to rank multiple companies

3. context

Identify the main business context of the question.

Examples:

- financial performance
- revenue
- profitability
- cloud business
- AI strategy
- products
- market share
- employees
- business strategy
- financial performance and growth

Keep the context concise.

4. metrics

Extract specific metrics explicitly mentioned OR clearly required
to answer the question.

Examples:

- revenue
- net income
- profit
- operating income
- gross margin
- operating margin
- revenue growth
- profit growth
- market share
- cloud revenue
- cloud revenue growth
- cloud operating income

IMPORTANT:

If the user asks about a specific business area, include the
metrics normally needed to analyze that business area.

For example:

Question:
"Compare Microsoft's cloud business with Amazon's cloud business."

Metrics should include:

[
    "cloud revenue",
    "cloud revenue growth",
    "cloud operating income",
    "cloud market share"
]

Another example:

Question:
"Compare Microsoft's AI strategy with NVIDIA's AI strategy."

Possible metrics/research areas:

[
    "AI products",
    "AI investments",
    "AI revenue",
    "AI market position"
]

Do not invent numerical values.

5. fiscal_year

Extract the fiscal year if explicitly mentioned.

Examples:

"FY2025" -> "FY2025"

"fiscal year 2025" -> "FY2025"

"2025" -> "FY2025" when clearly referring to a financial year

If no fiscal year is mentioned, return null.

6. research_targets

Create a concise list of specific information that downstream
research agents should search for.

These are research categories, NOT hypothetical questions.

Examples:

Question:
"Compare Microsoft's cloud business with Amazon's cloud business."

Return:

[
    "cloud revenue",
    "cloud revenue growth",
    "cloud operating income",
    "cloud market share"
]

Question:
"What was Microsoft's revenue in FY2025?"

Return:

[
    "revenue"
]

Question:
"Compare Microsoft and NVIDIA revenue growth in FY2025."

Return:

[
    "revenue growth"
]

Question:
"Compare Microsoft's AI strategy with NVIDIA's AI strategy."

Return:

[
    "AI products",
    "AI investments",
    "AI revenue",
    "AI market position"
]

Keep research targets directly related to the user's question.

IMPORTANT RULES:

- Return ONLY valid JSON.
- Do not include markdown.
- Do not explain your answer.
- Do not invent missing information.
- Identify ALL companies mentioned.
- Preserve the user's actual intent.
- If the question compares multiple companies, intent MUST be "comparison".
- If no fiscal year is mentioned, fiscal_year MUST be null.
- Do not invent numerical values.
- research_targets must describe information to research, not answers.
- Do not create hypothetical questions.
- Do not perform query expansion.
- Keep research_targets concise and relevant.

Return exactly this JSON structure:

{
    "companies": [],
    "intent": "",
    "context": "",
    "metrics": [],
    "fiscal_year": null,
    "research_targets": []
}

USER QUESTION:

{question}
"""


# ============================================================
# EXTRACT JSON FROM LLM RESPONSE
# ============================================================

def extract_json(text: str) -> dict:
    """
    Extract a JSON object from the LLM response.

    Handles cases where the LLM accidentally returns
    JSON inside markdown code fences.
    """

    if not text:
        raise ValueError("LLM returned an empty response.")

    text = text.strip()

    # --------------------------------------------------------
    # Remove markdown code fences
    # --------------------------------------------------------

    text = re.sub(
        r"```json\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )

    text = re.sub(
        r"```\s*",
        "",
        text,
    )

    # --------------------------------------------------------
    # Find JSON object
    # --------------------------------------------------------

    start = text.find("{")
    end = text.rfind("}")

    if start == -1 or end == -1 or end <= start:
        raise ValueError(
            "Could not find a JSON object in the LLM response.\n"
            f"LLM response:\n{text}"
        )

    json_text = text[start:end + 1]

    # --------------------------------------------------------
    # Parse JSON
    # --------------------------------------------------------

    try:
        return json.loads(json_text)

    except json.JSONDecodeError as e:
        raise ValueError(
            "LLM returned invalid JSON.\n"
            f"JSON response:\n{json_text}\n"
            f"Parsing error: {e}"
        ) from e


# ============================================================
# NORMALIZE AND VALIDATE ANALYSIS
# ============================================================

def normalize_analysis(data: dict) -> QueryAnalysis:
    """
    Normalize the LLM-generated query analysis.

    This ensures downstream LangGraph nodes receive
    predictable data types.
    """

    companies = data.get("companies", [])

    intent = data.get(
        "intent",
        "general_information",
    )

    context = data.get(
        "context",
        "",
    )

    metrics = data.get(
        "metrics",
        [],
    )

    fiscal_year = data.get(
        "fiscal_year",
    )

    research_targets = data.get(
        "research_targets",
        [],
    )

    # ========================================================
    # COMPANIES
    # ========================================================

    if not isinstance(companies, list):
        if companies:
            companies = [companies]
        else:
            companies = []

    companies = [
        str(company).strip()
        for company in companies
        if str(company).strip()
    ]

    # Remove duplicates while preserving order
    companies = list(dict.fromkeys(companies))

    # ========================================================
    # METRICS
    # ========================================================

    if not isinstance(metrics, list):
        if metrics:
            metrics = [metrics]
        else:
            metrics = []

    metrics = [
        str(metric).strip().lower()
        for metric in metrics
        if str(metric).strip()
    ]

    # Remove duplicates while preserving order
    metrics = list(dict.fromkeys(metrics))

    # ========================================================
    # CONTEXT
    # ========================================================

    context = str(context).strip()

    # ========================================================
    # FISCAL YEAR
    # ========================================================

    if fiscal_year:
        fiscal_year = str(
            fiscal_year
        ).strip().upper()
    else:
        fiscal_year = None

    # ========================================================
    # RESEARCH TARGETS
    # ========================================================

    if not isinstance(research_targets, list):
        if research_targets:
            research_targets = [research_targets]
        else:
            research_targets = []

    research_targets = [
        str(target).strip().lower()
        for target in research_targets
        if str(target).strip()
    ]

    # Remove duplicates while preserving order
    research_targets = list(
        dict.fromkeys(research_targets)
    )

    # ========================================================
    # INTENT VALIDATION
    # ========================================================

    allowed_intents = {
        "financial_analysis",
        "comparison",
        "business_analysis",
        "strategy_analysis",
        "general_information",
    }

    if intent not in allowed_intents:
        intent = "general_information"

    # ========================================================
    # RETURN NORMALIZED RESULT
    # ========================================================

    return {
        "companies": companies,
        "intent": intent,
        "context": context,
        "metrics": metrics,
        "fiscal_year": fiscal_year,
        "research_targets": research_targets,
    }


# ============================================================
# MAIN QUERY ANALYZER
# ============================================================

def analyze_query(
    question: str,
) -> QueryAnalysis:
    """
    Analyze a user's natural-language market intelligence
    question.

    Returns structured information containing:

    - companies
    - intent
    - context
    - metrics
    - fiscal year
    - research targets
    """

    if not question or not question.strip():
        raise ValueError(
            "Question cannot be empty."
        )

    # --------------------------------------------------------
    # Create prompt
    #
    # IMPORTANT:
    # Do NOT use .format() here because the prompt contains
    # a literal JSON object with { } braces.
    # --------------------------------------------------------

    prompt = QUERY_ANALYZER_PROMPT.replace(
        "{question}",
        question.strip(),
    )

    # --------------------------------------------------------
    # Call Azure OpenAI
    # --------------------------------------------------------

    response = llm.invoke(prompt)

    # --------------------------------------------------------
    # Extract response content
    # --------------------------------------------------------

    content = response.content

    if not isinstance(content, str):
        content = str(content)

    # --------------------------------------------------------
    # Convert JSON text to Python dictionary
    # --------------------------------------------------------

    data = extract_json(content)

    # --------------------------------------------------------
    # Normalize result
    # --------------------------------------------------------

    analysis = normalize_analysis(data)

    return analysis


# ============================================================
# PRINT ANALYSIS
# ============================================================

def print_analysis(
    question: str,
) -> None:
    """
    Run query analysis and print the result
    in a readable format.
    """

    print()
    print("=" * 70)
    print("QUERY ANALYZER TEST")
    print("=" * 70)

    print()
    print("User Question:")
    print(question)

    try:

        analysis = analyze_query(question)

        print()
        print("-" * 70)
        print("QUERY ANALYSIS")
        print("-" * 70)

        print(
            f"Companies        : "
            f"{analysis['companies']}"
        )

        print(
            f"Intent           : "
            f"{analysis['intent']}"
        )

        print(
            f"Context          : "
            f"{analysis['context']}"
        )

        print(
            f"Metrics          : "
            f"{analysis['metrics']}"
        )

        print(
            f"Fiscal Year      : "
            f"{analysis['fiscal_year']}"
        )

        print(
            f"Research Targets : "
            f"{analysis['research_targets']}"
        )

        print()
        print("-" * 70)
        print("JSON OUTPUT")
        print("-" * 70)

        print(
            json.dumps(
                analysis,
                indent=4,
            )
        )

    except Exception as e:

        print()
        print("QUERY ANALYZER ERROR")

        print(
            f"{type(e).__name__}: {e}"
        )


# ============================================================
# STANDALONE TESTS
# ============================================================

if __name__ == "__main__":

    test_questions = [

        # ----------------------------------------------------
        # TEST 1
        # ----------------------------------------------------

        "What was Microsoft's revenue in FY2025?",

        # ----------------------------------------------------
        # TEST 2
        # ----------------------------------------------------

        "What was NVIDIA's net income in fiscal year 2025?",

        # ----------------------------------------------------
        # TEST 3
        # ----------------------------------------------------

        "Compare Microsoft and NVIDIA revenue growth in FY2025.",

        # ----------------------------------------------------
        # TEST 4
        # ----------------------------------------------------

        "Which is more profitable, Microsoft, Apple, or Amazon?",

        # ----------------------------------------------------
        # TEST 5
        # ----------------------------------------------------

        "Compare Microsoft's cloud business with Amazon's cloud business.",

        # ----------------------------------------------------
        # TEST 6
        # ----------------------------------------------------

        "Compare Microsoft's AI strategy with NVIDIA's AI strategy.",

        # ----------------------------------------------------
        # TEST 7
        # ----------------------------------------------------

        "What are Microsoft's major business segments?",
    ]

    # --------------------------------------------------------
    # Run all tests
    # --------------------------------------------------------

    for question in test_questions:
        print_analysis(question)