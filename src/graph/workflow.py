from pathlib import Path
from typing import Any, TypedDict

import mlflow
from langgraph.graph import StateGraph, START, END

from src.graph.query_analyzer import analyze_query
from src.graph.analysis_agent import analysis_agent
from src.graph.report_drafter import draft_report

from src.tools.rag_tool import rag_tool
from src.tools.web_search_tool import (
    search_company_financials,
    format_financial_results,
)
from src.tools.general_web_search_tool import search_general_web

from src.guardrails.guardrails import (
    input_guardrail_node,
    output_guardrail_node,
)

from src.memory.memory_manager import (
    search_memory,
    format_memory_for_llm,
    save_state_to_memory,
)


# ============================================================
# MLFLOW CONFIGURATION
# ============================================================

MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"

MLFLOW_EXPERIMENT_NAME = (
    "MarketAnalysis_MultiAgentSystem"
)

mlflow.set_tracking_uri(
    MLFLOW_TRACKING_URI
)

mlflow.set_experiment(
    MLFLOW_EXPERIMENT_NAME
)

# Automatically trace LangChain / LangGraph operations
mlflow.langchain.autolog()


# ============================================================
# CONFIGURATION
# ============================================================

REPORT_DIRECTORY = Path(
    "data/reports"
)

RAG_ENABLED_COMPANIES = {
    "Microsoft",
}

MEMORY_RESULTS = 5


# ============================================================
# STATE
# ============================================================

class MarketAnalysisState(TypedDict, total=False):

    # --------------------------------------------------------
    # User question
    # --------------------------------------------------------

    question: str

    # --------------------------------------------------------
    # Query analysis
    # --------------------------------------------------------

    companies: list[str]

    intent: str

    context: str

    metrics: list[str]

    fiscal_year: str

    # --------------------------------------------------------
    # Memory
    # --------------------------------------------------------

    memory_context: str

    memory_results: list[dict[str, Any]]

    memory_id: int | None

    # --------------------------------------------------------
    # Research
    # --------------------------------------------------------

    company_research: dict[str, Any]

    research_context: str

    # --------------------------------------------------------
    # Analysis
    # --------------------------------------------------------

    analysis: str

    # --------------------------------------------------------
    # Final report
    # --------------------------------------------------------

    final_report: str

    report_path: str

    # --------------------------------------------------------
    # Guardrails
    # --------------------------------------------------------

    input_allowed: bool

    output_allowed: bool

    input_guardrail_reason: str

    output_guardrail_reason: str

    guardrail_message: str


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def format_rag_documents(
    documents: Any,
) -> str:
    """
    Convert RAG documents into readable text.
    """

    if not documents:
        return ""

    formatted = []

    for index, document in enumerate(
        documents,
        start=1,
    ):

        try:

            page_content = getattr(
                document,
                "page_content",
                str(document),
            )

            metadata = getattr(
                document,
                "metadata",
                {},
            )

            source = metadata.get(
                "source",
                "Unknown source",
            )

            page = metadata.get(
                "page",
                "",
            )

            header = (
                f"[RAG Document {index}] "
                f"Source: {source}"
            )

            if page != "":
                header += (
                    f" | Page: {page}"
                )

            formatted.append(
                f"{header}\n"
                f"{page_content}"
            )

        except Exception as exc:

            formatted.append(
                f"[RAG Document {index}]\n"
                f"{document}\n"
                f"Formatting error: {exc}"
            )

    return "\n\n".join(
        formatted
    )


def format_general_web_results(
    results: Any,
) -> str:
    """
    Convert general web search results
    into readable text.
    """

    if not results:
        return ""

    if isinstance(results, str):
        return results

    formatted = []

    for index, result in enumerate(
        results,
        start=1,
    ):

        if isinstance(result, dict):

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
                result.get(
                    "snippet",
                    "",
                ),
            )

            formatted.append(
                f"[Web Result {index}]\n"
                f"Title: {title}\n"
                f"URL: {url}\n"
                f"Content: {content}"
            )

        else:

            formatted.append(
                f"[Web Result {index}]\n"
                f"{result}"
            )

    return "\n\n".join(
        formatted
    )


def is_financial_query(
    state: MarketAnalysisState,
) -> bool:
    """
    Determine whether financial web search
    should be used.
    """

    intent = (
        state.get(
            "intent",
            "",
        )
        or ""
    ).lower()

    metrics = state.get(
        "metrics",
        [],
    )

    financial_keywords = {
        "revenue",
        "revenue growth",
        "profit",
        "profit growth",
        "net income",
        "earnings",
        "eps",
        "margin",
        "gross margin",
        "operating margin",
        "cash flow",
        "free cash flow",
        "fcf",
        "ebitda",
        "market cap",
        "valuation",
        "cagr",
        "yoy",
        "year over year",
        "financial",
    }

    if any(
        keyword in intent
        for keyword in financial_keywords
    ):
        return True

    for metric in metrics:

        metric_lower = (
            str(metric).lower()
        )

        if any(
            keyword in metric_lower
            for keyword in financial_keywords
        ):
            return True

    return False


def build_rag_query(
    company: str,
    question: str,
    metrics: list[str],
    fiscal_year: str,
) -> str:
    """
    Build a focused query for
    the internal RAG system.
    """

    parts = [
        company,
        question,
    ]

    if metrics:

        parts.append(
            "Metrics: "
            + ", ".join(metrics)
        )

    if fiscal_year:

        parts.append(
            f"Fiscal year: {fiscal_year}"
        )

    return " | ".join(
        parts
    )


def build_general_web_query(
    company: str,
    question: str,
    metrics: list[str],
    fiscal_year: str,
) -> str:
    """
    Build a general web-search query.
    """

    parts = [
        company,
        question,
    ]

    if metrics:

        parts.append(
            " ".join(metrics)
        )

    if fiscal_year:

        parts.append(
            fiscal_year
        )

    return " ".join(
        str(part)
        for part in parts
        if part
    )


# ============================================================
# COMPANY RESEARCH
# ============================================================

def research_single_company(
    company: str,
    state: MarketAnalysisState,
) -> dict[str, Any]:
    """
    Research one company using the
    appropriate tools.

    Microsoft:
        Internal RAG + financial search + general web

    Other companies:
        Financial search + general web
    """

    question = state.get(
        "question",
        "",
    )

    metrics = state.get(
        "metrics",
        [],
    )

    fiscal_year = state.get(
        "fiscal_year",
        "",
    )

    rag_documents = []

    financial_results = []

    general_web_results = []

    rag_context = ""

    financial_context = ""

    general_web_context = ""

    # ========================================================
    # RAG SEARCH
    # ========================================================

    rag_enabled = (
        company in RAG_ENABLED_COMPANIES
    )

    if rag_enabled:

        rag_query = build_rag_query(
            company=company,
            question=question,
            metrics=metrics,
            fiscal_year=fiscal_year,
        )

        print()

        print(
            f"[RAG] Searching internal "
            f"documents for {company}"
        )

        try:

            rag_documents = rag_tool(
                query=rag_query,
                k=5,
                company=company,
            )

            rag_context = (
                format_rag_documents(
                    rag_documents
                )
            )

            print(
                f"[RAG] Documents returned: "
                f"{len(rag_documents)}"
            )

        except Exception as exc:

            print(
                f"[RAG ERROR] "
                f"{company}: {exc}"
            )

            rag_documents = []

            rag_context = ""

    else:

        print(
            f"[RAG] Disabled for {company}"
        )

    # ========================================================
    # FINANCIAL WEB SEARCH
    # ========================================================

    financial_query_required = (
        is_financial_query(state)
    )

    if financial_query_required:

        print()

        print(
            f"[FINANCIAL WEB] Searching "
            f"financial information for {company}"
        )

        try:

            financial_results = (
                search_company_financials(
                    company=company,
                    metrics=metrics,
                    fiscal_year=fiscal_year,
                )
            )

            print(
                f"[FINANCIAL WEB] Results: "
                f"{len(financial_results)}"
            )

            financial_context = (
                format_financial_results(
                    financial_results
                )
            )

        except Exception as exc:

            print(
                f"[FINANCIAL WEB ERROR] "
                f"{company}: {exc}"
            )

            financial_results = []

            financial_context = ""

        # ========================================================
    # GENERAL WEB SEARCH
    # ========================================================

    general_web_query = (
        build_general_web_query(
            company=company,
            question=question,
            metrics=metrics,
            fiscal_year=fiscal_year,
        )
    )

    print()

    print(
        f"[GENERAL WEB] Searching for {company}"
    )

    try:

        # search_general_web() returns a dictionary.
        # Extract the actual results list from the "results" key.

        search_response = search_general_web(
            general_web_query
        )

        general_web_results = (
            search_response.get(
                "results",
                [],
            )
        )

        print(
            f"[GENERAL WEB] Results: "
            f"{len(general_web_results)}"
        )

        general_web_context = (
            format_general_web_results(
                general_web_results
            )
        )

    except Exception as exc:

        print(
            f"[GENERAL WEB ERROR] "
            f"{company}: {exc}"
        )

        general_web_results = []

        general_web_context = ""

    # ========================================================
    # COMBINED CONTEXT
    # ========================================================

    context_sections = []

    if rag_context:

        context_sections.append(
            "===== INTERNAL RAG =====\n"
            + rag_context
        )

    if financial_context:

        context_sections.append(
            "===== FINANCIAL WEB SEARCH =====\n"
            + financial_context
        )

    if general_web_context:

        context_sections.append(
            "===== GENERAL WEB SEARCH =====\n"
            + general_web_context
        )

    combined_context = (
        "\n\n".join(
            context_sections
        )
    )

    # ========================================================
    # RETURN
    # ========================================================

    return {
        "company": company,
        "rag_enabled": rag_enabled,
        "rag_documents": rag_documents,
        "rag_context": rag_context,
        "financial_results": financial_results,
        "financial_context": financial_context,
        "general_web_results": general_web_results,
        "general_web_context": general_web_context,
        "combined_context": combined_context,
    }


# ============================================================
# INPUT GUARDRAIL
# ============================================================

def input_guardrail_router(
    state: MarketAnalysisState,
):
    """
    Route based on input guardrail result.
    """

    if state.get(
        "input_allowed",
        True,
    ):

        return "query_analyzer"

    print()

    print(
        "[GUARDRAIL] Input blocked."
    )

    return END


def run_input_guardrail(
    state: MarketAnalysisState,
) -> MarketAnalysisState:
    """
    Execute the input guardrail node.
    """

    try:

        result = input_guardrail_node(
            state
        )

        if result is None:
            return state

        return result

    except TypeError:

        try:

            result = input_guardrail_node(
                question=state.get(
                    "question",
                    "",
                )
            )

            if isinstance(
                result,
                dict,
            ):

                return {
                    **state,
                    **result,
                }

        except Exception as exc:

            print(
                f"[INPUT GUARDRAIL ERROR] "
                f"{exc}"
            )

    except Exception as exc:

        print(
            f"[INPUT GUARDRAIL ERROR] "
            f"{exc}"
        )

    # Fail open if implementation
    # signature differs.

    return {
        **state,
        "input_allowed": True,
    }


# ============================================================
# QUERY ANALYZER NODE
# ============================================================

def query_analyzer_node(
    state: MarketAnalysisState,
) -> MarketAnalysisState:
    """
    Analyze the user's market question.
    """

    print()

    print(
        "=" * 70
    )

    print(
        "QUERY ANALYZER"
    )

    print(
        "=" * 70
    )

    question = state.get(
        "question",
        "",
    )

    print(
        f"Question: {question}"
    )

    try:

        result = analyze_query(
            question
        )

        print()

        print(
            "Query analysis completed."
        )

        print(
            f"Companies: "
            f"{result.get('companies', [])}"
        )

        print(
            f"Intent: "
            f"{result.get('intent', '')}"
        )

        print(
            f"Metrics: "
            f"{result.get('metrics', [])}"
        )

        print(
            f"Fiscal year: "
            f"{result.get('fiscal_year', '')}"
        )

        return {
            **state,

            "companies": result.get(
                "companies",
                [],
            ),

            "intent": result.get(
                "intent",
                "",
            ),

            "context": result.get(
                "context",
                "",
            ),

            "metrics": result.get(
                "metrics",
                [],
            ),

            "fiscal_year": result.get(
                "fiscal_year",
                "",
            ),
        }

    except Exception as exc:

        print(
            f"[QUERY ANALYZER ERROR] "
            f"{exc}"
        )

        return {
            **state,
            "companies": [],
            "intent": "",
            "context": "",
            "metrics": [],
            "fiscal_year": "",
        }


# ============================================================
# MEMORY NODE
# ============================================================

def memory_node(
    state: MarketAnalysisState,
) -> MarketAnalysisState:
    """
    Search previous market-analysis memories.
    """

    print()

    print(
        "=" * 70
    )

    print(
        "MEMORY SEARCH"
    )

    print(
        "=" * 70
    )

    question = state.get(
        "question",
        "",
    )

    companies = state.get(
        "companies",
        [],
    )

    context = state.get(
        "context",
        "",
    )

    query_parts = []

    if question:
        query_parts.append(question)

    if companies:
        query_parts.extend(companies)

    if context:
        query_parts.append(context)

    search_query = " ".join(
        str(part)
        for part in query_parts
        if part
    ).strip()

    if not search_query:

        print(
            "No query available "
            "for memory search."
        )

        return {
            **state,
            "memory_results": [],
            "memory_context": "",
        }

    print(
        f"Memory search query: "
        f"{search_query}"
    )

    try:

        memory_results = search_memory(
            query=search_query,
            limit=MEMORY_RESULTS,
        )

        if not memory_results:

            print(
                "No relevant memories found."
            )

            return {
                **state,
                "memory_results": [],
                "memory_context": "",
            }

        unique_results = []

        seen_ids = set()

        for result in memory_results:

            memory_id = result.get(
                "id"
            )

            if memory_id is not None:

                if memory_id in seen_ids:
                    continue

                seen_ids.add(
                    memory_id
                )

            unique_results.append(
                result
            )

        print(
            f"Memory results: "
            f"{len(unique_results)}"
        )

        memory_context = (
            format_memory_for_llm(
                unique_results
            )
        )

        return {
            **state,
            "memory_results": unique_results,
            "memory_context": memory_context,
        }

    except Exception as exc:

        print(
            f"[MEMORY ERROR] {exc}"
        )

        return {
            **state,
            "memory_results": [],
            "memory_context": "",
        }


# ============================================================
# RESEARCH NODE
# ============================================================

def research_node(
    state: MarketAnalysisState,
) -> MarketAnalysisState:
    """
    Research every company identified
    by the query analyzer.
    """

    print()

    print(
        "=" * 70
    )

    print(
        "RESEARCH AGENT"
    )

    print(
        "=" * 70
    )

    companies = state.get(
        "companies",
        [],
    )

    if not companies:

        print(
            "[RESEARCH] No companies found."
        )

        return {
            **state,
            "company_research": {},
            "research_context": "",
        }

    company_research = {}

    for company in companies:

        print()

        print(
            "-" * 70
        )

        print(
            f"Researching: {company}"
        )

        print(
            "-" * 70
        )

        result = research_single_company(
            company=company,
            state=state,
        )

        company_research[
            company
        ] = result

    research_sections = []

    for company, result in (
        company_research.items()
    ):

        company_context = (
            result.get(
                "combined_context",
                "",
            )
        )

        research_sections.append(
            "\n"
            "##################################################\n"
            f"COMPANY: {company}\n"
            "##################################################\n"
            f"{company_context or 'No usable research evidence found.'}"
        )

    research_context = (
        "\n\n".join(
            research_sections
        )
    )

    print()

    print(
        "Research completed."
    )

    return {
        **state,
        "company_research": company_research,
        "research_context": research_context,
    }


# ============================================================
# ANALYSIS NODE
# ============================================================

def analysis_node(
    state: MarketAnalysisState,
) -> MarketAnalysisState:
    """
    Analyze the collected research evidence.
    """

    print()

    print(
        "=" * 70
    )

    print(
        "ANALYSIS AGENT"
    )

    print(
        "=" * 70
    )

    question = state.get(
        "question",
        "",
    )

    query_analysis = {
        "companies": state.get(
            "companies",
            [],
        ),

        "intent": state.get(
            "intent",
            "",
        ),

        "context": state.get(
            "context",
            "",
        ),

        "metrics": state.get(
            "metrics",
            [],
        ),

        "fiscal_year": state.get(
            "fiscal_year",
            "",
        ),
    }

    research_context = state.get(
        "research_context",
        "",
    )

    memory_context = state.get(
        "memory_context",
        "",
    )

    analysis_context = (
        research_context
    )

    if memory_context:

        analysis_context += (
            "\n\n"
            "##################################################\n"
            "RELEVANT PREVIOUS MEMORY\n"
            "##################################################\n"
            f"{memory_context}"
        )

    try:

        analysis = analysis_agent(
            question=question,
            query_analysis=query_analysis,
            research_context=analysis_context,
        )

        print(
            "Analysis completed."
        )

        return {
            **state,
            "analysis": analysis,
        }

    except Exception as exc:

        print(
            f"[ANALYSIS ERROR] "
            f"{exc}"
        )

        return {
            **state,
            "analysis": (
                "Analysis could not be completed "
                f"because of an error: {exc}"
            ),
        }


# ============================================================
# REPORT DRAFTER NODE
# ============================================================

def report_drafter_node(
    state: MarketAnalysisState,
) -> MarketAnalysisState:
    """
    Draft the final market analysis report.
    """

    print()

    print(
        "=" * 70
    )

    print(
        "REPORT DRAFTER"
    )

    print(
        "=" * 70
    )

    question = state.get(
        "question",
        "",
    )

    analysis = state.get(
        "analysis",
        "",
    )

    research_context = state.get(
        "research_context",
        "",
    )

    try:

        final_report = draft_report(
            question=question,
            analysis=analysis,
            research_context=research_context,
        )

        print(
            "Report drafting completed."
        )

        return {
            **state,
            "final_report": final_report,
        }

    except Exception as exc:

        print(
            f"[REPORT DRAFTER ERROR] "
            f"{exc}"
        )

        return {
            **state,
            "final_report": (
                "Report generation failed.\n\n"
                f"Error: {exc}"
            ),
        }


# ============================================================
# OUTPUT GUARDRAIL
# ============================================================

def run_output_guardrail(
    state: MarketAnalysisState,
) -> MarketAnalysisState:
    """
    Execute the output guardrail node.
    """

    try:

        result = output_guardrail_node(
            state
        )

        if result is None:
            return state

        return result

    except TypeError:

        try:

            result = output_guardrail_node(
                final_report=state.get(
                    "final_report",
                    "",
                )
            )

            if isinstance(
                result,
                dict,
            ):

                return {
                    **state,
                    **result,
                }

        except Exception as exc:

            print(
                f"[OUTPUT GUARDRAIL ERROR] "
                f"{exc}"
            )

    except Exception as exc:

        print(
            f"[OUTPUT GUARDRAIL ERROR] "
            f"{exc}"
        )

    return {
        **state,
        "output_allowed": True,
    }


def output_guardrail_router(
    state: MarketAnalysisState,
):
    """
    Route based on output guardrail result.
    """

    output_allowed = state.get(
        "output_allowed",
        True,
    )

    output_reason = state.get(
        "output_guardrail_reason",
        "",
    )

    guardrail_message = state.get(
        "guardrail_message",
        "",
    )

    print()

    print(
        "=" * 70
    )

    print(
        "OUTPUT GUARDRAIL RESULT"
    )

    print(
        "=" * 70
    )

    print(
        f"Output allowed: "
        f"{output_allowed}"
    )

    print(
        f"Reason: "
        f"{output_reason}"
    )

    print(
        f"Message: "
        f"{guardrail_message}"
    )

    if output_allowed:

        print(
            "Output guardrail passed."
        )

        return "save_report"

    print()

    print(
        "[GUARDRAIL] Output blocked."
    )

    return END


# ============================================================
# SAVE REPORT NODE
# ============================================================

def save_report_node(
    state: MarketAnalysisState,
) -> MarketAnalysisState:
    """
    Save final report to disk.
    """

    print()

    print(
        "=" * 70
    )

    print(
        "SAVE REPORT"
    )

    print(
        "=" * 70
    )

    REPORT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    report_path = (
        REPORT_DIRECTORY
        / "market_analysis_report.txt"
    )

    final_report = state.get(
        "final_report",
        "",
    )

    try:

        report_path.write_text(
            final_report,
            encoding="utf-8",
        )

        print(
            f"Report saved to: "
            f"{report_path}"
        )

        return {
            **state,
            "report_path": str(
                report_path
            ),
        }

    except Exception as exc:

        print(
            f"[SAVE REPORT ERROR] "
            f"{exc}"
        )

        return {
            **state,
            "report_path": "",
        }


# ============================================================
# SAVE MEMORY NODE
# ============================================================

def save_memory_node(
    state: MarketAnalysisState,
) -> MarketAnalysisState:
    """
    Save the completed analysis into memory.
    """

    print()

    print(
        "=" * 70
    )

    print(
        "SAVE MEMORY"
    )

    print(
        "=" * 70
    )

    try:

        memory_id = save_state_to_memory(
            state
        )

        print(
            "Memory saved successfully."
        )

        print(
            f"Memory ID: {memory_id}"
        )

        return {
            **state,
            "memory_id": memory_id,
        }

    except Exception as exc:

        print(
            f"[SAVE MEMORY ERROR] "
            f"{exc}"
        )

        return {
            **state,
            "memory_id": None,
        }


# ============================================================
# BUILD WORKFLOW
# ============================================================

def build_workflow():
    """
    Build and compile the LangGraph workflow.
    """

    workflow = StateGraph(
        MarketAnalysisState
    )

    # ========================================================
    # NODES
    # ========================================================

    workflow.add_node(
        "input_guardrail",
        run_input_guardrail,
    )

    workflow.add_node(
        "query_analyzer",
        query_analyzer_node,
    )

    workflow.add_node(
        "memory",
        memory_node,
    )

    workflow.add_node(
        "research",
        research_node,
    )

    workflow.add_node(
        "analysis",
        analysis_node,
    )

    workflow.add_node(
        "report_drafter",
        report_drafter_node,
    )

    workflow.add_node(
        "output_guardrail",
        run_output_guardrail,
    )

    workflow.add_node(
        "save_report",
        save_report_node,
    )

    workflow.add_node(
        "save_memory",
        save_memory_node,
    )

    # ========================================================
    # EDGES
    # ========================================================

    workflow.add_edge(
        START,
        "input_guardrail",
    )

    workflow.add_conditional_edges(
        "input_guardrail",
        input_guardrail_router,
        {
            "query_analyzer": "query_analyzer",
            END: END,
        },
    )

    workflow.add_edge(
        "query_analyzer",
        "memory",
    )

    workflow.add_edge(
        "memory",
        "research",
    )

    workflow.add_edge(
        "research",
        "analysis",
    )

    workflow.add_edge(
        "analysis",
        "report_drafter",
    )

    workflow.add_edge(
        "report_drafter",
        "output_guardrail",
    )

    workflow.add_conditional_edges(
        "output_guardrail",
        output_guardrail_router,
        {
            "save_report": "save_report",
            END: END,
        },
    )

    workflow.add_edge(
        "save_report",
        "save_memory",
    )

    workflow.add_edge(
        "save_memory",
        END,
    )

    # ========================================================
    # COMPILE
    # ========================================================

    return workflow.compile()


# ============================================================
# REUSABLE WORKFLOW ENTRY FUNCTION
# ============================================================

def run_market_analysis(
    question: str,
) -> MarketAnalysisState:
    """
    Execute the complete market-analysis workflow.

    This function can be called from:

        - Terminal
        - Streamlit
        - API
        - Automated evaluation
    """

    question = question.strip()

    if not question:

        raise ValueError(
            "Market analysis question cannot be empty."
        )

    print()

    print(
        "#" * 70
    )

    print(
        "# MARKET ANALYSIS MULTI-AGENT SYSTEM"
    )

    print(
        "#" * 70
    )

    print()

    print(
        f"Question: {question}"
    )

    # ========================================================
    # BUILD GRAPH
    # ========================================================

    app = build_workflow()

    # ========================================================
    # INITIAL STATE
    # ========================================================

    initial_state: MarketAnalysisState = {

        "question": question,

        "companies": [],

        "intent": "",

        "context": "",

        "metrics": [],

        "fiscal_year": "",

        "memory_results": [],

        "memory_context": "",

        "memory_id": None,

        "company_research": {},

        "research_context": "",

        "analysis": "",

        "final_report": "",

        "report_path": "",

        "input_allowed": True,

        "output_allowed": True,

        "input_guardrail_reason": "",

        "output_guardrail_reason": "",

        "guardrail_message": "",
    }

    # ========================================================
    # MLFLOW RUN
    # ========================================================

    try:

        with mlflow.start_run(
            run_name="market_analysis"
        ):

            print()

            print(
                "[MLFLOW] Run started."
            )

            # ------------------------------------------------
            # LOG INITIAL PARAMETERS
            # ------------------------------------------------

            mlflow.log_param(
                "question",
                question,
            )

            mlflow.log_param(
                "workflow",
                "MarketAnalysis_MultiAgentSystem",
            )

            mlflow.log_param(
                "rag_enabled_companies",
                ", ".join(
                    sorted(
                        RAG_ENABLED_COMPANIES
                    )
                ),
            )

            # ------------------------------------------------
            # EXECUTE LANGGRAPH WORKFLOW
            # ------------------------------------------------

            final_state = app.invoke(
                initial_state
            )

            # ------------------------------------------------
            # EXTRACT FINAL STATE
            # ------------------------------------------------

            companies = final_state.get(
                "companies",
                [],
            )

            intent = final_state.get(
                "intent",
                "",
            )

            metrics = final_state.get(
                "metrics",
                [],
            )

            fiscal_year = final_state.get(
                "fiscal_year",
                "",
            )

            output_allowed = (
                final_state.get(
                    "output_allowed",
                    True,
                )
            )

            input_allowed = (
                final_state.get(
                    "input_allowed",
                    True,
                )
            )

            memory_results = (
                final_state.get(
                    "memory_results",
                    [],
                )
            )

            company_research = (
                final_state.get(
                    "company_research",
                    {},
                )
            )

            final_report = (
                final_state.get(
                    "final_report",
                    "",
                )
            )

            report_path = (
                final_state.get(
                    "report_path",
                    "",
                )
            )

            memory_id = (
                final_state.get(
                    "memory_id",
                    None,
                )
            )

            # =================================================
            # LOG QUERY ANALYSIS PARAMETERS
            # =================================================

            mlflow.log_param(
                "companies",
                ", ".join(
                    str(company)
                    for company in companies
                ),
            )

            mlflow.log_param(
                "intent",
                intent,
            )

            mlflow.log_param(
                "metrics",
                ", ".join(
                    str(metric)
                    for metric in metrics
                ),
            )

            mlflow.log_param(
                "fiscal_year",
                fiscal_year,
            )

            # =================================================
            # LOG WORKFLOW METRICS
            # =================================================

            mlflow.log_metric(
                "companies_count",
                len(companies),
            )

            mlflow.log_metric(
                "metrics_count",
                len(metrics),
            )

            mlflow.log_metric(
                "memory_results_count",
                len(memory_results),
            )

            mlflow.log_metric(
                "companies_researched",
                len(company_research),
            )

            mlflow.log_metric(
                "input_allowed",
                1 if input_allowed else 0,
            )

            mlflow.log_metric(
                "output_allowed",
                1 if output_allowed else 0,
            )

            mlflow.log_metric(
                "final_report_generated",
                1 if final_report else 0,
            )

            # =================================================
            # LOG RESEARCH METRICS
            # =================================================

            rag_company_count = 0

            financial_search_count = 0

            general_web_search_count = 0

            rag_document_count = 0

            financial_result_count = 0

            general_web_result_count = 0

            for company, research in (
                company_research.items()
            ):

                if research.get(
                    "rag_enabled",
                    False,
                ):

                    rag_company_count += 1

                rag_documents = research.get(
                    "rag_documents",
                    [],
                )

                financial_results = research.get(
                    "financial_results",
                    [],
                )

                general_web_results = research.get(
                    "general_web_results",
                    [],
                )

                if rag_documents:
                    rag_document_count += len(
                        rag_documents
                    )

                if financial_results:
                    financial_search_count += 1
                    financial_result_count += len(
                        financial_results
                    )

                if general_web_results:
                    general_web_search_count += 1
                    general_web_result_count += len(
                        general_web_results
                    )

            mlflow.log_metric(
                "rag_company_count",
                rag_company_count,
            )

            mlflow.log_metric(
                "rag_document_count",
                rag_document_count,
            )

            mlflow.log_metric(
                "financial_search_company_count",
                financial_search_count,
            )

            mlflow.log_metric(
                "financial_result_count",
                financial_result_count,
            )

            mlflow.log_metric(
                "general_web_search_company_count",
                general_web_search_count,
            )

            mlflow.log_metric(
                "general_web_result_count",
                general_web_result_count,
            )

            # =================================================
            # LOG TEXT ARTIFACTS
            # =================================================

            if final_report:

                mlflow.log_text(
                    final_report,
                    "final_report.txt",
                )

            analysis = final_state.get(
                "analysis",
                "",
            )

            if analysis:

                mlflow.log_text(
                    analysis,
                    "analysis.txt",
                )

            research_context = (
                final_state.get(
                    "research_context",
                    "",
                )
            )

            if research_context:

                mlflow.log_text(
                    research_context,
                    "research_context.txt",
                )

            # =================================================
            # LOG REPORT PATH
            # =================================================

            if report_path:

                mlflow.set_tag(
                    "report_path",
                    report_path,
                )

            # =================================================
            # LOG MEMORY ID
            # =================================================

            if memory_id is not None:

                mlflow.set_tag(
                    "memory_id",
                    str(memory_id),
                )

            # =================================================
            # LOG GUARDRAIL RESULTS
            # =================================================

            mlflow.set_tag(
                "input_guardrail_reason",
                final_state.get(
                    "input_guardrail_reason",
                    "",
                ),
            )

            mlflow.set_tag(
                "output_guardrail_reason",
                final_state.get(
                    "output_guardrail_reason",
                    "",
                ),
            )

            # =================================================
            # LOG COMPLETION STATUS
            # =================================================

            mlflow.set_tag(
                "workflow_status",
                "completed",
            )

            print()

            print(
                "[MLFLOW] Run completed successfully."
            )

            print(
                f"[MLFLOW] Experiment: "
                f"{MLFLOW_EXPERIMENT_NAME}"
            )

            # ------------------------------------------------
            # WORKFLOW COMPLETED
            # ------------------------------------------------

            print()

            print(
                "#" * 70
            )

            print(
                "# WORKFLOW COMPLETED"
            )

            print(
                "#" * 70
            )

            return final_state

    except KeyboardInterrupt:

        print()

        print(
            "Workflow interrupted by user."
        )

        # ----------------------------------------------------
        # Try to mark the active run as interrupted
        # ----------------------------------------------------

        active_run = (
            mlflow.active_run()
        )

        if active_run:

            mlflow.set_tag(
                "workflow_status",
                "interrupted",
            )

        raise

    except Exception as exc:

        print()

        print(
            "=" * 70
        )

        print(
            "WORKFLOW ERROR"
        )

        print(
            "=" * 70
        )

        print(
            exc
        )

        # ----------------------------------------------------
        # Log failure information to MLflow
        # ----------------------------------------------------

        active_run = (
            mlflow.active_run()
        )

        if active_run:

            mlflow.set_tag(
                "workflow_status",
                "failed",
            )

            mlflow.set_tag(
                "error_type",
                type(exc).__name__,
            )

            mlflow.log_text(
                str(exc),
                "error.txt",
            )

        raise


# ============================================================
# TERMINAL MAIN
# ============================================================

def main():
    """
    Terminal entry point.
    """

    print()

    print(
        "#" * 70
    )

    print(
        "# MARKET ANALYSIS MULTI-AGENT SYSTEM"
    )

    print(
        "#" * 70
    )

    print()

    question = input(
        "Enter your market analysis question:\n> "
    ).strip()

    if not question:

        print()

        print(
            "No question entered."
        )

        return

    try:

        final_state = (
            run_market_analysis(
                question
            )
        )

    except KeyboardInterrupt:

        print()

        print(
            "Workflow interrupted by user."
        )

        return

    except Exception:

        return

    # ========================================================
    # FINAL OUTPUT
    # ========================================================

    print()

    print(
        "#" * 70
    )

    print(
        "# FINAL REPORT"
    )

    print(
        "#" * 70
    )

    print()

    final_report = (
        final_state.get(
            "final_report",
            "",
        )
    )

    if final_report:

        print(
            final_report
        )

    else:

        print(
            "No final report was generated."
        )

    print()

    report_path = (
        final_state.get(
            "report_path"
        )
    )

    if report_path:

        print(
            f"Report file: "
            f"{report_path}"
        )

    memory_id = (
        final_state.get(
            "memory_id"
        )
    )

    if memory_id is not None:

        print(
            f"Memory ID: "
            f"{memory_id}"
        )

    print()


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()