from src.graph.analysis_agent import analysis_agent


question = "Compare Microsoft and NVIDIA revenue growth in FY2025."

query_analysis = {
    "companies": ["Microsoft", "NVIDIA"],
    "intent": "comparison",
    "context": "revenue growth",
    "metrics": ["revenue growth"],
    "fiscal_year": "FY2025",
}

research_context = """
COMPANY: Microsoft

Internal RAG Evidence:
- Microsoft FY2025 total revenue increased by 15% year-over-year.
- Microsoft Cloud revenue increased by 23%.

Web Evidence:
- Microsoft FY2025 annual report confirms the revenue-growth figures.


COMPANY: NVIDIA

Internal RAG Evidence:
- No internal NVIDIA documents were available.

Web Evidence:
- No suitable NVIDIA FY2025 revenue-growth evidence was found.
"""

result = analysis_agent(
    question=question,
    query_analysis=query_analysis,
    research_context=research_context,
)

print("\n" + "=" * 60)
print("ANALYSIS AGENT TEST")
print("=" * 60)
print(result)