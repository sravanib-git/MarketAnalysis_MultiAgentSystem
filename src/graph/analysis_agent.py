from typing import Any

from src.config.azure_models import get_chat_model


llm = get_chat_model()


ANALYSIS_AGENT_PROMPT = """
You are a financial and competitive intelligence analysis agent.

Your task is to analyze research collected from internal RAG
and web search.

You must produce evidence-based analysis.

Rules:
1. Use only the information provided in the research.
2. Never invent financial values, percentages, dates, or facts.
3. Clearly identify missing information.
4. Do not treat one company's information as another company's information.
5. For comparisons, analyze each company separately first.
6. Compare companies only when the evidence is sufficiently available.
7. Check whether the fiscal years and reporting periods are compatible.
8. If values are available, calculate differences or growth rates carefully.
9. If evidence conflicts, mention the conflict instead of choosing randomly.
10. Do not provide investment advice.
11. Do not make unsupported predictions.
12. Keep the analysis clear and structured.

User Question:
{question}

Query Analysis:
{query_analysis}

Research Context:
{research_context}

Prepare an analysis containing:

1. Main findings
2. Company-by-company findings
3. Comparison, if applicable
4. Missing information
5. Data limitations
6. Calculations, if supported
7. Evidence-based conclusion

Return only the analysis.
"""


def analysis_agent(
    question: str,
    query_analysis: dict[str, Any],
    research_context: str,
) -> str:
    """
    Analyze the research collected for the user's question.
    """

    prompt = ANALYSIS_AGENT_PROMPT.format(
        question=question,
        query_analysis=query_analysis,
        research_context=research_context,
    )

    response = llm.invoke(prompt)

    return response.content