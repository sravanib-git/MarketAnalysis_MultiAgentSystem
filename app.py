import streamlit as st

from src.graph.workflow import run_market_analysis


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Market Analysis Multi-Agent System",
    page_icon="📊",
    layout="wide",
)


# ============================================================
# PAGE HEADER
# ============================================================

st.title("📊 Market Analysis Multi-Agent System")

st.markdown(
    """
    Ask a market-analysis question and let the multi-agent
    workflow research, analyze, and generate a final report.
    """
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("System Components")

    st.markdown(
        """
        - Query Analyzer
        - Memory Search
        - Internal RAG
        - Financial Web Search
        - General Web Search
        - Analysis Agent
        - Report Drafter
        - Input Guardrails
        - Output Guardrails
        - MLflow Observability
        """
    )

    st.divider()

    st.caption(
        "Internal RAG is currently enabled for Microsoft."
    )


# ============================================================
# USER INPUT
# ============================================================

user_question = st.text_area(
    "Enter your market analysis question",
    placeholder=(
        "Example: Compare Microsoft and NVIDIA "
        "revenue growth"
    ),
    height=120,
)


# ============================================================
# RUN BUTTON
# ============================================================

run_button = st.button(
    "🚀 Run Market Analysis",
    type="primary",
)


# ============================================================
# EXECUTE WORKFLOW
# ============================================================

if run_button:

    if not user_question.strip():

        st.warning(
            "Please enter a market-analysis question."
        )

    else:

        with st.spinner(
            "Running the multi-agent market-analysis workflow..."
        ):

            try:

                result = run_market_analysis(
                    user_question
                )

                st.success(
                    "Market analysis completed successfully."
                )

                # ------------------------------------------------
                # FINAL REPORT
                # ------------------------------------------------

                st.subheader("Final Report")

                final_report = result.get(
                    "final_report",
                    "",
                )

                if final_report:

                    st.markdown(
                        final_report
                    )

                else:

                    st.warning(
                        "No final report was generated."
                    )

                # ------------------------------------------------
                # WORKFLOW DETAILS
                # ------------------------------------------------

                with st.expander(
                    "View workflow details"
                ):

                    companies = result.get(
                        "companies",
                        [],
                    )

                    intent = result.get(
                        "intent",
                        "",
                    )

                    metrics = result.get(
                        "metrics",
                        [],
                    )

                    fiscal_year = result.get(
                        "fiscal_year",
                        "",
                    )

                    report_path = result.get(
                        "report_path",
                        "",
                    )

                    memory_id = result.get(
                        "memory_id",
                        None,
                    )

                    st.write(
                        "Companies:",
                        companies,
                    )

                    st.write(
                        "Intent:",
                        intent,
                    )

                    st.write(
                        "Metrics:",
                        metrics,
                    )

                    st.write(
                        "Fiscal year:",
                        fiscal_year,
                    )

                    st.write(
                        "Report path:",
                        report_path,
                    )

                    st.write(
                        "Memory ID:",
                        memory_id,
                    )

            except Exception as exc:

                st.error(
                    "The workflow failed."
                )

                st.exception(
                    exc
                )