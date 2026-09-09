from pathlib import Path
from typing import Any

import json
import sqlite3
from datetime import datetime


# ============================================================
# CONFIGURATION
# ============================================================

MEMORY_DIRECTORY = Path("data/memory")
MEMORY_DIRECTORY.mkdir(
    parents=True,
    exist_ok=True,
)

DATABASE_PATH = (
    MEMORY_DIRECTORY
    / "market_memory.db"
)


# ============================================================
# DATABASE CONNECTION
# ============================================================

def get_connection() -> sqlite3.Connection:
    """
    Create a connection to the memory database.
    """

    connection = sqlite3.connect(
        DATABASE_PATH
    )

    connection.row_factory = (
        sqlite3.Row
    )

    return connection


# ============================================================
# INITIALIZE DATABASE
# ============================================================

def initialize_memory() -> None:
    """
    Create the memory table if it does not already exist.
    """

    connection = get_connection()

    try:

        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS market_memory (
                id INTEGER PRIMARY KEY AUTOINCREMENT,

                timestamp TEXT NOT NULL,

                question TEXT NOT NULL,

                companies TEXT,

                intent TEXT,

                context TEXT,

                metrics TEXT,

                fiscal_year TEXT,

                analysis TEXT,

                final_report TEXT,

                report_path TEXT
            )
            """
        )

        connection.commit()

    finally:

        connection.close()


# ============================================================
# SAVE MEMORY
# ============================================================

def save_memory(
    question: str,
    companies: list[str] | None = None,
    intent: str = "",
    context: str = "",
    metrics: list[str] | None = None,
    fiscal_year: str | None = None,
    analysis: str = "",
    final_report: str = "",
    report_path: str = "",
) -> int:
    """
    Save one completed market-analysis interaction.

    Returns:
        Database ID of the saved memory.
    """

    initialize_memory()

    timestamp = datetime.now().isoformat(
        timespec="seconds"
    )

    companies = companies or []
    metrics = metrics or []

    connection = get_connection()

    try:

        cursor = connection.execute(
            """
            INSERT INTO market_memory (
                timestamp,
                question,
                companies,
                intent,
                context,
                metrics,
                fiscal_year,
                analysis,
                final_report,
                report_path
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                timestamp,
                question,
                json.dumps(companies),
                intent,
                context,
                json.dumps(metrics),
                fiscal_year,
                analysis,
                final_report,
                report_path,
            ),
        )

        connection.commit()

        return int(cursor.lastrowid)

    finally:

        connection.close()


# ============================================================
# GET RECENT MEMORIES
# ============================================================

def get_recent_memories(
    limit: int = 5,
) -> list[dict[str, Any]]:
    """
    Retrieve the most recent market-analysis interactions.
    """

    initialize_memory()

    connection = get_connection()

    try:

        cursor = connection.execute(
            """
            SELECT
                id,
                timestamp,
                question,
                companies,
                intent,
                context,
                metrics,
                fiscal_year,
                analysis,
                final_report,
                report_path
            FROM market_memory
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,),
        )

        rows = cursor.fetchall()

    finally:

        connection.close()

    memories = []

    for row in rows:

        memories.append(
            deserialize_memory(
                dict(row)
            )
        )

    return memories


# ============================================================
# SEARCH MEMORY
# ============================================================

def search_memory(
    query: str,
    limit: int = 5,
) -> list[dict[str, Any]]:
    """
    Search previous interactions using SQLite text matching.

    The search checks:
    - question
    - companies
    - context
    - intent
    """

    initialize_memory()

    if not query.strip():
        return []

    search_term = (
        f"%{query.strip()}%"
    )

    connection = get_connection()

    try:

        cursor = connection.execute(
            """
            SELECT
                id,
                timestamp,
                question,
                companies,
                intent,
                context,
                metrics,
                fiscal_year,
                analysis,
                final_report,
                report_path
            FROM market_memory
            WHERE
                question LIKE ?
                OR companies LIKE ?
                OR context LIKE ?
                OR intent LIKE ?
                OR metrics LIKE ?
            ORDER BY id DESC
            LIMIT ?
            """,
            (
                search_term,
                search_term,
                search_term,
                search_term,
                search_term,
                limit,
            ),
        )

        rows = cursor.fetchall()

    finally:

        connection.close()

    memories = []

    for row in rows:

        memories.append(
            deserialize_memory(
                dict(row)
            )
        )

    return memories


# ============================================================
# GET MEMORY BY ID
# ============================================================

def get_memory(
    memory_id: int,
) -> dict[str, Any] | None:
    """
    Retrieve a specific memory by database ID.
    """

    initialize_memory()

    connection = get_connection()

    try:

        cursor = connection.execute(
            """
            SELECT
                id,
                timestamp,
                question,
                companies,
                intent,
                context,
                metrics,
                fiscal_year,
                analysis,
                final_report,
                report_path
            FROM market_memory
            WHERE id = ?
            """,
            (memory_id,),
        )

        row = cursor.fetchone()

    finally:

        connection.close()

    if row is None:
        return None

    return deserialize_memory(
        dict(row)
    )


# ============================================================
# DELETE MEMORY
# ============================================================

def delete_memory(
    memory_id: int,
) -> bool:
    """
    Delete one memory.

    Returns:
        True if a record was deleted.
    """

    initialize_memory()

    connection = get_connection()

    try:

        cursor = connection.execute(
            """
            DELETE FROM market_memory
            WHERE id = ?
            """,
            (memory_id,),
        )

        connection.commit()

        return cursor.rowcount > 0

    finally:

        connection.close()


# ============================================================
# CLEAR ALL MEMORY
# ============================================================

def clear_memory() -> None:
    """
    Delete all stored memories.
    """

    initialize_memory()

    connection = get_connection()

    try:

        connection.execute(
            """
            DELETE FROM market_memory
            """
        )

        connection.commit()

    finally:

        connection.close()


# ============================================================
# DESERIALIZE MEMORY
# ============================================================

def deserialize_memory(
    memory: dict[str, Any],
) -> dict[str, Any]:
    """
    Convert JSON strings stored in SQLite back into Python lists.
    """

    for field in (
        "companies",
        "metrics",
    ):

        value = memory.get(field)

        if isinstance(value, str):

            try:

                memory[field] = json.loads(
                    value
                )

            except json.JSONDecodeError:

                memory[field] = []

        elif value is None:

            memory[field] = []

    return memory


# ============================================================
# FORMAT MEMORY FOR LLM
# ============================================================

def format_memory_for_llm(
    memories: list[dict[str, Any]],
) -> str:
    """
    Convert previous memories into concise context
    that can be supplied to an LLM.
    """

    if not memories:

        return (
            "No previous market-analysis "
            "memory is available."
        )

    sections = []

    for index, memory in enumerate(
        memories,
        start=1,
    ):

        companies = ", ".join(
            memory.get(
                "companies",
                [],
            )
        )

        metrics = ", ".join(
            memory.get(
                "metrics",
                [],
            )
        )

        sections.append(
            f"""
[PREVIOUS ANALYSIS {index}]

Timestamp:
{memory.get("timestamp", "")}

Question:
{memory.get("question", "")}

Companies:
{companies}

Intent:
{memory.get("intent", "")}

Context:
{memory.get("context", "")}

Metrics:
{metrics}

Fiscal Year:
{memory.get("fiscal_year")}

Previous Analysis:
{memory.get("analysis", "")}

Previous Report:
{memory.get("final_report", "")}
""".strip()
        )

    return "\n\n".join(
        sections
    )


# ============================================================
# BUILD MEMORY FROM WORKFLOW STATE
# ============================================================

def save_state_to_memory(
    state: dict[str, Any],
) -> int:
    """
    Save a completed LangGraph workflow state to memory.

    This keeps workflow.py clean by handling the
    state-to-memory conversion here.
    """

    return save_memory(
        question=state.get(
            "question",
            "",
        ),

        companies=state.get(
            "companies",
            [],
        ),

        intent=state.get(
            "intent",
            "",
        ),

        context=state.get(
            "context",
            "",
        ),

        metrics=state.get(
            "metrics",
            [],
        ),

        fiscal_year=state.get(
            "fiscal_year"
        ),

        analysis=state.get(
            "analysis",
            "",
        ),

        final_report=state.get(
            "final_report",
            "",
        ),

        report_path=state.get(
            "report_path",
            "",
        ),
    )


# ============================================================
# STANDALONE TEST
# ============================================================

def main() -> None:
    """
    Test the memory system independently.
    """

    print("\n" + "=" * 70)
    print("MEMORY SYSTEM TEST")
    print("=" * 70)

    initialize_memory()

    # --------------------------------------------------------
    # Save test memory
    # --------------------------------------------------------

    memory_id = save_memory(
        question=(
            "Compare Microsoft's cloud business "
            "with Amazon's cloud business."
        ),

        companies=[
            "Microsoft",
            "Amazon",
        ],

        intent="comparison",

        context="cloud business",

        metrics=[
            "cloud revenue",
            "cloud revenue growth",
        ],

        fiscal_year="FY2025",

        analysis=(
            "Microsoft showed strong cloud "
            "business growth."
        ),

        final_report=(
            "Microsoft cloud revenue increased "
            "during FY2025."
        ),

        report_path=(
            "data/reports/"
            "market_analysis_report.txt"
        ),
    )

    print(
        f"\nSaved memory ID: {memory_id}"
    )

    # --------------------------------------------------------
    # Get recent memories
    # --------------------------------------------------------

    print("\nRecent memories:")

    memories = get_recent_memories(
        limit=5
    )

    for memory in memories:

        print(
            f"\nID: {memory['id']}"
        )

        print(
            f"Question: "
            f"{memory['question']}"
        )

        print(
            f"Companies: "
            f"{memory['companies']}"
        )

    # --------------------------------------------------------
    # Search memory
    # --------------------------------------------------------

    print(
        "\nSearching memory for "
        "'Microsoft'..."
    )

    results = search_memory(
        "Microsoft",
        limit=5,
    )

    print(
        f"Matches found: "
        f"{len(results)}"
    )

    # --------------------------------------------------------
    # Format for LLM
    # --------------------------------------------------------

    formatted = format_memory_for_llm(
        results
    )

    print(
        "\nFormatted memory:"
    )

    print(formatted)

    print(
        "\n" + "=" * 70
    )

    print(
        "MEMORY TEST COMPLETED"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()