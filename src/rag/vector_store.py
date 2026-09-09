import shutil
import time
from pathlib import Path

from langchain_chroma import Chroma

from src.rag.document_loader import load_documents
from src.rag.text_splitter import split_documents
from src.config.azure_models import get_embedding_model


# =========================================================
# CONFIGURATION
# =========================================================

PERSIST_DIRECTORY = "data/vector_store"
COLLECTION_NAME = "microsoft_documents"

# Number of chunks added to Chroma at one time
BATCH_SIZE = 10

# True means the existing vector store will be deleted
# and recreated from the beginning.
RESET_VECTOR_STORE = True


# =========================================================
# CREATE VECTOR STORE
# =========================================================

def create_vector_store():
    """
    Create a Chroma vector store using only original
    document chunks.

    No hypothetical questions are generated.
    """

    # =====================================================
    # STEP 1 - RESET OLD VECTOR STORE
    # =====================================================

    if RESET_VECTOR_STORE:
        persist_path = Path(PERSIST_DIRECTORY)

        if persist_path.exists():
            print("\nRemoving existing vector store...")

            shutil.rmtree(persist_path)

            print("Existing vector store removed.")

    # =====================================================
    # STEP 2 - LOAD DOCUMENTS
    # =====================================================

    print("\nLoading documents...")

    documents = load_documents()

    print(
        f"Loaded {len(documents)} pages."
    )

    # =====================================================
    # STEP 3 - SPLIT DOCUMENTS INTO CHUNKS
    # =====================================================

    print("\nSplitting documents...")

    chunks = split_documents(documents)

    print(
        f"Created {len(chunks)} chunks."
    )

    # =====================================================
    # STEP 4 - LOAD EMBEDDING MODEL
    # =====================================================

    print("\nLoading Azure Embedding model...")

    embeddings = get_embedding_model()

    print(
        "Azure Embedding model loaded successfully."
    )

    # =====================================================
    # STEP 5 - CREATE CHROMA COLLECTION
    # =====================================================

    print("\nCreating Chroma collection...")

    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=PERSIST_DIRECTORY,
        collection_metadata={
            "hnsw:space": "cosine"
        }
    )

    print("Chroma collection created.")

    print("Similarity metric: cosine")
    print("Index: HNSW")

    # =====================================================
    # STEP 6 - ADD ORIGINAL CHUNKS IN BATCHES
    # =====================================================

    print("\nAdding chunks to Chroma...")

    print(
        f"Total chunks: {len(chunks)}"
    )

    print(
        f"Batch size: {BATCH_SIZE}"
    )

    for start in range(
        0,
        len(chunks),
        BATCH_SIZE
    ):
        batch = chunks[
            start:start + BATCH_SIZE
        ]

        print(
            f"\nAdding chunks "
            f"{start + 1} - "
            f"{start + len(batch)} "
            f"of {len(chunks)}"
        )

        max_retries = 3
        batch_added = False

        for attempt in range(max_retries):
            try:
                vector_store.add_documents(batch)

                print("Batch added successfully.")

                batch_added = True

                break

            except Exception as error:
                error_message = str(error)

                if "429" in error_message:
                    wait_time = 60 * (attempt + 1)

                    print(
                        f"Rate limit reached. "
                        f"Waiting {wait_time} seconds..."
                    )

                    time.sleep(wait_time)

                else:
                    raise error

        if not batch_added:
            raise RuntimeError(
                f"Failed to add batch "
                f"{start + 1} - "
                f"{start + len(batch)} "
                f"after {max_retries} attempts."
            )

        # Small delay between batches
        time.sleep(2)

    # =====================================================
    # STEP 7 - CHECK VECTOR COUNT
    # =====================================================

    stored_count = vector_store._collection.count()

    expected_count = len(chunks)

    print("\n========================================")
    print("VECTOR STORE CREATED SUCCESSFULLY")
    print("========================================")

    print(
        f"Expected vectors: {expected_count}"
    )

    print(
        f"Actual vectors stored: {stored_count}"
    )

    print(
        f"Location: {PERSIST_DIRECTORY}"
    )

    print("========================================")

    # =====================================================
    # STEP 8 - VALIDATE VECTOR COUNT
    # =====================================================

    if stored_count != expected_count:
        raise RuntimeError(
            f"Vector count mismatch. "
            f"Expected {expected_count}, "
            f"but found {stored_count}."
        )

    return vector_store


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":
    create_vector_store()