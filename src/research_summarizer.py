
import hashlib
import json
import os
import time

from src.llm import generate_answer


# ---------------------------------------------------------
# STEP 1: CHECKPOINT CONFIGURATION
# ---------------------------------------------------------

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

CHECKPOINT_DIR = os.path.join(
    PROJECT_ROOT,
    ".cache",
    "research_summaries",
)


# ---------------------------------------------------------
# STEP 2: DOCUMENT IDENTIFICATION
# ---------------------------------------------------------

def create_document_id(document_name, document_chunks, batch_size=8):
    """
    Create a unique ID for a document and its settings.
    """

    hash_input = {
        "document_name": str(document_name),
        "batch_size": batch_size,
        "chunks": [
            {
                "page_number": chunk.get("page_number"),
                "text": chunk.get("text", ""),
            }
            for chunk in document_chunks
        ],
    }

    serialized = json.dumps(
        hash_input,
        ensure_ascii=False,
        sort_keys=True,
    )

    return hashlib.sha256(
        serialized.encode("utf-8")
    ).hexdigest()


# ---------------------------------------------------------
# STEP 3: CHECKPOINT MANAGEMENT
# ---------------------------------------------------------

def get_checkpoint_path(document_id):
    """Return the JSON checkpoint file path."""

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    return os.path.join(
        CHECKPOINT_DIR,
        f"{document_id}.json",
    )


def load_checkpoint(document_id):
    """Load saved progress or return an empty checkpoint."""

    checkpoint_path = get_checkpoint_path(document_id)

    if os.path.exists(checkpoint_path):

        try:
            with open(
                checkpoint_path,
                "r",
                encoding="utf-8",
            ) as file:

                checkpoint = json.load(file)

            if not isinstance(checkpoint, dict):
                raise ValueError(
                    "Checkpoint must contain a JSON object."
                )

            completed_batches = checkpoint.get(
                "completed_batches",
                {}
            )

            if not isinstance(completed_batches, dict):
                raise ValueError(
                    "Invalid completed_batches data."
                )

            # Keep only valid, non-empty batch summaries.
            completed_batches = {
                key: value
                for key, value in completed_batches.items()
                if isinstance(key, str)
                and isinstance(value, str)
                and value.strip()
            }

            final_summary = checkpoint.get("final_summary")

            if (
                final_summary is not None
                and not isinstance(final_summary, str)
            ):
                raise ValueError(
                    "Invalid final_summary data."
                )

            return {
                "completed_batches": completed_batches,
                "final_summary": final_summary,
            }

        except (
            json.JSONDecodeError,
            OSError,
            ValueError,
        ) as error:

            print(
                "Warning: Could not read checkpoint "
                f"({error}). Starting a fresh summary."
            )

    return {
        "completed_batches": {},
        "final_summary": None,
    }


def save_checkpoint(document_id, checkpoint):
    """Save progress safely using a temporary file."""

    checkpoint_path = get_checkpoint_path(document_id)
    temporary_path = checkpoint_path + ".tmp"

    try:
        with open(
            temporary_path,
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                checkpoint,
                file,
                indent=4,
                ensure_ascii=False,
            )

        os.replace(
            temporary_path,
            checkpoint_path,
        )

    finally:

        if os.path.exists(temporary_path):

            try:
                os.remove(temporary_path)

            except OSError:
                pass


# ---------------------------------------------------------
# STEP 4: ANALYSE INDIVIDUAL BATCH
# ---------------------------------------------------------

def analyse_batch(document_name, batch, batch_number):
    """Analyse one batch of research-paper chunks."""

    context_parts = []

    for chunk in batch:

        context_parts.append(
            f"""
Page {chunk.get('page_number', 'Unknown')}:

{chunk.get('text', '')}
"""
        )

    context = "\n\n".join(context_parts)

    prompt = f"""
You are an expert academic research analyst.

Analyse the following section of a research paper.

Extract:
1. Research objective
2. Problem being addressed
3. Methodology
4. Important technical concepts
5. Key findings
6. Limitations, if mentioned

Rules:
- Use only the provided text.
- Do not invent information.
- Clearly identify missing information.
- Preserve important technical details.
- Keep the analysis concise but useful.

DOCUMENT: {document_name}

SECTION NUMBER: {batch_number}

RESEARCH PAPER SECTION:

{context}

SECTION ANALYSIS:
"""

    result = generate_answer(prompt)

    if not isinstance(result, str) or not result.strip():

        raise ValueError(
            f"Empty analysis returned for batch {batch_number}."
        )

    return result


# ---------------------------------------------------------
# STEP 5: CREATE FINAL RESEARCH SUMMARY
# ---------------------------------------------------------

def create_final_summary(document_name, partial_summaries):
    """Combine all section analyses into one final summary."""

    if not partial_summaries:

        raise ValueError(
            "No section analyses are available."
        )

    combined_context = "\n\n".join(partial_summaries)

    final_prompt = f"""
You are an expert academic research analyst.

Prepare a structured summary of this research paper
using the section analyses below.

DOCUMENT: {document_name}

SECTION ANALYSES:

{combined_context}

Write the final summary using these headings:

1. Research Objective
2. Problem Statement
3. Methodology
4. Technologies and Algorithms
5. Key Findings
6. Limitations
7. Future Research Scope

Important instructions:
- Be accurate and concise.
- Do not invent facts.
- If a section is not discussed in the paper,
  write "Not clearly specified in the available text."
- Distinguish the authors' findings from your own interpretations.
- Mention page references when available.

FINAL RESEARCH SUMMARY:
"""

    result = generate_answer(final_prompt)

    if not isinstance(result, str) or not result.strip():

        raise ValueError(
            "The final summary returned an empty response."
        )

    return result


# ---------------------------------------------------------
# STEP 6: SUMMARIZE RESEARCH PAPERS
# ---------------------------------------------------------

def summarize_research_papers(
    chunks,
    batch_size=8,
    force_refresh=False,
):
    """
    Summarise multiple research papers.

    Features:
    - Processes multiple documents.
    - Saves each completed batch.
    - Resumes from saved progress.
    - Avoids repeating completed API calls.
    - Reuses completed final summaries.
    - Measures batch, document, and total execution time.

    Set force_refresh=True to ignore saved checkpoints.
    """

    # Validate input chunks.
    if not isinstance(chunks, list) or not chunks:

        raise ValueError(
            "No research paper chunks were provided."
        )

    # Validate batch size.
    if (
        not isinstance(batch_size, int)
        or isinstance(batch_size, bool)
        or batch_size <= 0
    ):

        raise ValueError(
            "Batch size must be a positive integer."
        )

    # Validate force_refresh.
    if not isinstance(force_refresh, bool):

        raise ValueError(
            "force_refresh must be True or False."
        )

    # Validate individual chunks.
    for index, chunk in enumerate(chunks):

        if not isinstance(chunk, dict):

            raise ValueError(
                f"Chunk {index} must be a dictionary."
            )

        if (
            not isinstance(chunk.get("text"), str)
            or not chunk["text"].strip()
        ):

            raise ValueError(
                f"Chunk {index} has no usable text."
            )

    # Start total timer.
    total_start_time = time.perf_counter()

    # Group chunks by document name.
    documents = {}

    for chunk in chunks:

        document_name = (
            chunk.get("document_name")
            or "Unknown document"
        )

        documents.setdefault(
            document_name,
            []
        ).append(chunk)

    summaries = {}

    # Process each document.
    for document_name, document_chunks in documents.items():

        document_start_time = time.perf_counter()

        print("\n" + "=" * 55)
        print(f"Processing document: {document_name}")
        print(f"Total chunks: {len(document_chunks)}")

        document_id = create_document_id(
            document_name,
            document_chunks,
            batch_size,
        )

        # Load or reset checkpoint.
        if force_refresh:

            checkpoint = {
                "completed_batches": {},
                "final_summary": None,
            }

        else:

            checkpoint = load_checkpoint(document_id)

        # Reuse saved final summary if available.
        cached_summary = checkpoint.get("final_summary")

        if (
            isinstance(cached_summary, str)
            and cached_summary.strip()
        ):

            print(
                f"Using saved summary for: {document_name}"
            )

            summaries[document_name] = cached_summary

            document_time = (
                time.perf_counter() - document_start_time
            )

            print(
                "Document processing time (cached): "
                f"{document_time:.2f} seconds."
            )

            print("=" * 55)

            continue

        completed_batches = checkpoint.get(
            "completed_batches",
            {}
        )

        total_batches = (
            len(document_chunks) + batch_size - 1
        ) // batch_size

        print(f"Total batches: {total_batches}")

        # -------------------------------------------------
        # STEP 6A: PROCESS BATCHES
        # -------------------------------------------------

        for batch_index, start in enumerate(
            range(0, len(document_chunks), batch_size)
        ):

            batch_number = batch_index + 1
            batch_key = str(batch_index)

            # Reuse previously completed batch.
            saved_batch = completed_batches.get(batch_key)

            if (
                isinstance(saved_batch, str)
                and saved_batch.strip()
            ):

                print(
                    f"Skipping completed batch "
                    f"{batch_number}/{total_batches}"
                )

                continue

            batch = document_chunks[
                start:start + batch_size
            ]

            print(
                f"\nAnalysing batch "
                f"{batch_number}/{total_batches}..."
            )

            batch_start_time = time.perf_counter()

            # Generate batch analysis.
            partial_summary = analyse_batch(
                document_name,
                batch,
                batch_number,
            )

            batch_time = (
                time.perf_counter() - batch_start_time
            )

            print(
                f"Batch {batch_number} completed in "
                f"{batch_time:.2f} seconds."
            )

            # Save completed batch.
            completed_batches[batch_key] = partial_summary

            checkpoint["completed_batches"] = completed_batches

            save_checkpoint(
                document_id,
                checkpoint,
            )

            print("Batch saved successfully.")

        # -------------------------------------------------
        # STEP 6B: COLLECT PARTIAL SUMMARIES
        # -------------------------------------------------

        ordered_partial_summaries = []

        for index in range(total_batches):

            batch_key = str(index)

            partial_summary = completed_batches.get(batch_key)

            if (
                not isinstance(partial_summary, str)
                or not partial_summary.strip()
            ):

                raise RuntimeError(
                    f"Missing completed batch {batch_key} "
                    f"for {document_name}."
                )

            ordered_partial_summaries.append(
                partial_summary
            )

        # -------------------------------------------------
        # STEP 6C: GENERATE FINAL SUMMARY
        # -------------------------------------------------

        print(
            f"\nGenerating final summary for: {document_name}"
        )

        final_start_time = time.perf_counter()

        final_summary = create_final_summary(
            document_name,
            ordered_partial_summaries,
        )

        final_time = (
            time.perf_counter() - final_start_time
        )

        print(
            f"Final summary generated in "
            f"{final_time:.2f} seconds."
        )

        # Save final summary.
        checkpoint["final_summary"] = final_summary

        save_checkpoint(
            document_id,
            checkpoint,
        )

        summaries[document_name] = final_summary

        document_time = (
            time.perf_counter() - document_start_time
        )

        print(f"\nCompleted summary: {document_name}")

        print(
            f"Total document processing time: "
            f"{document_time:.2f} seconds."
        )

        print("=" * 55)

    # -----------------------------------------------------
    # STEP 7: DISPLAY TOTAL EXECUTION TIME
    # -----------------------------------------------------

    total_time = (
        time.perf_counter() - total_start_time
    )

    print("\n" + "=" * 55)

    print(
        f"Total summarization time: {total_time:.2f} seconds."
    )

    print(
        f"Documents processed: {len(summaries)}"
    )

    print("=" * 55)

    return summaries
