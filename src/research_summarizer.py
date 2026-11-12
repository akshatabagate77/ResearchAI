

import os
import json
import hashlib

from src.llm import generate_answer


# Store checkpoints inside the project folder.
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

CHECKPOINT_DIR = os.path.join(
    PROJECT_ROOT,
    ".cache",
    "research_summaries"
)


def create_document_id(document_name, document_chunks):
    """
    Create a unique ID using the document name and its content.

    If the document changes, it gets a different ID.
    """

    content = document_name

    for chunk in document_chunks:
        content += chunk.get("text", "")

    return hashlib.sha256(
        content.encode("utf-8")
    ).hexdigest()


def get_checkpoint_path(document_id):
    """
    Return the JSON checkpoint file path.
    """

    os.makedirs(
        CHECKPOINT_DIR,
        exist_ok=True
    )

    return os.path.join(
        CHECKPOINT_DIR,
        f"{document_id}.json"
    )


def load_checkpoint(document_id):
    """
    Load previously saved progress.
    """

    checkpoint_path = get_checkpoint_path(
        document_id
    )

    if os.path.exists(checkpoint_path):

        try:
            with open(
                checkpoint_path,
                "r",
                encoding="utf-8"
            ) as file:

                return json.load(file)

        except (json.JSONDecodeError, OSError):
            print(
                "Warning: Could not read checkpoint. "
                "Starting a fresh summary."
            )

    return {
        "completed_batches": {},
        "final_summary": None
    }


def save_checkpoint(document_id, checkpoint):
    """
    Save progress safely using a temporary file.
    """

    checkpoint_path = get_checkpoint_path(
        document_id
    )

    temporary_path = checkpoint_path + ".tmp"

    with open(
        temporary_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            checkpoint,
            file,
            indent=4,
            ensure_ascii=False
        )

    os.replace(
        temporary_path,
        checkpoint_path
    )


def analyse_batch(document_name, batch, batch_number):
    """
    Analyse one batch of research paper chunks.
    """

    context_parts = []

    for chunk in batch:

        context_parts.append(
            f"""
Page {chunk['page_number']}:

{chunk['text']}
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

    return generate_answer(prompt)


def create_final_summary(document_name, partial_summaries):
    """
    Combine all section analyses into one final summary.
    """

    combined_context = "\n\n".join(
        partial_summaries
    )

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
- Distinguish the authors' findings from your own
  interpretations.
- Mention page references when available.

FINAL RESEARCH SUMMARY:
"""

    return generate_answer(final_prompt)


def summarize_research_papers(
    chunks,
    batch_size=8,
    force_refresh=False
):
    """
    Summarise multiple research papers.

    Features:
    - Processes multiple documents.
    - Saves each completed batch.
    - Resumes from saved progress.
    - Avoids repeating completed API calls.
    - Reuses completed final summaries.

    Set force_refresh=True to ignore saved checkpoints.
    """

    if not chunks:
        raise ValueError(
            "No research paper chunks were provided."
        )

    if batch_size <= 0:
        raise ValueError(
            "Batch size must be greater than zero."
        )

    # Group chunks by document name.
    documents = {}

    for chunk in chunks:

        document_name = chunk.get(
            "document_name",
            "Unknown document"
        )

        documents.setdefault(
            document_name,
            []
        ).append(chunk)

    summaries = {}

    # Process each research paper.
    for document_name, document_chunks in documents.items():

        print(
            f"\nProcessing document: {document_name}"
        )

        document_id = create_document_id(
            document_name,
            document_chunks
        )

        # Load previous progress.
        if force_refresh:

            checkpoint = {
                "completed_batches": {},
                "final_summary": None
            }

        else:

            checkpoint = load_checkpoint(
                document_id
            )

        # If final summary already exists, reuse it.
        if checkpoint.get("final_summary"):

            print(
                f"Using saved summary for: {document_name}"
            )

            summaries[document_name] = (
                checkpoint["final_summary"]
            )

            continue

        completed_batches = checkpoint.get(
            "completed_batches",
            {}
        )

        total_batches = (
            len(document_chunks) + batch_size - 1
        ) // batch_size

        # Process each batch.
        for batch_index, start in enumerate(
            range(0, len(document_chunks), batch_size)
        ):

            batch_number = batch_index + 1

            batch_key = str(batch_index)

            # Skip previously completed batches.
            if batch_key in completed_batches:

                print(
                    f"Skipping completed batch "
                    f"{batch_number}/{total_batches}"
                )

                continue

            batch = document_chunks[
                start:start + batch_size
            ]

            print(
                f"Analysing batch "
                f"{batch_number}/{total_batches}..."
            )

            # Call Gemini.
            partial_summary = analyse_batch(
                document_name,
                batch,
                batch_number
            )

            # Save immediately after successful completion.
            completed_batches[batch_key] = (
                partial_summary
            )

            checkpoint["completed_batches"] = (
                completed_batches
            )

            save_checkpoint(
                document_id,
                checkpoint
            )

            print(
                f"Batch {batch_number} saved successfully."
            )

        # Collect partial summaries in their original order.
        ordered_partial_summaries = [
            completed_batches[str(i)]
            for i in range(total_batches)
        ]

        print(
            f"Generating final summary for: {document_name}"
        )

        # One final Gemini call per document.
        final_summary = create_final_summary(
            document_name,
            ordered_partial_summaries
        )

        # Save the final summary too.
        checkpoint["final_summary"] = final_summary

        save_checkpoint(
            document_id,
            checkpoint
        )

        summaries[document_name] = final_summary

        print(
            f"Completed summary: {document_name}"
        )

    return summaries
