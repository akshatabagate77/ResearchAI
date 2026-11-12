
import re


DOI_PATTERN = re.compile(
    r"\b10\.\d{4,9}/[-._;()/:A-Z0-9]+\b",
    re.IGNORECASE
)

URL_PATTERN = re.compile(
    r"https?://[^\s<>]+",
    re.IGNORECASE
)

YEAR_PATTERN = re.compile(
    r"\b(?:19|20)\d{2}\b"
)


def extract_doi(text):
    """Extract DOI identifiers from text."""

    matches = DOI_PATTERN.findall(text)

    return list(dict.fromkeys(
        match.rstrip(".,;")
        for match in matches
    ))


def extract_urls(text):
    """Extract URLs from text."""

    matches = URL_PATTERN.findall(text)

    return list(dict.fromkeys(
        match.rstrip(".,;)")
        for match in matches
    ))


def extract_year(text):
    """Extract the first likely publication year."""

    match = YEAR_PATTERN.search(text)

    return match.group(0) if match else None


def extract_references_from_text(text):
    """
    Extract reference entries from a paper's text.

    Looks for a References or Bibliography heading
    and processes the text following that heading.
    """

    if not text or not text.strip():
        return []

    lines = text.splitlines()

    start_index = None

    reference_heading = re.compile(
        r"^\s*(?:references|bibliography|works cited)"
        r"\s*[:.]?\s*$",
        re.IGNORECASE
    )

    for index, line in enumerate(lines):

        if reference_heading.match(line):
            start_index = index + 1
            break

    if start_index is None:
        return []

    reference_lines = lines[start_index:]

    references = []
    current_reference = ""

    for line in reference_lines:

        line = line.strip()

        if not line:
            continue

        # Detect common numbered reference entries
        starts_new_reference = bool(
            re.match(
                r"^(?:\[\d+\]|\d+[.)])\s+",
                line
            )
        )

        if starts_new_reference:

            if current_reference:
                references.append(
                    current_reference.strip()
                )

            current_reference = line

        elif current_reference:

            # Join wrapped lines belonging to the same reference
            current_reference += " " + line

    if current_reference:
        references.append(current_reference.strip())

    results = []

    for reference in references:

        results.append({
            "reference": reference,
            "year": extract_year(reference),
            "dois": extract_doi(reference),
            "urls": extract_urls(reference)
        })

    return results


def extract_document_references(chunks):
    """
    Group extracted references by uploaded document.

    Combines chunk text by document before extraction,
    avoiding repeated extraction from individual chunks.
    """

    document_texts = {}

    for chunk in chunks:

        document_name = chunk.get(
            "document_name",
            "Unknown document"
        )

        text = chunk.get("text", "")

        document_texts.setdefault(
            document_name, []
        ).append(text)

    document_results = {}

    for document_name, texts in document_texts.items():

        full_text = "\n".join(texts)

        document_results[document_name] = (
            extract_references_from_text(full_text)
        )

    return document_results
