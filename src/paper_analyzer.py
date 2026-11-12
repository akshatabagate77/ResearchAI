
import re


SECTION_PATTERNS = {
    "Abstract": [
        r"\babstract\b"
    ],
    "Introduction": [
        r"\bintroduction\b",
        r"\bbackground\b"
    ],
    "Methodology": [
        r"\bmethodology\b",
        r"\bmethods\b",
        r"\bmaterials and methods\b",
        r"\bproposed approach\b",
        r"\bexperimental setup\b"
    ],
    "Results": [
        r"\bresults\b",
        r"\bfindings\b",
        r"\bexperimental results\b"
    ],
    "Discussion": [
        r"\bdiscussion\b",
        r"\bresults and discussion\b"
    ],
    "Limitations": [
        r"\blimitations\b",
        r"\blimitations of the study\b",
        r"\bthreats to validity\b"
    ],
    "Conclusion": [
        r"\bconclusion\b",
        r"\bconclusions\b",
        r"\bconcluding remarks\b"
    ],
    "References": [
        r"\breferences\b",
        r"\bbibliography\b",
        r"\bworks cited\b"
    ]
}


def detect_sections(text):
    """
    Detect common research paper sections.

    This is a heuristic check and may produce false positives
    when section names appear in the body of the paper.
    """

    text = text.lower()

    detected = {}

    for section, patterns in SECTION_PATTERNS.items():

        detected[section] = any(
            re.search(pattern, text)
            for pattern in patterns
        )

    return detected


def analyze_paper(text):
    """
    Analyze the structural completeness of one research paper.
    """

    if not text or not text.strip():
        raise ValueError("Paper text cannot be empty.")

    sections = detect_sections(text)

    total_sections = len(sections)

    present_sections = sum(sections.values())

    completeness_score = round(
        (present_sections / total_sections) * 100,
        2
    )

    missing_sections = [
        section
        for section, present in sections.items()
        if not present
    ]

    return {
        "completeness_score": completeness_score,
        "present_sections": present_sections,
        "total_sections": total_sections,
        "detected_sections": sections,
        "missing_sections": missing_sections
    }


def analyze_research_collection(chunks):
    """
    Group chunks by document and analyze each paper.
    """

    document_texts = {}

    for chunk in chunks:

        document_name = chunk.get(
            "document_name",
            "Unknown document"
        )

        text = chunk.get("text", "")

        document_texts.setdefault(
            document_name,
            []
        ).append(text)

    results = {}

    for document_name, texts in document_texts.items():

        full_text = "\n".join(texts)

        if not full_text.strip():
            continue

        results[document_name] = analyze_paper(
            full_text
        )

    return results
