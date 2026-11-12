
from src.citation_extractor import (
    extract_references_from_text,
    extract_document_references
)


sample_text = """
Introduction

Artificial intelligence is transforming research.

References

[1] Smith, J. (2022). Machine Learning in Healthcare.
Journal of AI Research. https://example.com/paper1

[2] Brown, A. (2023). Deep Learning Applications.
DOI: 10.1234/example.doi

[3] Wilson, R. (2021). Artificial Intelligence Systems.
International Journal of Computing.
"""

print("Testing Citation Extractor")
print("--------------------------")

references = extract_references_from_text(sample_text)

print("Total references:", len(references))

for number, reference in enumerate(references, start=1):

    print(f"\nReference {number}")
    print("Text:", reference["reference"])
    print("Year:", reference["year"])
    print("DOI:", reference["dois"])
    print("URLs:", reference["urls"])


sample_chunks = [
    {
        "document_name": "AI_Research.pdf",
        "text": sample_text
    }
]

document_results = extract_document_references(
    sample_chunks
)

print("\nDocument extraction successful!")

for document, refs in document_results.items():
    print(document, ":", len(refs), "references")
