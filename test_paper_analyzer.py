
from src.paper_analyzer import (
    analyze_paper,
    analyze_research_collection
)


sample_paper = """
ABSTRACT

This paper explores artificial intelligence in healthcare.

1. INTRODUCTION

Artificial intelligence is transforming medical research.

2. METHODOLOGY

We use machine learning models to analyze medical data.

3. RESULTS

The experiments demonstrate improved prediction accuracy.

4. DISCUSSION

The findings are discussed in detail.

5. CONCLUSION

This research highlights the potential of AI in healthcare.

REFERENCES

[1] Smith, J. (2022). Artificial Intelligence in Medicine.
"""


print("Testing Research Paper Analyzer")
print("--------------------------------")

result = analyze_paper(sample_paper)

print("Completeness Score:", result["completeness_score"], "%")
print(
    "Present Sections:",
    result["present_sections"],
    "/",
    result["total_sections"]
)

print("\nDetected Sections:")

for section, present in result["detected_sections"].items():

    status = "Present" if present else "Missing"

    print(section, ":", status)

print("\nMissing Sections:")

for section in result["missing_sections"]:
    print("-", section)


# Test multiple-document analysis

sample_chunks = [
    {
        "document_name": "AI_Healthcare.pdf",
        "text": sample_paper
    },
    {
        "document_name": "Machine_Learning.pdf",
        "text": """
        ABSTRACT
        INTRODUCTION
        METHODOLOGY
        RESULTS
        CONCLUSION
        REFERENCES
        """
    }
]

collection_results = analyze_research_collection(
    sample_chunks
)

print("\nMultiple Paper Analysis:")

for document, analysis in collection_results.items():

    print(
        document,
        "->",
        analysis["completeness_score"],
        "%"
    )

print("\nAll tests completed!")
