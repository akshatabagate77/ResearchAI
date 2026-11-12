
from src.report_generator import generate_research_report


# Sample summary

summaries = {
    "Sample Research Paper 1": """
    ## Research Objective

    Study machine learning approaches for academic risk prediction.

    ## Methodology

    The paper explores logistic regression,
    decision trees and random forests.

    ## Key Findings

    Different models produce different prediction results.
    """
}


# Sample comparison

comparison = """
## Similarities

Both papers focus on academic risk prediction.

## Differences

They explore different machine learning methods.
"""


# Sample research gaps

gaps = """
## Identified Research Gaps

Further evaluation across diverse student populations
could be explored.

## Future Research Directions

Investigate more robust prediction approaches.
"""


print("Generating PDF report...")

pdf_data = generate_research_report(
    summaries=summaries,
    comparison_report=comparison,
    gap_report=gaps
)

with open("test_research_report.pdf", "wb") as file:

    file.write(pdf_data)

print("PDF generated successfully!")
