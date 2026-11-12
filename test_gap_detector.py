
from src.research_gap_detector import detect_research_gaps

# Temporary test summaries
summaries = {
    "Paper 1": """
    The paper studies student academic-risk prediction
    using logistic regression, decision trees, random
    forests and SVM. It reports that model performance
    varies across evaluation methods.
    """,

    "Paper 2": """
    The paper explores cost-sensitive gradient boosting
    for early academic support identification. It focuses
    on class imbalance, precision-recall evaluation and
    probability calibration.
    """
}

selected_documents = ["Paper 1", "Paper 2"]

print("Analysing research gaps...")

result = detect_research_gaps(
    summaries,
    selected_documents
)

print("\nRESEARCH GAP REPORT\n")
print(result)
