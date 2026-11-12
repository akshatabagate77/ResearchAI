from src.topic_extractor import (
    extract_keywords,
    extract_document_keywords,
    extract_common_keywords
)

sample_chunks = [
    {
        "document_name": "Paper A",
        "text": "Machine learning models improve medical diagnosis. Deep learning helps analyse medical images."
    },
    {
        "document_name": "Paper A",
        "text": "Neural networks and deep learning are widely used in medical image classification."
    },
    {
        "document_name": "Paper B",
        "text": "Artificial intelligence improves healthcare. Machine learning supports disease prediction."
    }
]

print("\nKEYWORDS:")
print(extract_keywords(
    [chunk["text"] for chunk in sample_chunks]
))

print("\nDOCUMENT KEYWORDS:")
document_keywords = extract_document_keywords(
    sample_chunks
)
print(document_keywords)

print("\nCOMMON KEYWORDS:")
print(extract_common_keywords(document_keywords))