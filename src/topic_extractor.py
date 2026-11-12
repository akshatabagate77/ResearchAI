
from collections import Counter

from sklearn.feature_extraction.text import TfidfVectorizer


def extract_keywords(texts, top_n=10):
    """
    Extract important keywords from a collection
    of research-paper text passages using TF-IDF.
    """

    if not texts:
        return []

    cleaned_texts = [
        text.strip()
        for text in texts
        if text and text.strip()
    ]

    if not cleaned_texts:
        return []

    try:
        vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            max_features=3000
        )

        tfidf_matrix = vectorizer.fit_transform(
            cleaned_texts
        )

        feature_names = vectorizer.get_feature_names_out()

        scores = tfidf_matrix.sum(axis=0).A1

        keyword_scores = list(
            zip(feature_names, scores)
        )

        keyword_scores.sort(
            key=lambda item: item[1],
            reverse=True
        )

        return [
            {
                "keyword": keyword,
                "score": round(float(score), 4)
            }
            for keyword, score in keyword_scores[:top_n]
        ]

    except ValueError:
        return []


def extract_document_keywords(chunks, top_n=10):
    """
    Extract keywords separately for each uploaded paper.
    """

    document_texts = {}

    for chunk in chunks:

        document_name = chunk.get(
            "document_name",
            "Unknown document"
        )

        text = chunk.get("text", "")

        if document_name not in document_texts:
            document_texts[document_name] = []

        if text:
            document_texts[document_name].append(text)

    document_keywords = {}

    for document_name, texts in document_texts.items():

        document_keywords[document_name] = extract_keywords(
            texts,
            top_n=top_n
        )

    return document_keywords


def extract_common_keywords(document_keywords, top_n=15):
    """
    Find keywords that appear across uploaded papers.
    """

    keyword_counts = Counter()

    for keywords in document_keywords.values():

        # Count a keyword once per document.
        unique_keywords = {
            item["keyword"]
            for item in keywords
        }

        keyword_counts.update(unique_keywords)

    return [
        {
            "keyword": keyword,
            "paper_count": count
        }
        for keyword, count in keyword_counts.most_common(top_n)
    ]
