
def chunk_pages(pages, chunk_size=1000, overlap=150):

    if chunk_size <= 0:
        raise ValueError("Chunk size must be positive.")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError(
            "Overlap must be non-negative and smaller than chunk size."
        )

    all_chunks = []

    for page in pages:

        text = page["text"].strip()
        page_number = page["page_number"]

        start = 0

        while start < len(text):

            end = min(start + chunk_size, len(text))

            # Try to end at a word boundary
            if end < len(text):
                space_position = text.rfind(" ", start, end)

                if space_position > start + chunk_size * 0.6:
                    end = space_position

            chunk_text = text[start:end].strip()

            if chunk_text:

                all_chunks.append({
                    "chunk_id": len(all_chunks) + 1,
                    "page_number": page_number,
                    "text": chunk_text
                })

            if end == len(text):
                break

            start = end - overlap

    return all_chunks
