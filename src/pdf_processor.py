
import pymupdf


def extract_pages_from_pdf(pdf_bytes):

    document = pymupdf.open(
        stream=pdf_bytes,
        filetype="pdf"
    )

    pages = []

    try:
        for page_number, page in enumerate(document):

            page_text = page.get_text("text")

            pages.append({
                "page_number": page_number + 1,
                "text": page_text
            })

    finally:
        document.close()

    return pages
