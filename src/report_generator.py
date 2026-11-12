
from io import BytesIO
from datetime import datetime

from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import (
    getSampleStyleSheet,
    ParagraphStyle
)
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    PageBreak,
    HRFlowable
)


# ==========================================
# MARKDOWN TEXT CONVERSION
# ==========================================

def format_inline_text(text):

    text = escape(text)

    # Basic Markdown formatting

    import re

    text = re.sub(
        r"\*\*(.*?)\*\*",
        r"<b>\1</b>",
        text
    )

    text = re.sub(
        r"\*(.*?)\*",
        r"<i>\1</i>",
        text
    )

    return text


# ==========================================
# CONVERT REPORT TEXT INTO PDF ELEMENTS
# ==========================================

def add_markdown_content(story, content, styles):

    if not content:
        story.append(
            Paragraph(
                "No content available.",
                styles["Normal"]
            )
        )
        return

    lines = content.split("\n")

    for line in lines:

        line = line.strip()

        if not line:
            story.append(Spacer(1, 6))
            continue

        # Main heading

        if line.startswith("# "):

            story.append(
                Paragraph(
                    format_inline_text(line[2:]),
                    styles["Heading1"]
                )
            )

        # Subheading

        elif line.startswith("## "):

            story.append(
                Paragraph(
                    format_inline_text(line[3:]),
                    styles["Heading2"]
                )
            )

        # Smaller heading

        elif line.startswith("### "):

            story.append(
                Paragraph(
                    format_inline_text(line[4:]),
                    styles["Heading3"]
                )
            )

        # Bullet points

        elif line.startswith("- ") or line.startswith("* "):

            story.append(
                Paragraph(
                    "&#8226; "
                    + format_inline_text(line[2:]),
                    styles["BulletText"]
                )
            )

        # Numbered points

        elif (
            len(line) > 2
            and line[0].isdigit()
            and line[1:3] == ". "
        ):

            story.append(
                Paragraph(
                    format_inline_text(line),
                    styles["BulletText"]
                )
            )

        # Markdown horizontal divider

        elif line in ["---", "***"]:

            story.append(
                HRFlowable(
                    width="100%",
                    thickness=0.7,
                    color=colors.lightgrey
                )
            )

            story.append(Spacer(1, 8))

        # Normal paragraph

        else:

            story.append(
                Paragraph(
                    format_inline_text(line),
                    styles["BodyText"]
                )
            )


# ==========================================
# PDF PAGE NUMBER AND FOOTER
# ==========================================

def add_page_number(canvas, document):

    canvas.saveState()

    page_width, page_height = A4

    canvas.setFont("Helvetica", 9)
    canvas.setFillColor(colors.grey)

    canvas.drawString(
        0.75 * inch,
        0.45 * inch,
        "AI Research Assistant | Research Analysis Report"
    )

    canvas.drawRightString(
        page_width - 0.75 * inch,
        0.45 * inch,
        f"Page {document.page}"
    )

    canvas.restoreState()


# ==========================================
# MAIN PDF GENERATOR
# ==========================================

def generate_research_report(
    summaries,
    comparison_report=None,
    gap_report=None
):

    if not summaries:
        raise ValueError(
            "Generate at least one research paper summary first."
        )

    # Create an in-memory PDF file

    buffer = BytesIO()

    document = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=0.75 * inch,
        leftMargin=0.75 * inch,
        topMargin=0.75 * inch,
        bottomMargin=0.75 * inch
    )

    # Define PDF styles

    styles = getSampleStyleSheet()

    styles.add(
        ParagraphStyle(
            name="CoverTitle",
            parent=styles["Title"],
            fontSize=25,
            leading=32,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#17365D"),
            spaceAfter=20
        )
    )

    styles.add(
        ParagraphStyle(
            name="CoverSubtitle",
            parent=styles["Normal"],
            fontSize=12,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#555555"),
            leading=18
        )
    )

    styles.add(
        ParagraphStyle(
            name="BulletText",
            parent=styles["BodyText"],
            leftIndent=18,
            firstLineIndent=-10,
            spaceAfter=7,
            leading=16
        )
    )

    styles["Heading1"].textColor = colors.HexColor("#17365D")
    styles["Heading1"].spaceBefore = 12
    styles["Heading1"].spaceAfter = 10

    styles["Heading2"].textColor = colors.HexColor("#2874A6")
    styles["Heading2"].spaceBefore = 10
    styles["Heading2"].spaceAfter = 7

    styles["Heading3"].textColor = colors.HexColor("#444444")

    styles["BodyText"].leading = 16
    styles["BodyText"].spaceAfter = 8

    # Content of the PDF

    story = []

    current_date = datetime.now().strftime(
        "%d %B %Y"
    )

    # ======================================
    # COVER PAGE
    # ======================================

    story.append(Spacer(1, 1.5 * inch))

    story.append(
        Paragraph(
            "AI RESEARCH ASSISTANT",
            styles["CoverTitle"]
        )
    )

    story.append(
        Paragraph(
            "Research Analysis Report",
            styles["CoverSubtitle"]
        )
    )

    story.append(Spacer(1, 0.5 * inch))

    story.append(
        HRFlowable(
            width="75%",
            thickness=1.5,
            color=colors.HexColor("#2874A6"),
            hAlign="CENTER"
        )
    )

    story.append(Spacer(1, 0.5 * inch))

    story.append(
        Paragraph(
            f"<b>Report Generated:</b> {current_date}",
            styles["CoverSubtitle"]
        )
    )

    story.append(
        Paragraph(
            f"<b>Research Papers Analysed:</b> {len(summaries)}",
            styles["CoverSubtitle"]
        )
    )

    story.append(Spacer(1, 1 * inch))

    story.append(
        Paragraph(
            "This report contains AI-generated research "
            "summaries, comparative analysis, and potential "
            "research gaps based on the uploaded documents.",
            styles["CoverSubtitle"]
        )
    )

    story.append(PageBreak())

    # ======================================
    # TABLE OF CONTENTS
    # ======================================

    story.append(
        Paragraph(
            "Report Contents",
            styles["Heading1"]
        )
    )

    story.append(
        Paragraph(
            "1. Individual Research Paper Summaries",
            styles["BodyText"]
        )
    )

    if comparison_report:

        story.append(
            Paragraph(
                "2. Comparative Research Analysis",
                styles["BodyText"]
            )
        )

    if gap_report:

        story.append(
            Paragraph(
                "3. Research Gap Analysis",
                styles["BodyText"]
            )
        )

    story.append(PageBreak())

    # ======================================
    # INDIVIDUAL PAPER SUMMARIES
    # ======================================

    story.append(
        Paragraph(
            "Individual Research Paper Summaries",
            styles["Heading1"]
        )
    )

    for document_name, summary in summaries.items():

        story.append(
            Paragraph(
                format_inline_text(document_name),
                styles["Heading2"]
            )
        )

        add_markdown_content(
            story,
            summary,
            styles
        )

        story.append(Spacer(1, 15))

        story.append(
            HRFlowable(
                width="100%",
                thickness=0.6,
                color=colors.lightgrey
            )
        )

    # ======================================
    # COMPARISON REPORT
    # ======================================

    if comparison_report:

        story.append(PageBreak())

        story.append(
            Paragraph(
                "Comparative Research Analysis",
                styles["Heading1"]
            )
        )

        add_markdown_content(
            story,
            comparison_report,
            styles
        )

    # ======================================
    # RESEARCH GAP REPORT
    # ======================================

    if gap_report:

        story.append(PageBreak())

        story.append(
            Paragraph(
                "Research Gap Analysis",
                styles["Heading1"]
            )
        )

        add_markdown_content(
            story,
            gap_report,
            styles
        )

    # ======================================
    # GENERATE FINAL PDF
    # ======================================

    document.build(
        story,
        onFirstPage=add_page_number,
        onLaterPages=add_page_number
    )

    buffer.seek(0)

    return buffer.getvalue()
