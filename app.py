import html
import re
from datetime import datetime

import streamlit as st

from rag import analyze_proposal, load_index, GEMINI_MODEL, EMBEDDING_MODEL


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="ResearchGrant AI",
    page_icon="📑",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    :root {
        --brand-1: #4F46E5;
        --brand-2: #7C3AED;
        --ink: #1F2430;
        --muted: #6B7280;
        --border: #E5E7EB;
        --surface: #FFFFFF;
        --surface-muted: #F7F8FC;
    }

    /* Main page */

    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 3rem;
        max-width: 1300px;
    }

    /* Hero header */

    .hero {
        background: linear-gradient(120deg, var(--brand-1), var(--brand-2));
        border-radius: 18px;
        padding: 34px 38px;
        margin-bottom: 28px;
        color: white;
        box-shadow: 0 10px 30px -12px rgba(79, 70, 229, 0.55);
    }

    .hero-title {
        font-size: 40px;
        font-weight: 800;
        margin-bottom: 6px;
        display: flex;
        align-items: center;
        gap: 12px;
    }

    .hero-subtitle {
        font-size: 16px;
        opacity: 0.92;
        max-width: 720px;
    }

    /* Metric cards */

    .metric-card {
        padding: 18px;
        border-radius: 14px;
        border: 1px solid var(--border);
        background: var(--surface);
        text-align: center;
        min-height: 110px;
        border-top: 4px solid var(--brand-1);
        box-shadow: 0 2px 10px -6px rgba(31, 36, 48, 0.15);
    }

    .metric-value {
        font-size: 26px;
        font-weight: 700;
        color: var(--ink);
    }

    .metric-label {
        font-size: 13px;
        color: var(--muted);
        margin-top: 5px;
        text-transform: uppercase;
        letter-spacing: 0.03em;
    }

    /* Grant cards */

    .grant-card {
        padding: 20px 22px;
        border-radius: 16px;
        border: 1px solid var(--border);
        background: var(--surface);
        margin-bottom: 16px;
        box-shadow: 0 2px 10px -6px rgba(31, 36, 48, 0.12);
    }

    .grant-card-top {
        display: flex;
        justify-content: space-between;
        align-items: flex-start;
        gap: 12px;
        flex-wrap: wrap;
    }

    .grant-title {
        font-size: 20px;
        font-weight: 700;
        color: var(--ink);
        margin-bottom: 4px;
    }

    .grant-source {
        font-size: 12.5px;
        color: var(--muted);
    }

    .funding {
        font-size: 16px;
        font-weight: 700;
        color: var(--brand-1);
        margin-top: 10px;
    }

    .objective-line {
        font-size: 13.5px;
        color: var(--muted);
        font-style: italic;
        margin-top: 6px;
    }

    /* Similarity badge */

    .sim-badge {
        display: inline-block;
        padding: 5px 12px;
        border-radius: 999px;
        font-size: 12px;
        font-weight: 700;
        white-space: nowrap;
    }

    .sim-high {
        background: #E7F8EE;
        color: #1B7F4C;
    }

    .sim-mid {
        background: #FFF4E0;
        color: #B4720C;
    }

    .sim-low {
        background: #F1F2F6;
        color: #5B6270;
    }

    /* Tag chips */

    .tag-row {
        margin-top: 12px;
        display: flex;
        flex-wrap: wrap;
        gap: 6px;
    }

    .tag {
        display: inline-block;
        padding: 4px 11px;
        border-radius: 999px;
        font-size: 12px;
        font-weight: 600;
        background: var(--surface-muted);
        color: var(--ink);
        border: 1px solid var(--border);
    }

    .tag-focus {
        background: #EEF2FF;
        color: var(--brand-1);
        border-color: #E0E4FF;
    }

    .tag-applicant {
        background: #F5EEFF;
        color: var(--brand-2);
        border-color: #ECE0FF;
    }

    /* Section headings */

    .section-title {
        font-size: 24px;
        font-weight: 700;
        margin-top: 18px;
        margin-bottom: 10px;
        color: var(--ink);
    }

    /* Notice */

    .notice {
        padding: 14px 18px;
        border-radius: 12px;
        border: 1px solid #E5E7EB;
        background: var(--surface-muted);
        margin-bottom: 20px;
        font-size: 14px;
        color: var(--ink);
    }

    /* Footer */

    .footer {
        text-align: center;
        color: var(--muted);
        font-size: 13px;
        padding-top: 20px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def format_inr(value):
    """
    Convert an integer/float INR amount into Indian formatting.
    Example:
    1000000 -> ₹10,00,000
    """

    try:

        value = int(value)

        return "₹" + format(value, ",")

    except Exception:

        return str(value)


def get_grant_data(result):
    """
    Extract grant data safely from the result returned by rag.py.

    Always returns a (grant, data, score) triple so callers can
    unpack it unconditionally, even when the result is malformed.
    """

    grant = result.get("grant", {})

    if not isinstance(grant, dict):
        return {}, {}, 0.0

    data = grant.get("data", {})

    if not isinstance(data, dict):
        data = {}

    score = result.get("score", 0)

    try:
        score = float(score)
    except Exception:
        score = 0.0

    return grant, data, score


def extract_funding(data):
    """
    Extract maximum funding from the JSON structure.
    """

    funding = data.get("funding", {})

    if isinstance(funding, dict):

        amount = funding.get("maximum_inr")

        if amount is not None:
            return format_inr(amount)

    return "Not specified"


def similarity_badge(score):
    """
    Turn a raw cosine-similarity score into a (label, css_class)
    pair for display. Thresholds are heuristic, not calibrated
    probabilities.
    """

    if score >= 0.55:
        return "Strong match", "sim-high"

    if score >= 0.35:
        return "Possible match", "sim-mid"

    return "Weak match", "sim-low"


def render_tags(items, css_class):
    """
    Render a list of strings as HTML chip elements, escaping each
    value so grant data can never break out of the surrounding
    markup.
    """

    if not items:
        return ""

    chips = "".join(
        f'<span class="tag {css_class}">{html.escape(str(item))}</span>'
        for item in items
    )

    return chips


def clean_markdown(text):
    """
    Safely clean Gemini/LLM output before displaying it.
    Handles strings, dictionaries, lists, and other response types.
    """

    # ---------------------------------------------------------
    # Convert non-string responses to readable text
    # ---------------------------------------------------------
    if text is None:
        return ""

    if isinstance(text, str):
        pass

    elif isinstance(text, dict):
        # Handle common LLM response structures
        if "text" in text:
            text = text["text"]

        elif "content" in text:
            text = text["content"]

        else:
            text = str(text)

    elif isinstance(text, list):
        # Handle list of response blocks
        parts = []

        for item in text:

            if isinstance(item, str):
                parts.append(item)

            elif isinstance(item, dict):
                if "text" in item:
                    parts.append(str(item["text"]))

                elif "content" in item:
                    parts.append(str(item["content"]))

                else:
                    parts.append(str(item))

            else:
                parts.append(str(item))

        text = "\n".join(parts)

    else:
        text = str(text)

    # ---------------------------------------------------------
    # Now it is guaranteed to be a string
    # ---------------------------------------------------------

    text = re.sub(
        r'<div[^>]*>',
        '',
        text,
        flags=re.IGNORECASE
    )

    text = re.sub(
        r'</div>',
        '',
        text,
        flags=re.IGNORECASE
    )

    text = re.sub(
        r'<br\s*/?>',
        '\n',
        text,
        flags=re.IGNORECASE
    )

    # Remove unnecessary HTML tags
    text = re.sub(
        r'<[^>]+>',
        '',
        text
    )

    # Clean excessive blank lines
    text = re.sub(
        r'\n{3,}',
        '\n\n',
        text
    )

    return text.strip()


def build_report_text(proposal, cleaned_answer, sources):
    """
    Assemble a plain-text report combining the proposal, the
    retrieved grants, and the full Gemini analysis, for the
    download button.
    """

    lines = [
        "RESEARCHGRANT AI — ANALYSIS REPORT",
        "=" * 50,
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "",
        "SUBMITTED PROPOSAL",
        "-" * 50,
        proposal.strip(),
        "",
        "RETRIEVED GRANTS",
        "-" * 50,
    ]

    for i, result in enumerate(sources, start=1):

        grant, data, score = get_grant_data(result)

        name = data.get("name", "Grant")
        source_file = grant.get("source", "Unknown")
        funding = extract_funding(data)

        lines.append(
            f"{i}. {name}  "
            f"(source: {source_file}, "
            f"similarity: {score:.4f}, "
            f"max funding: {funding})"
        )

    lines += [
        "",
        "DETAILED ANALYSIS",
        "-" * 50,
        cleaned_answer,
    ]

    return "\n".join(lines)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
    <div class="hero">
        <div class="hero-title">📑 ResearchGrant AI</div>
        <div class="hero-subtitle">
            Retrieval-augmented matching between your research
            proposal and a curated set of funding opportunities,
            with a source-grounded analysis of fit and eligibility.
        </div>
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("ResearchGrant AI")

    st.write(
        """
        ResearchGrant AI uses Retrieval-Augmented Generation (RAG)
        to retrieve relevant grant opportunities and analyze their
        relationship with a research proposal.
        """
    )

    st.divider()

    st.subheader("System")

    try:
        _grants, _ = load_index()
        _grant_count = len(_grants)
    except Exception:
        _grant_count = "N/A"

    st.write(f"Grant documents: **{_grant_count}**")
    st.write("Retrieval: **Semantic similarity**")
    st.write(f"Embeddings: **{EMBEDDING_MODEL}**")
    st.write(f"Generation: **{GEMINI_MODEL}**")

    st.divider()

    st.subheader("Retrieval settings")

    top_k = st.slider(
        "Grants to retrieve",
        min_value=3,
        max_value=9,
        value=5,
        help=(
            "How many of the most semantically similar grants "
            "to retrieve and pass to Gemini for analysis."
        )
    )

    st.divider()

    st.subheader("Pipeline")

    st.write(
        """
        1. Proposal input
        2. Query embedding
        3. Semantic retrieval
        4. Top-K grant selection
        5. Gemini analysis
        6. Evidence display
        """
    )

    st.divider()

    st.caption(
        "Results are informational only. They do not represent "
        "official funding or eligibility decisions."
    )


# ============================================================
# PROPOSAL INPUT
# ============================================================

st.markdown(
    '<div class="section-title">Research Proposal</div>',
    unsafe_allow_html=True
)

EXAMPLE_PROPOSAL = (
    "We are developing an autonomous underwater robot for "
    "detecting and collecting plastic waste from water bodies.\n\n"
    "The system uses computer vision, AI-based plastic "
    "detection, underwater robotics, sensors, motors and an "
    "embedded controller.\n\n"
    "The project is being developed by undergraduate engineering "
    "students at an academic institution."
)

if "proposal_text" not in st.session_state:
    st.session_state["proposal_text"] = ""


def _load_example():
    st.session_state["proposal_text"] = EXAMPLE_PROPOSAL


col_intro, col_example = st.columns([5, 1.3])

with col_intro:
    st.write(
        "Enter your research proposal below. The system will retrieve "
        "the most relevant grant documents and analyze the proposal "
        "against them."
    )

with col_example:
    st.button(
        "Load example",
        on_click=_load_example,
        use_container_width=True
    )

proposal = st.text_area(
    "Proposal",
    height=280,
    key="proposal_text",
    placeholder=(
        "Example:\n\n"
        "We are developing an autonomous underwater robot for "
        "detecting and collecting plastic waste from water bodies.\n\n"
        "The system uses computer vision, AI-based plastic "
        "detection, underwater robotics, sensors and an embedded "
        "controller."
    ),
    label_visibility="collapsed"
)

st.caption(f"{len(proposal.strip())} characters")


# ============================================================
# ANALYZE BUTTON
# ============================================================

col_analyze, col_clear = st.columns([4, 1])

with col_analyze:
    analyze_clicked = st.button(
        "Analyze Proposal",
        type="primary",
        use_container_width=True
    )

with col_clear:
    if st.button("Clear results", use_container_width=True):
        for key in ("answer", "sources", "proposal"):
            st.session_state.pop(key, None)
        st.rerun()

if analyze_clicked:

    if not proposal.strip():

        st.warning(
            "Please enter a research proposal before analysis."
        )

    elif len(proposal.strip()) < 50:

        st.warning(
            "Please provide a more detailed proposal. "
            "At least a few sentences are recommended for "
            "meaningful retrieval."
        )

    else:

        with st.spinner(
            "Retrieving grants and generating analysis..."
        ):

            try:

                answer, sources = analyze_proposal(
                    proposal,
                    top_k=top_k
                )

                st.session_state["answer"] = answer
                st.session_state["sources"] = sources
                st.session_state["proposal"] = proposal

            except Exception as e:

                st.error(
                    "The analysis could not be completed."
                )

                st.code(
                    f"{type(e).__name__}: {str(e)}"
                )


# ============================================================
# DISPLAY RESULTS
# ============================================================

if "answer" in st.session_state:

    answer = st.session_state["answer"]
    sources = st.session_state["sources"]

    # --------------------------------------------------------
    # RESULT HEADER
    # --------------------------------------------------------

    st.divider()

    st.markdown(
        '<div class="section-title">Analysis Overview</div>',
        unsafe_allow_html=True
    )

    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    number_of_grants = len(sources)

    funding_values = []

    for result in sources:

        grant, data, score = get_grant_data(result)

        funding = data.get("funding", {})

        if isinstance(funding, dict):

            amount = funding.get("maximum_inr")

            if isinstance(amount, (int, float)):

                funding_values.append(
                    int(amount)
                )

    if funding_values:

        maximum_funding = max(funding_values)

        funding_display = format_inr(
            maximum_funding
        )

    else:

        funding_display = "N/A"

    if sources:

        best_similarity = max(
            float(result.get("score", 0))
            for result in sources
        )

    else:

        best_similarity = 0

    col1, col2, col3 = st.columns(3)

    with col1:

        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-value">
                    {number_of_grants}
                </div>
                <div class="metric-label">
                    Grants Retrieved
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:

        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-value">
                    {funding_display}
                </div>
                <div class="metric-label">
                    Highest Maximum Funding
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col3:

        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-value">
                    {best_similarity:.3f}
                </div>
                <div class="metric-label">
                    Highest Retrieval Similarity
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    # --------------------------------------------------------
    # INFORMATION NOTICE
    # --------------------------------------------------------

    st.markdown(
        """
        <div class="notice">
        <strong>Notice:</strong> This system provides an
        informational analysis only. It does not make an
        official funding or eligibility decision, nor does it
        guarantee eligibility or funding approval.
        </div>
        """,
        unsafe_allow_html=True
    )

    # --------------------------------------------------------
    # RETRIEVED GRANTS
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">'
        'Potentially Relevant Grants'
        '</div>',
        unsafe_allow_html=True
    )

    st.caption(
        "The grants below were retrieved using semantic "
        "similarity against the proposal."
    )

    for i, result in enumerate(
        sources,
        start=1
    ):

        grant, data, score = get_grant_data(
            result
        )

        grant_name = data.get(
            "name",
            "Grant"
        )

        source_file = grant.get(
            "source",
            "Unknown"
        )

        funding = extract_funding(
            data
        )

        objective = data.get("objective", "")

        sim_label, sim_class = similarity_badge(score)

        focus_tags = render_tags(
            data.get("focus_areas", []),
            "tag-focus"
        )

        applicant_tags = render_tags(
            data.get("eligible_applicants", []),
            "tag-applicant"
        )

        objective_html = (
            f'<div class="objective-line">{html.escape(objective)}</div>'
            if objective else ""
        )

        # ----------------------------------------------------
        # Grant card
        # ----------------------------------------------------

        st.markdown(
            f"""
            <div class="grant-card">

                <div class="grant-card-top">
                    <div>
                        <div class="grant-title">
                            {i}. {html.escape(str(grant_name))}
                        </div>
                        <div class="grant-source">
                            Source: {html.escape(str(source_file))}
                        </div>
                    </div>
                    <div class="sim-badge {sim_class}">
                        {sim_label} · {score:.3f}
                    </div>
                </div>

                {objective_html}

                <div class="funding">
                    Maximum Funding: {funding}
                </div>

                <div class="tag-row">
                    {focus_tags}
                    {applicant_tags}
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )

        with st.expander(
            f"View evidence — {grant_name}"
        ):

            st.write(
                f"**Source file:** `{source_file}`"
            )

            st.write(
                f"**Semantic similarity:** "
                f"`{score:.4f}`"
            )

            st.markdown(
                "### Grant Information"
            )

            st.code(
                grant.get("text", ""),
                language="text"
            )

    # --------------------------------------------------------
    # GEMINI ANALYSIS
    # --------------------------------------------------------

    st.divider()

    st.markdown(
        '<div class="section-title">'
        'Detailed Proposal Analysis'
        '</div>',
        unsafe_allow_html=True
    )

    cleaned_answer = clean_markdown(
        answer
    )

    st.markdown(
        cleaned_answer
    )

    # --------------------------------------------------------
    # USER PROPOSAL
    # --------------------------------------------------------

    st.divider()

    st.markdown(
        '<div class="section-title">'
        'Submitted Proposal'
        '</div>',
        unsafe_allow_html=True
    )

    with st.expander(
        "View submitted proposal"
    ):

        st.write(
            st.session_state.get(
                "proposal",
                ""
            )
        )

    # --------------------------------------------------------
    # SOURCE DOCUMENTS
    # --------------------------------------------------------

    st.divider()

    st.markdown(
        '<div class="section-title">'
        'Retrieved Source Documents'
        '</div>',
        unsafe_allow_html=True
    )

    for i, result in enumerate(
        sources,
        start=1
    ):

        grant, data, score = get_grant_data(
            result
        )

        grant_name = data.get(
            "name",
            "Grant"
        )

        source_file = grant.get(
            "source",
            "Unknown"
        )

        with st.expander(
            f"Source {i}: {grant_name}"
        ):

            st.write(
                f"**File:** `{source_file}`"
            )

            st.write(
                f"**Similarity:** `{score:.4f}`"
            )

            st.code(
                grant.get("text", ""),
                language="text"
            )

    # --------------------------------------------------------
    # DOWNLOAD REPORT
    # --------------------------------------------------------

    st.divider()

    st.markdown(
        '<div class="section-title">'
        'Export'
        '</div>',
        unsafe_allow_html=True
    )

    report_text = build_report_text(
        st.session_state.get("proposal", ""),
        cleaned_answer,
        sources
    )

    st.download_button(
        label="Download Analysis Report",
        data=report_text,
        file_name="researchgrant_analysis.txt",
        mime="text/plain",
        use_container_width=True
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.markdown(
    """
    <div class="footer">
        ResearchGrant AI • RAG-based Grant Matching System
    </div>
    """,
    unsafe_allow_html=True
)
