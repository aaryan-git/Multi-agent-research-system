"""Rule-based Explainable AI (XAI) layer.

Instead of explaining images, this module explains:
- why a paper was selected / how it should be ranked (score_papers, explain_ranking)
- what research gaps exist across the gathered research (detect_research_gaps)

Kept intentionally rule-based (no extra LLM calls) so it's fast, deterministic,
and easy to reason about/debug.
"""

from datetime import datetime


def score_papers(papers: list) -> list:
    """Score and rank papers using simple rule-based criteria: relevance
    (position in search results), recency, and citation count.

    Each item in `papers` should be a dict with optional keys:
        title, year (int), citations (int), relevance_rank (int, 0 = most relevant)

    Returns the same papers, each augmented with `xai_score` and
    `xai_reasons`, sorted by score descending.
    """
    current_year = datetime.now().year
    scored = []

    for paper in papers:
        score = 0
        reasons = []

        relevance_rank = paper.get("relevance_rank", 5)
        relevance_points = max(0, 5 - relevance_rank)
        score += relevance_points
        if relevance_points >= 3:
            reasons.append("high relevance")

        year = paper.get("year")
        if isinstance(year, int):
            age = current_year - year
            if age <= 2:
                score += 3
                reasons.append("recent publication")
            elif age <= 5:
                score += 1

        citations = paper.get("citations")
        if isinstance(citations, int):
            if citations >= 100:
                score += 3
                reasons.append("high citation count")
            elif citations >= 20:
                score += 1
                reasons.append("moderate citation count")

        scored_paper = dict(paper)
        scored_paper["xai_score"] = score
        scored_paper["xai_reasons"] = reasons
        scored.append(scored_paper)

    return sorted(scored, key=lambda p: p["xai_score"], reverse=True)


def explain_ranking(paper: dict) -> str:
    """Generate a human-readable rule-based explanation for why a paper was
    selected/ranked, in the style:

        Paper selected because:
        - high relevance
        - recent publication
        - high citation count
    """
    reasons = paper.get("xai_reasons")
    title = paper.get("title", "Unknown paper")

    if not reasons:
        return f"'{title}' selected based on general relevance to the query."

    reason_lines = "\n".join(f"- {r}" for r in reasons)
    return f"'{title}' selected because:\n{reason_lines}"


def detect_research_gaps(research_text: str) -> str:
    """Rule-based research gap detection. Scans combined research text for
    recurring methods/topics that appear WITHOUT commonly-expected
    complementary coverage (e.g. deep learning without explainability),
    and flags those as potential gaps.

    Example: "Most papers use CNN but lack XAI."
    """
    if not research_text:
        return "No research content available to analyze for gaps."

    text_lower = research_text.lower()

    gap_rules = [
        (
            ["cnn", "convolutional", "deep learning", "neural network"],
            ["xai", "explainab", "interpretab"],
            "Most papers use CNN/deep-learning methods but lack XAI (explainability).",
        ),
        (
            ["dataset", "benchmark"],
            ["real-world", "real world", "clinical", "production", "deployment"],
            "Research is largely benchmark/dataset-driven with limited real-world validation.",
        ),
        (
            ["accuracy", "performance", "state-of-the-art", "sota"],
            ["fairness", "bias", "ethic"],
            "Performance is frequently reported without discussion of fairness, bias, or ethics.",
        ),
        (
            ["survey", "review"],
            ["novel", "new method", "new approach", "propose"],
            "Coverage leans on surveys/reviews with fewer novel methodological contributions.",
        ),
        (
            ["small dataset", "limited data", "few samples"],
            ["large-scale", "large scale", "scalab"],
            "Several studies rely on small datasets, raising open questions about scalability.",
        ),
    ]

    gaps = []
    for present_terms, missing_terms, gap_text in gap_rules:
        has_present = any(term in text_lower for term in present_terms)
        has_missing = any(term in text_lower for term in missing_terms)
        if has_present and not has_missing:
            gaps.append(gap_text)

    if not gaps:
        return "No obvious research gaps detected using current rule-based heuristics."

    return "\n".join(f"- {g}" for g in gaps)
