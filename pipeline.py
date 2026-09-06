from groq import RateLimitError, GroqError
import time
import re

from agents import (
    build_reader_agent,
    primary_llm,
    gemini_llm,
    writer_chain,
    critic_chain,
)
from tools import (
    search_arxiv,
    search_ieee,
    search_semantic,
)
from xai import detect_research_gaps


# ============================================================
# GROQ RETRY
# ============================================================

def invoke_with_retry(chain_or_agent, inputs, max_wait: int = 30):
    """
    Retry a LangChain chain/agent when Groq returns a rate-limit error.

    Uses exponential backoff (3s, 6s, 12s, ...) instead of a fixed
    3s sleep - fewer wasted retries against a hard rate limit that
    won't clear in 3 seconds anyway.
    """

    waited = 0
    wait_time = 3

    while True:
        try:
            return chain_or_agent.invoke(inputs)

        except RateLimitError as e:

            print(
                f"\n⏳ Rate limit reached. "
                f"Waiting {wait_time}s before retrying..."
            )

            time.sleep(wait_time)
            waited += wait_time

            if waited >= max_wait:
                print(
                    "\n⚠️ Giving up on Groq after repeated rate limits."
                )
                raise

            wait_time = min(wait_time * 2, 15)


# ============================================================
# AGENT / CHAIN FALLBACK
# ============================================================

def invoke_agent_with_fallback(build_fn, inputs, max_wait: int = 15):
    """
    Try the primary Groq model first.

    If Groq fails because of:
    - rate limits
    - API errors
    - malformed tool calls
    - other Groq errors

    rebuild the same agent using Gemini.
    """

    groq_agent = build_fn()

    try:
        return invoke_with_retry(
            groq_agent,
            inputs,
            max_wait=max_wait
        )

    except GroqError as e:

        print(
            f"\n🔁 Groq failed "
            f"({type(e).__name__}: {e})"
        )

        print(
            "➡️ Switching to Gemini for this step..."
        )

        gemini_agent = build_fn(model=gemini_llm)

        return gemini_agent.invoke(inputs)


def extract_text(content) -> str:
    """
    Normalize an LLM response's .content into a plain string.

    Groq/OpenAI-style responses return .content as a plain string.
    Gemini sometimes returns a list of content blocks instead, e.g.
    [{'type': 'text', 'text': '...'}]. This flattens either shape
    into a single string so downstream code (extract_urls, prompt
    building) never has to care which provider answered.
    """

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and "text" in block:
                parts.append(block["text"])
        return "\n".join(parts)

    return str(content)


def truncate_text(text: str, max_chars: int) -> str:
    """
    Hard-cap a block of text by character count (~4 chars/token
    rule of thumb) so combined raw search results stay small
    enough to fit Groq's free-tier 8,000 TPM limit in one request.
    """

    if len(text) <= max_chars:
        return text

    return text[:max_chars] + "\n... [truncated]"


def invoke_llm_with_fallback(prompt: str, max_wait: int = 15) -> str:
    """
    Same Groq -> Gemini fallback as invoke_agent_with_fallback, but
    for a single plain LLM call (no agent/tool loop). Used by the
    search step, which no longer needs tool-calling reasoning.
    """

    try:
        response = invoke_with_retry(
            primary_llm,
            prompt,
            max_wait=max_wait
        )
        return extract_text(response.content)

    except GroqError as e:

        print(
            f"\n🔁 Groq failed "
            f"({type(e).__name__}: {e})"
        )

        print(
            "➡️ Switching to Gemini for this step..."
        )

        response = gemini_llm.invoke(prompt)
        return extract_text(response.content)


# ============================================================
# EXTRACT URLS
# ============================================================

def extract_urls(text: str):
    """
    Extract HTTP/HTTPS URLs from the search-agent response.
    """

    if not text:
        return []

    urls = re.findall(
        r'https?://[^\s<>"\]\)]+',
        text
    )

    cleaned_urls = []

    for url in urls:

        # Remove punctuation that may be attached to URLs
        url = url.rstrip(".,;:")

        if url not in cleaned_urls:
            cleaned_urls.append(url)

    return cleaned_urls


# ============================================================
# CALL A TOOL DIRECTLY (no LLM involved)
# ============================================================

def call_tool_safe(tool, query: str, label: str, max_chars: int = 2500) -> str:
    """
    Call a LangChain tool directly with plain Python, bypassing the
    LLM entirely. Works whether `tool` is a @tool-decorated
    LangChain Tool object (has .invoke) or a plain function.
    Any failure is swallowed and reported inline, so one dead
    source doesn't take down the whole search step.

    Output is truncated to max_chars: raw search results (full
    abstracts x multiple papers x 3 sources) can easily add up to
    more than Groq's 8,000 TPM limit on their own before the
    formatting prompt's own text is even added. ~2,500 chars per
    source (~625 tokens x 3 sources = ~1,875 tokens) leaves plenty
    of room for the prompt template and the model's output.
    """

    try:
        if hasattr(tool, "invoke"):
            result = tool.invoke(query)
        else:
            result = tool(query)

        result = truncate_text(str(result), max_chars)

        return f"--- {label} RESULTS ---\n{result}"

    except Exception as e:
        print(f"⚠️ {label} search failed: {type(e).__name__}: {e}")
        return f"--- {label} RESULTS ---\n(no results - {label} search failed)"


# ============================================================
# SEARCH AGENT
# ============================================================
#
# Previously this ran a full LLM agent loop (model decides which
# tool to call, calls it, decides the next tool, calls it, ...)
# which cost 5+ Groq round trips per topic and a long instruction
# prompt on every one of them.
#
# Now: call all 3 search tools directly in Python (no LLM, no
# tokens spent), then make exactly ONE LLM call to select/format
# the combined raw results. Same output format, far fewer tokens
# and far fewer round trips.

def run_search_agent(topic: str):
    """
    Search IEEE, arXiv and Semantic Scholar for relevant papers.
    """

    print("\n" + " =" * 50)
    print(
        "step 1 - search agent is working "
        "(IEEE / arXiv / Semantic Scholar) ..."
    )
    print("=" * 50)

    raw_results = "\n\n".join([
        call_tool_safe(search_arxiv, topic, "ARXIV"),
        call_tool_safe(search_ieee, topic, "IEEE"),
        call_tool_safe(search_semantic, topic, "SEMANTIC SCHOLAR"),
    ])

    format_prompt = f"""
Research topic: {topic}

Below are raw search results from arXiv, IEEE and Semantic
Scholar. Select the most relevant, recent (2024-2026 where
possible) papers and format them.

Rules:
- Use ONLY the URLs that literally appear in the raw results
  below. Never invent or modify a URL.
- Skip any paper that has no usable URL in the raw results.
- Do not invent authors or citations.

Return each paper as:

PAPER N
Title: ...
Authors: ...
Year: ...
Source: ...
URL: ...
Relevance: ...

RAW RESULTS:
{raw_results}
"""

    content = invoke_llm_with_fallback(format_prompt, max_wait=30)

    print("\n🔎 SEARCH RESULTS:\n")
    print(content)

    urls = extract_urls(content)

    print(
        f"\n🔗 URLs detected by pipeline: {len(urls)}"
    )

    for i, url in enumerate(urls, start=1):
        print(f"{i}. {url}")

    return content, urls


# ============================================================
# READER AGENT
# ============================================================

def run_reader_agent(topic: str, search_results: str, urls: list):
    """
    Read papers using PDF-first / HTML fallback.
    """

    print("\n" + " =" * 50)
    print(
        "step 2 - Reader agent is reading top papers "
        "(PDF-first, HTML fallback) ..."
    )
    print("=" * 50)

    if not urls:

        print(
            "\n❌ No usable URLs were found in the search results."
        )

        print(
            "The Reader Agent will NOT be called because "
            "there is nothing valid to read."
        )

        return (
            "No papers could be retrieved because the Search Agent "
            "did not return usable URLs."
        )

    # Read a reasonable number of papers.
    # Start with the first 5 URLs.
    selected_urls = urls[:5]

    url_list = "\n".join(
        f"{i}. {url}"
        for i, url in enumerate(selected_urls, start=1)
    )

    # NOTE: previously this also re-embedded the full search_results
    # text here, duplicating everything the URL list already says.
    # Dropped to cut prompt size - the URLs are all the reader
    # actually needs to act on.

    reader_prompt = f"""
You are the academic paper Reader Agent.

Research topic:
{topic}

Candidate URLs (in priority order):
{url_list}

IMPORTANT:

You MUST actually call a reading tool.

Do NOT simply explain what you would do.

Choose the first usable URL from the list.

If the URL:
- ends with .pdf
- contains /pdf/
- clearly points to a PDF

then call read_pdf_tool.

Otherwise call read_html_tool.

After calling the tool, summarize the paper.

Extract:

1. Paper title
2. Authors
3. Year
4. Research problem
5. Methodology
6. AI/ML model used
7. Dataset
8. Important results/performance
9. Limitations
10. Relevance to the research topic
11. Original paper URL

IMPORTANT:
Do not invent information that is not present in the retrieved paper.

If the first URL cannot be read, try the next URL.

You MUST call a reading tool before responding.
"""

    reader_result = invoke_agent_with_fallback(
        build_reader_agent,
        {
            "messages": [
                ("user", reader_prompt)
            ]
        },
        max_wait=30
    )

    content = reader_result["messages"][-1].content

    print("\n📖 READER OUTPUT:\n")
    print(content)

    return content


# ============================================================
# WRITER
# ============================================================

def run_writer(topic: str, search_results: str, reader_content: str):
    """
    Generate the literature review.
    """

    print("\n" + " =" * 50)
    print(
        "step 3 - Writer is drafting the report ..."
    )
    print("=" * 50)

    research_combined = f"""
SEARCH RESULTS:

{search_results}


DETAILED PAPER CONTENT:

{reader_content}
"""

    report = invoke_with_retry(
        writer_chain,
        {
            "topic": topic,
            "research": research_combined
        },
        max_wait=30
    )

    print("\n📄 FINAL REPORT:\n")
    print(report)

    return report, research_combined


# ============================================================
# CRITIC
# ============================================================

def run_critic(report: str):
    """
    Critic evaluates the generated research report.
    """

    print("\n" + " =" * 50)
    print(
        "step 4 - critic is reviewing the report ..."
    )
    print("=" * 50)

    feedback = invoke_with_retry(
        critic_chain,
        {
            "report": report
        },
        max_wait=30
    )

    print("\n🧐 CRITIC REPORT:\n")
    print(feedback)

    return feedback


# ============================================================
# XAI RESEARCH GAP DETECTION
# ============================================================

def run_xai(research_combined: str):
    """
    Detect research gaps from the gathered research.
    """

    print("\n" + " =" * 50)
    print(
        "step 5 - XAI layer is detecting research gaps ..."
    )
    print("=" * 50)

    research_gaps = detect_research_gaps(
        research_combined
    )

    print("\n🧠 RESEARCH GAPS:\n")
    print(research_gaps)

    return research_gaps


# ============================================================
# COMPLETE PIPELINE
# ============================================================

def run_research_pipeline(topic: str) -> dict:

    state = {}

    print("\n")
    print("=" * 70)
    print("      MULTI-AGENT ACADEMIC RESEARCH SYSTEM")
    print("=" * 70)
    print(f"\nResearch Topic: {topic}")

    # --------------------------------------------------------
    # STEP 1
    # --------------------------------------------------------

    search_results, urls = run_search_agent(topic)

    state["search_results"] = search_results
    state["urls"] = urls

    # --------------------------------------------------------
    # STEP 2
    # --------------------------------------------------------

    reader_content = run_reader_agent(
        topic,
        search_results,
        urls
    )

    state["scraped_content"] = reader_content

    # --------------------------------------------------------
    # STEP 3
    # --------------------------------------------------------

    report, research_combined = run_writer(
        topic,
        search_results,
        reader_content
    )

    state["report"] = report

    # --------------------------------------------------------
    # STEP 4
    # --------------------------------------------------------

    feedback = run_critic(report)

    state["feedback"] = feedback

    # --------------------------------------------------------
    # STEP 5
    # --------------------------------------------------------

    research_gaps = run_xai(
        research_combined
    )

    state["research_gaps"] = research_gaps

    # --------------------------------------------------------
    # FINAL
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("              PIPELINE COMPLETED")
    print("=" * 70)

    return state


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    topic = input(
        "\n Enter a research topic : "
    ).strip()

    if not topic:
        print(
            "\n❌ Research topic cannot be empty."
        )

    else:
        run_research_pipeline(topic)