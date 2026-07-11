from groq import RateLimitError, GroqError
import time

from agents import (
    build_reader_agent, build_search_agent, writer_chain, critic_chain,
    gemini_llm,
)
from xai import detect_research_gaps


def invoke_with_retry(chain_or_agent, inputs, max_wait: int = 30):
    """Retries a LangChain chain/agent .invoke() call if Groq rate-limits it.

    Waits a few seconds and retries whenever a RateLimitError (HTTP 429) is
    raised, up to `max_wait` total seconds, then gives up and re-raises.
    """
    waited = 0
    while True:
        try:
            return chain_or_agent.invoke(inputs)
        except RateLimitError as e:
            wait_time = 3
            print(f"\n⏳ Rate limit reached. Waiting {wait_time}s before retrying... ({e})")
            time.sleep(wait_time)
            waited += wait_time
            if waited >= max_wait:
                print("⚠️ Giving up on Groq after repeated rate limits.")
                raise


def invoke_agent_with_fallback(build_fn, inputs, max_wait: int = 15):
    """For tool-using agents (search/reader), create_agent's tool-binding
    strips out .with_fallbacks(), so automatic model fallback doesn't work.
    This tries Groq first (with brief retries on rate limits), and falls back
    to rebuilding the SAME agent with Gemini if Groq fails for ANY reason —
    rate limits, malformed tool calls (a known Groq/Llama quirk), or other
    API errors.
    """
    groq_agent = build_fn()  # uses primary_llm (Groq) by default
    try:
        return invoke_with_retry(groq_agent, inputs, max_wait=max_wait)
    except GroqError as e:
        print(f"\n🔁 Groq failed ({type(e).__name__}: {e}). Switching to Gemini for this step...")
        gemini_agent = build_fn(model=gemini_llm)
        return gemini_agent.invoke(inputs)


def run_research_pipeline(topic: str) -> dict:

    state = {}

    # step 1 - search agent working
    print("\n" + " =" * 50)
    print("step 1 - search agent is working (IEEE / arXiv / Semantic Scholar) ...")
    print("=" * 50)

    search_result = invoke_agent_with_fallback(build_search_agent, {
        "messages": [("user",
            f"Find recent, reliable and detailed academic papers about: {topic}. "
            f"For every paper you list, always include its full URL exactly as returned by the tools."
        )]
    })
    state["search_results"] = search_result['messages'][-1].content

    print("\n search result ", state['search_results'])

    # step 2 - reader agent
    print("\n" + " =" * 50)
    print("step 2 - Reader agent is reading top papers (PDF-first, HTML fallback) ...")
    print("=" * 50)

    reader_result = invoke_agent_with_fallback(build_reader_agent, {
        "messages": [("user",
            f"Below are academic paper search results about '{topic}', including URLs.\n\n"
            f"Search Results:\n{state['search_results']}\n\n"
            f"You MUST call one tool right now on one of the URLs above — do not just describe "
            f"what you would do. If a URL is labeled '(PDF)' or ends in .pdf, call read_pdf_tool "
            f"on it. Otherwise, call read_html_tool on it. Call the tool now, then summarize what "
            f"it returned."
        )]
    })

    state['scraped_content'] = reader_result['messages'][-1].content

    print("\nread content: \n", state['scraped_content'])

    # step 3 - writer chain
    print("\n" + " =" * 50)
    print("step 3 - Writer is drafting the report ...")
    print("=" * 50)

    research_combined = (
        f"SEARCH RESULTS : \n {state['search_results']} \n\n"
        f"DETAILED PAPER CONTENT : \n {state['scraped_content']}"
    )

    state["report"] = invoke_with_retry(writer_chain, {
        "topic": topic,
        "research": research_combined
    })

    print("\n Final Report\n", state['report'])

    # step 4 - critic report
    print("\n" + " =" * 50)
    print("step 4 - critic is reviewing the report ")
    print("=" * 50)

    state["feedback"] = invoke_with_retry(critic_chain, {
        "report": state['report']
    })

    print("\n critic report \n", state['feedback'])

    # step 5 - XAI layer: research gap detection
    print("\n" + " =" * 50)
    print("step 5 - XAI layer is detecting research gaps ")
    print("=" * 50)

    state["research_gaps"] = detect_research_gaps(research_combined)

    print("\n research gaps \n", state['research_gaps'])

    return state


if __name__ == "__main__":
    topic = input("\n Enter a research topic : ")
    run_research_pipeline(topic)