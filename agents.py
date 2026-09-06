from langchain.agents import create_agent
from langchain.agents.middleware import before_model
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import trim_messages, get_buffer_string

from tools import (
    search_ieee,
    search_arxiv,
    search_semantic,
    read_pdf_tool,
    read_html_tool,
)

from dotenv import load_dotenv
import os


# ============================================================
# ENVIRONMENT
# ============================================================

load_dotenv()


# ============================================================
# LLM CONFIGURATION
# ============================================================

# ------------------------------------------------------------
# 1. Groq - PRIMARY
# ------------------------------------------------------------
#
# Model: openai/gpt-oss-20b
#
# llama-3.1-8b-instant and llama-3.3-70b-versatile were both
# deprecated by Groq (announced 2026-06-17) and fully shut down
# 2026-08-16. Groq's official 1:1 replacements are:
#   - llama-3.1-8b-instant  -> openai/gpt-oss-20b
#   - llama-3.3-70b-versatile -> openai/gpt-oss-120b
#
# gpt-oss-20b is used here because it's the current fast/
# lightweight tier: low latency on Groq's LPU hardware, smaller
# footprint than the 120b model so it uses fewer tokens per
# response. Good fit for high call-volume, multi-agent loops
# (many quick round trips) rather than deep single-shot
# reasoning.
#
# NOTE: on the free tier, gpt-oss-20b AND gpt-oss-120b share the
# same 8,000 TPM (tokens/minute) limit. Swapping between them
# will NOT fix a "Request too large ... TPM" 413 error - that
# error means the *prompt itself* (e.g. raw search-tool results
# stuffed into one message) is too big. The fix is trimming the
# message history below, not picking a different model.

groq_llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0,
    max_retries=2,
    groq_api_key=os.getenv("GROQ_API_KEY"),
)


# ------------------------------------------------------------
# 2. Gemini - FALLBACK
# ------------------------------------------------------------
#
# Model: gemini-flash-lite-latest
#
# "gemini-flash-latest" resolves to the current full Flash model
# (gemini-3.8-flash at time of writing), which has a very low
# free-tier daily request cap (as low as ~20-50 RPD). The
# Flash-Lite line carries a much more generous free-tier daily
# quota (~1,500 RPD), which is a better fit for a fallback that
# needs to absorb overflow traffic from Groq.

gemini_llm = ChatGoogleGenerativeAI(
    model="gemini-flash-lite-latest",
    temperature=0,
    google_api_key=os.getenv("GOOGLE_API_KEY"),
)


# ============================================================
# PROVIDER SELECTION
# ============================================================

provider = os.getenv("LLM_PROVIDER", "groq").lower()


if provider == "groq":

    primary_llm = groq_llm
    secondary_llm = gemini_llm

    print(
        "[agents.py] Using Groq (primary, openai/gpt-oss-20b), "
        "Gemini (fallback, gemini-flash-lite-latest)"
    )


elif provider == "gemini":

    primary_llm = gemini_llm
    secondary_llm = groq_llm

    print(
        "[agents.py] Using Gemini (primary), "
        "Groq (fallback)"
    )


else:

    primary_llm = groq_llm
    secondary_llm = gemini_llm

    print(
        "[agents.py] Invalid/unavailable provider. "
        "Using Groq (primary), Gemini (fallback)"
    )


# ============================================================
# WRITER / CRITIC FALLBACK
# ============================================================

# This fallback works for chains that don't use tools.

llm = primary_llm.with_fallbacks(
    [secondary_llm]
)


# ============================================================
# MESSAGE TRIMMING (keeps requests under Groq's 8,000 TPM cap)
# ============================================================
#
# Search/reader agents can accumulate large tool outputs (raw
# abstracts, PDF/HTML text) in the message history. Trimming
# before each model call keeps the request comfortably under
# the free-tier TPM ceiling. max_tokens is set conservatively
# below 8,000 to leave room for the model's own output tokens.
#
# NOTE: create_agent() in LangChain v1 no longer accepts
# pre_model_hook (that param only exists on the older/deprecated
# langgraph.prebuilt.create_react_agent). The v1 way to hook into
# the loop is via middleware + the @before_model decorator.

# NOTE: token_counter=groq_llm would fall back to LangChain's
# default GPT-2 tokenizer (via the `transformers` package), which
# raises ImportError if `transformers` isn't installed - and it's
# a heavy dependency to pull in just for trimming. Groq's own
# client doesn't expose a native tokenizer either, so we use a
# simple, dependency-free approximation instead: ~4 characters
# per token, which is a reasonable rule of thumb for English text
# and errs on the side of undercounting slightly (safer than
# overcounting when the goal is staying under a hard TPM limit).

def approx_token_counter(messages) -> int:
    text = get_buffer_string(messages)
    return len(text) // 4


@before_model
def trim_for_groq(state, runtime):
    trimmed = trim_messages(
        state["messages"],
        strategy="last",
        token_counter=approx_token_counter,
        max_tokens=5000,
        start_on="human",
        include_system=True,
    )
    return {"llm_input_messages": trimmed}


# ============================================================
# 1. SEARCH AGENT
# ============================================================

def build_search_agent(model=None):

    return create_agent(
        model=model or primary_llm,
        tools=[
            search_arxiv,
            search_semantic,
            search_ieee,
        ],
        middleware=[trim_for_groq],
    )


# ============================================================
# 2. READER AGENT
# ============================================================

def build_reader_agent(model=None):

    return create_agent(
        model=model or primary_llm,
        tools=[
            read_pdf_tool,
            read_html_tool,
        ],
        middleware=[trim_for_groq],
    )


# ============================================================
# 3. WRITER CHAIN
# ============================================================

writer_prompt = ChatPromptTemplate.from_messages([

    (
        "system",
        """
You are an expert academic research writer.

Your job is to create a clear, factual and
well-structured literature review.

Use ONLY the research information provided to you.
Do not invent papers, URLs, authors, datasets,
results or statistics.
"""
    ),

    (
        "human",
        """
Write a detailed academic literature review
report on the following topic.

Topic:
{topic}

Research Gathered:
{research}


Use this structure:

1. Introduction

2. Key Findings
   - Finding 1
   - Finding 2
   - Finding 3
   - Additional findings if available

3. Comparison of Existing Approaches

4. Research Gaps

5. Conclusion

6. Sources
   - List the papers and URLs found in the research.


IMPORTANT:

- Use factual information only.
- Do not create fake citations.
- Preserve URLs exactly as provided.
- Clearly distinguish between information
  actually found in papers and general observations.
- If the research information is insufficient,
  explicitly say so.
"""
    ),
])


writer_chain = (
    writer_prompt
    | llm
    | StrOutputParser()
)


# ============================================================
# 4. CRITIC CHAIN
# ============================================================

critic_prompt = ChatPromptTemplate.from_messages([

    (
        "system",
        """
You are a strict academic research critic.

Evaluate the report for:
- factual accuracy
- research quality
- completeness
- unsupported claims
- citation quality
- logical structure
- relevance to the topic
"""
    ),

    (
        "human",
        """
Review the research report below.

Report:
{report}


Respond in exactly this format:


Score: X/10

Strengths:
- ...
- ...
- ...

Areas to Improve:
- ...
- ...
- ...

Possible Hallucinations or Unsupported Claims:
- ...
- ...

Missing Evidence:
- ...
- ...

One line verdict:
...


Be honest and specific.
"""
    ),
])


critic_chain = (
    critic_prompt
    | llm
    | StrOutputParser()
)