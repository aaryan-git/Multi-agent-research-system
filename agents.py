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
# API KEYS
# ============================================================

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")


# ============================================================
# LLM CONFIGURATION
# ============================================================

# ------------------------------------------------------------
# 1. GROQ - PRIMARY
# ------------------------------------------------------------
#
# IMPORTANT:
# max_retries=0
#
# If Groq returns 429 (rate limit), do NOT repeatedly retry.
# The application will immediately use Gemini instead.
#

groq_llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0,
    max_retries=0,
    groq_api_key=GROQ_API_KEY,
)


# ------------------------------------------------------------
# 2. GEMINI - FALLBACK
# ------------------------------------------------------------

gemini_llm = ChatGoogleGenerativeAI(
    model="gemini-flash-lite-latest",
    temperature=0,
    google_api_key=GOOGLE_API_KEY,
)


# ============================================================
# PROVIDER SELECTION
# ============================================================

provider = os.getenv("LLM_PROVIDER", "groq").lower()


if provider == "gemini":

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
        "[agents.py] Using Groq (primary, openai/gpt-oss-20b), "
        "Gemini (fallback, gemini-flash-lite-latest)"
    )


# ============================================================
# LLM FALLBACK
# ============================================================
#
# Writer and Critic chains do not use tools.
#
# If Groq fails, LangChain can automatically switch to Gemini.
#

llm = primary_llm.with_fallbacks(
    [secondary_llm]
)


# ============================================================
# MESSAGE TRIMMING
# ============================================================
#
# Search and Reader agents can accumulate large tool outputs.
#
# Keeping the conversation smaller reduces:
# - TPM usage
# - request size
# - memory usage
# - probability of Groq 429 errors
#
# Approximation:
# ~4 characters ≈ 1 token
#


def approx_token_counter(messages) -> int:

    text = get_buffer_string(messages)

    return max(1, len(text) // 4)


@before_model
def trim_for_groq(state, runtime):

    messages = state["messages"]

    trimmed = trim_messages(
        messages,
        strategy="last",
        token_counter=approx_token_counter,

        # Keep enough context for the agent,
        # but stay comfortably below the Groq limit.
        max_tokens=4000,

        start_on="human",
        include_system=True,
    )

    return {
        "llm_input_messages": trimmed
    }


# ============================================================
# 1. SEARCH AGENT
# ============================================================

def build_search_agent(model=None):

    selected_model = model or primary_llm

    return create_agent(
        model=selected_model,

        tools=[
            search_arxiv,
            search_semantic,
            search_ieee,
        ],

        middleware=[
            trim_for_groq
        ],
    )


# ============================================================
# 2. READER AGENT
# ============================================================

def build_reader_agent(model=None):

    selected_model = model or primary_llm

    return create_agent(
        model=selected_model,

        tools=[
            read_pdf_tool,
            read_html_tool,
        ],

        middleware=[
            trim_for_groq
        ],
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

Do not invent:

- papers
- authors
- URLs
- datasets
- results
- statistics
- citations
- research findings

If information is missing, clearly state that
the available research information is insufficient.
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

Do not invent evidence that is not present
in the report or research.
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


# ============================================================
# DEBUG INFORMATION
# ============================================================

print(
    "[agents.py] LLM configuration loaded successfully."
)

print(
    f"[agents.py] Primary provider: "
    f"{'Gemini' if provider == 'gemini' else 'Groq'}"
)

print(
    "[agents.py] Groq retries: 0 "
    "(429 errors immediately use fallback)"
)

print(
    "[agents.py] Message trimming: 4000-token target"
)