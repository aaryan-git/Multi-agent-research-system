from langchain.agents import create_agent
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from tools import search_ieee, search_arxiv, search_semantic, read_pdf_tool, read_html_tool
from dotenv import load_dotenv
import os

load_dotenv()

# model setup — Groq is primary (fast, generous per-minute limits), Gemini is
# the fallback if Groq fails. NOTE: .with_fallbacks() does NOT survive
# create_agent's tool-binding, so tool-using agents (search/reader) are
# rebuilt manually with gemini_llm on persistent failure — see pipeline.py /
# app.py's invoke_agent_with_fallback(). writer_chain / critic_chain (no
# tools) DO get automatic fallback via .with_fallbacks() below.

groq_llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,
    groq_api_key=os.getenv("GROQ_API_KEY"),
)

gemini_llm = ChatGoogleGenerativeAI(
    model="gemini-flash-latest",
    temperature=0,
    google_api_key=os.getenv("GOOGLE_API_KEY"),
)

# Optional manual override: set LLM_PROVIDER=gemini in .env to make Gemini
# primary everywhere (e.g. if Groq is fully out of daily tokens).
provider = os.getenv("LLM_PROVIDER", "groq").lower()

if provider == "gemini":
    primary_llm, secondary_llm = gemini_llm, groq_llm
    print("[agents.py] Using LLM provider: gemini (primary), groq (fallback)")
else:
    primary_llm, secondary_llm = groq_llm, gemini_llm
    print("[agents.py] Using LLM provider: groq (primary), gemini (fallback)")

llm = primary_llm.with_fallbacks([secondary_llm])  # works for writer_chain/critic_chain


# 1st agent - Search Agent (academic sources: IEEE, arXiv, Semantic Scholar)
def build_search_agent(model=None):
    return create_agent(
        model=model or primary_llm,
        tools=[search_arxiv, search_semantic, search_ieee]
    )

# 2nd agent - Reader Agent (PDF parsing, HTML fallback)

def build_reader_agent(model=None):
    return create_agent(
        model=model or primary_llm,
        tools=[read_pdf_tool, read_html_tool]
    )


# writer chain (unchanged)

writer_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert research writer. Write clear, structured and insightful literature review reports."),
    ("human", """Write a detailed academic literature review report on the topic below.

Topic: {topic}

Research Gathered:
{research}

Structure the report as:
- Introduction
- Key Findings (minimum 3 well-explained points)
- Conclusion
- Sources (list all papers/URLs found in the research)

Be detailed, factual and professional."""),
])

writer_chain = writer_prompt | llm | StrOutputParser()

# critic_chain (now also flags unsupported claims, feeding into the XAI gap layer)

critic_prompt = ChatPromptTemplate.from_messages([
     ("system", "You are a sharp and constructive academic research critic. Be honest and specific."),
    ("human", """Review the research report below and evaluate it strictly.

Report:
{report}

Respond in this exact format:

Score: X/10

Strengths:
- ...
- ...

Areas to Improve:
- ...
- ...

Possible Hallucinations or Unsupported Claims:
- ...

One line verdict:
..."""),
])

critic_chain = critic_prompt | llm | StrOutputParser()
