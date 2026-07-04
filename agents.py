from langchain.agents import create_agent
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from tools import search_ieee, search_arxiv, search_semantic, read_pdf_tool, read_html_tool
from dotenv import load_dotenv
import os

load_dotenv()

# model setup - switches based on LLM_PROVIDER in .env
provider = os.getenv("LLM_PROVIDER", "groq").lower()

if provider == "groq":
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0,
        groq_api_key=os.getenv("GROQ_API_KEY"),
    )
elif provider == "gemini":
    llm = ChatGoogleGenerativeAI(
        model="gemini-flash-latest",
        temperature=0,
        google_api_key=os.getenv("GOOGLE_API_KEY"),
    )
else:
    raise ValueError(f"Unknown LLM_PROVIDER: {provider}. Use 'groq' or 'gemini'.")

print(f"[agents.py] Using LLM provider: {provider} ({llm.model_name if hasattr(llm, 'model_name') else llm.model})")

#1st agent - Search Agent (academic sources: IEEE, arXiv, Semantic Scholar)
def build_search_agent():
    return create_agent(
        model = llm,
        tools= [search_arxiv, search_semantic, search_ieee]
    )

#2nd agent - Reader Agent (PDF parsing, HTML fallback)

def build_reader_agent():
    return create_agent(
        model = llm,
        tools = [read_pdf_tool, read_html_tool]
    )


#writer chain (unchanged)

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

#critic_chain (now also flags unsupported claims, feeding into the XAI gap layer)

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
