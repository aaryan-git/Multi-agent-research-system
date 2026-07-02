from langchain.tools import tool
import requests
from bs4 import BeautifulSoup
import os
from dotenv import load_dotenv
from rich import print
import arxiv

from parser import extract_pdf_text, extract_sections

load_dotenv()

IEEE_API_KEY = os.getenv("IEEE_API_KEY")
SEMANTIC_SCHOLAR_API_KEY = os.getenv("SEMANTIC_SCHOLAR_API_KEY")


# ── Search tools (replace Tavily) ────────────────────────────────────────────

@tool
def search_arxiv(query: str) -> str:
    """Search arXiv for academic papers on a topic. Returns Titles, URLs, authors,
    publish dates and abstracts. No API key required."""
    try:
        client = arxiv.Client()
        search = arxiv.Search(query=query, max_results=5, sort_by=arxiv.SortCriterion.Relevance)
        out = []
        for r in client.results(search):
            out.append(
                f"Title: {r.title}\n"
                f"Authors: {', '.join(a.name for a in r.authors)}\n"
                f"Published: {r.published.date()}\n"
                f"URL: {r.pdf_url}\n"
                f"Abstract: {r.summary[:400]}\n"
            )
        return "\n----\n".join(out) if out else "No arXiv results found."
    except Exception as e:
        return f"arXiv search failed: {str(e)}"


@tool
def search_semantic(query: str) -> str:
    """Search Semantic Scholar for academic papers, including citation counts.
    Returns Titles, URLs, citation counts, year and abstracts."""
    try:
        headers = {"x-api-key": SEMANTIC_SCHOLAR_API_KEY} if SEMANTIC_SCHOLAR_API_KEY else {}
        params = {
            "query": query,
            "limit": 5,
            "fields": "title,url,abstract,year,citationCount,authors",
        }
        resp = requests.get(
            "https://api.semanticscholar.org/graph/v1/paper/search",
            params=params, headers=headers, timeout=10
        )
        resp.raise_for_status()
        data = resp.json()
        out = []
        for p in data.get("data", []):
            out.append(
                f"Title: {p.get('title')}\n"
                f"Year: {p.get('year')}\n"
                f"Citations: {p.get('citationCount')}\n"
                f"URL: {p.get('url')}\n"
                f"Abstract: {(p.get('abstract') or '')[:400]}\n"
            )
        return "\n----\n".join(out) if out else "No Semantic Scholar results found."
    except Exception as e:
        return f"Semantic Scholar search failed: {str(e)}"


@tool
def search_ieee(query: str) -> str:
    """Search IEEE Xplore for academic papers. Returns Titles, URLs, abstracts,
    and metadata. Requires IEEE_API_KEY to be set in .env."""
    if not IEEE_API_KEY:
        return "IEEE search unavailable: IEEE_API_KEY not set in .env"
    try:
        params = {
            "apikey": IEEE_API_KEY,
            "querytext": query,
            "max_records": 5,
            "format": "json",
        }
        resp = requests.get(
            "https://ieeexploreapi.ieee.org/api/v1/search/articles",
            params=params, timeout=10
        )
        resp.raise_for_status()
        data = resp.json()
        out = []
        for a in data.get("articles", []):
            out.append(
                f"Title: {a.get('title')}\n"
                f"Year: {a.get('publication_year')}\n"
                f"Citations: {a.get('citing_paper_count', 'N/A')}\n"
                f"URL: {a.get('pdf_url') or a.get('html_url')}\n"
                f"Abstract: {(a.get('abstract') or '')[:400]}\n"
            )
        return "\n----\n".join(out) if out else "No IEEE results found."
    except Exception as e:
        return f"IEEE search failed: {str(e)}"


# ── Reader tools (replace scrape_url) ────────────────────────────────────────

@tool
def read_pdf_tool(url: str) -> str:
    """Download and extract clean text content from a PDF paper URL. Splits the
    text into common academic sections (Abstract, Introduction, Methods,
    Results, Conclusion, References) when possible. Use this for arXiv/IEEE/
    Semantic Scholar paper links that point to PDFs."""
    try:
        resp = requests.get(url, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
        text = extract_pdf_text(resp.content)
        sections = extract_sections(text)
        if sections:
            formatted = "\n\n".join(f"## {name}\n{content[:1000]}" for name, content in sections.items())
            return formatted[:4000]
        return text[:4000]
    except Exception as e:
        return f"Could not read PDF: {str(e)}"


@tool
def read_html_tool(url: str) -> str:
    """Scrape and return clean text content from a given webpage URL. Fallback
    for sources that are not direct PDFs (e.g. abstract pages, blog posts)."""
    try:
        resp = requests.get(url, timeout=8, headers={"User-Agent": "Mozilla/5.0"})
        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer"]):
            tag.decompose()
        return soup.get_text(separator=" ", strip=True)[:3000]
    except Exception as e:
        return f"Could not scrape URL: {str(e)}"
