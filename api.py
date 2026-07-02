"""FastAPI backend for the Multi-Agent Academic Research Intelligence System.

Exposes the existing Search -> Reader -> Writer -> Critic (+ XAI) pipeline
as a single, stable JSON API. This is the contract the React + Tailwind
frontend will consume. Keep this response shape stable — per the project
plan, the frontend should never need to change just because backend
internals (agents, tools, prompts) change.

Run locally with:
    uvicorn api:app --reload --port 8000

Then POST to http://localhost:8000/research with {"topic": "..."}
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from pipeline import run_research_pipeline

app = FastAPI(
    title="Academic Research Intelligence System API",
    description="Multi-agent pipeline: Search -> Reader -> Writer -> Critic -> XAI",
    version="1.0.0",
)

# CORS: wide open for local dev so the React app (likely on :5173 or :3000)
# can call this freely. Tighten allow_origins to your real frontend URL
# before deploying anywhere public.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ResearchRequest(BaseModel):
    topic: str = Field(..., min_length=1, description="Research topic to investigate")


class ResearchResponse(BaseModel):
    topic: str
    search_results: str
    scraped_content: str
    report: str
    feedback: str
    research_gaps: str


@app.get("/health")
def health():
    """Simple liveness check for the frontend / deployment platform."""
    return {"status": "ok"}


@app.post("/research", response_model=ResearchResponse)
def research(payload: ResearchRequest):
    """Run the full multi-agent pipeline for a topic and return stable JSON.

    This is the ONLY endpoint the React frontend needs for the core flow:
    query input -> this call -> render search/reader/writer/critic/gaps.
    """
    topic = payload.topic.strip()
    if not topic:
        raise HTTPException(status_code=400, detail="Topic must not be empty.")

    try:
        state = run_research_pipeline(topic)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline failed: {str(e)}")

    return ResearchResponse(
        topic=topic,
        search_results=state.get("search_results", ""),
        scraped_content=state.get("scraped_content", ""),
        report=state.get("report", ""),
        feedback=state.get("feedback", ""),
        research_gaps=state.get("research_gaps", ""),
    )
