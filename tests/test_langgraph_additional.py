import asyncio
import json

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field

from AgenticAI.agentic.decision_logger import DecisionLogger
from AgenticAI.agentic.langgraph_runner import AgenticGraphRunner, AsyncRateLimiter
from AgenticAI.agentic.models import (
    ContextAssessment,
    EvidenceDigest,
    RequirementBundle,
    RequirementItem,
    WebContentDecision,
    WebSourceCandidate,
    WebSourceSelection,
    WebSelectionResponse,
)


class FakeChatModel(BaseChatModel):
    responses: list[object] = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "fake-chat"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        content = self.responses.pop(0) if self.responses else ""
        message = AIMessage(content=content)
        return ChatResult(generations=[ChatGeneration(message=message)])


class FakeMCPClient:
    def __init__(self):
        self.calls = []

    async def call_tool(self, tool_name: str, arguments: dict) -> str:
        self.calls.append((tool_name, arguments))
        if tool_name == "web_search":
            return json.dumps(
                {
                    "results": [
                        {
                            "title": "Example",
                            "snippet": "Snippet",
                            "href": "http://example.com",
                        }
                    ]
                }
            )
        if tool_name == "fetch_web_page":
            return json.dumps({"content": "page content"})
        if tool_name == "retrieve_chunks":
            return json.dumps({"results": []})
        return json.dumps({})


def _make_runner(responses):
    model = FakeChatModel(responses=responses)
    logger = DecisionLogger("/tmp/decision_log.jsonl")
    return AgenticGraphRunner(
        query_model=model,
        requirements_model=model,
        mcp_client=FakeMCPClient(),
        decision_logger=logger,
        throttle_enabled=False,
        web_search_enabled=True,
        max_web_queries_per_doc=1,
    )


def test_context_node_web_search_flow(monkeypatch):
    assessment = ContextAssessment(
        needs_additional_context=True,
        missing_information_queries=["esrs"],
        explanation="need more",
    )
    selection = WebSelectionResponse(
        selections=[
            WebSourceSelection(identifier="result_1", fetch=True, rationale="relevant")
        ]
    )
    decision = WebContentDecision(include=True, rationale="ok", summary="summary")

    responses = [
        assessment.model_dump_json(),
        selection.model_dump_json(),
        decision.model_dump_json(),
    ]
    runner = _make_runner(responses)

    state = {
        "doc_source": "doc.pdf",
        "document_summary": "summary",
        "document_type": "Directive",
        "retrieval_results": [],
    }

    result = asyncio.run(runner._context_node(state))
    assert result["web_context"]
    assert result["web_context"][0]["href"] == "http://example.com"


def test_context_node_web_search_disabled():
    assessment = ContextAssessment(
        needs_additional_context=True,
        missing_information_queries=["esrs"],
        explanation="need more",
    )
    responses = [assessment.model_dump_json()]
    runner = AgenticGraphRunner(
        query_model=FakeChatModel(responses=responses),
        requirements_model=FakeChatModel(responses=[]),
        mcp_client=FakeMCPClient(),
        web_search_enabled=False,
        max_web_queries_per_doc=1,
        throttle_enabled=False,
    )

    state = {
        "doc_source": "doc.pdf",
        "document_summary": "summary",
        "document_type": "Directive",
        "retrieval_results": [],
    }

    result = asyncio.run(runner._context_node(state))
    assert result["web_context"] == []


def test_distill_evidence_node_success(monkeypatch):
    evidence = EvidenceDigest(
        cards=[
            {
                "origin": "document",
                "claim": "Report emissions",
                "source": "doc.pdf",
                "page": 1,
                "chunk_id": "c1",
                "certainty": "high",
                "support": "support",
            }
        ]
    )
    responses = [evidence.model_dump_json()]
    runner = _make_runner(responses)
    monkeypatch.setattr(runner, "_build_structured_chain", lambda *_: None)

    state = {
        "doc_source": "doc.pdf",
        "document_type": "Directive",
        "retrieval_results": [
            {
                "chunk_id": "c1",
                "source": "doc.pdf",
                "page": 1,
                "parent_heading": None,
                "text": "Line 1",
            }
        ],
        "web_context": [],
    }

    result = asyncio.run(runner._distill_evidence_node(state))
    assert result["evidence_cards"][0]["claim"] == "Report emissions"


def test_requirements_refinement_and_merge(monkeypatch):
    bundle = RequirementBundle(
        document="doc.pdf",
        document_type="Directive",
        business_requirements=[
            RequirementItem(
                id="BR-1",
                description="Do thing",
                rationale="Why",
                document_sources=["doc.pdf, page 1, chunk c1"],
                online_sources=[],
            )
        ],
        data_requirements=[],
        assumptions=[],
    )
    responses = [bundle.model_dump_json(), bundle.model_dump_json()]
    runner = _make_runner(responses)
    monkeypatch.setattr(runner, "_build_structured_chain", lambda *_: None)

    state = {
        "doc_source": "doc.pdf",
        "document_type": "Directive",
        "document_summary": "summary",
        "requirements": bundle.model_dump(),
        "evidence_cards": [
            {
                "origin": "document",
                "claim": "Claim",
                "source": "doc.pdf",
                "page": 1,
                "chunk_id": "c1",
                "certainty": "high",
            }
        ],
    }

    refined = asyncio.run(runner._requirements_refinement_node(state))
    assert refined["requirements"]["document"] == "doc.pdf"

    merged = asyncio.run(runner.merge_requirements([bundle], ["doc.pdf"]))
    assert merged.document == "doc.pdf"


def test_async_rate_limiter_no_delay():
    limiter = AsyncRateLimiter(0)
    asyncio.run(limiter.wait())
