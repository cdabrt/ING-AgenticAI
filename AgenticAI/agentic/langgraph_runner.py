from __future__ import annotations

import json
from typing import Any, Dict, List, Tuple, TypedDict, cast

from langchain_core.exceptions import OutputParserException
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, StateGraph

from AgenticAI.agentic.models import (
    ContextAssessment,
    QuerySpecification,
    RequirementBundle,
    RetrievalChunk,
    WebContentDecision,
    WebResult,
    WebSelectionResponse,
    WebSourceCandidate,
    WebSourceSelection,
)
from AgenticAI.agentic.decision_logger import DecisionLogger
from AgenticAI.mcp.client import MCPToolClient


class AgenticState(TypedDict, total=False):
    doc_source: str
    doc_text: str
    doc_headings: List[str]
    document_summary: str
    document_type: str
    queries: List[str]
    retrieval_results: List[Dict]
    web_context: List[Dict]
    requirements: Dict


class AgenticGraphRunner:
    def __init__(
        self,
        query_model: ChatGoogleGenerativeAI,
        requirements_model: ChatGoogleGenerativeAI,
        mcp_client: MCPToolClient,
        retrieval_top_k: int = 8,
        decision_logger: DecisionLogger | None = None,
    ):
        self.query_model = query_model
        self.requirements_model = requirements_model
        self.mcp_client = mcp_client
        self.retrieval_top_k = retrieval_top_k
        self.decision_logger = decision_logger
        self.graph = self._build_graph()

    def _build_graph(self):
        workflow = StateGraph(AgenticState)
        workflow.add_node("query_agent", self._query_agent_node)
        workflow.add_node("retrieval", self._retrieval_node)
        workflow.add_node("context", self._context_node)
        workflow.add_node("requirements", self._requirements_node)

        workflow.set_entry_point("query_agent")
        workflow.add_edge("query_agent", "retrieval")
        workflow.add_edge("retrieval", "context")
        workflow.add_edge("context", "requirements")
        workflow.add_edge("requirements", END)

        return workflow.compile()

    async def _query_agent_node(self, state: AgenticState) -> AgenticState:
        parser = PydanticOutputParser(pydantic_object=QuerySpecification)
        format_instructions = parser.get_format_instructions()
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are Agent 1, a regulatory discovery strategist supporting a large publicly listed EU bank."
                    " Study the provided directive/policy text, extract its dominant themes (scope, timelines, assurance, data needs),"
                    " and craft sharply focused retrieval queries that will surface the smallest set of chunks needed to understand implementation obligations."
                    " Return JSON matching the schema described in the instructions and make the summary actionable for a financial-services audience.",
                ),
                (
                    "human",
                    "Document name: {doc_source}\n\nHeadings: {headings}\n\nContent:\n{doc_text}\n\n"
                    "{format_instructions}",
                ),
            ]
        )
        chain = prompt | self.query_model | parser
        headings_text = " | ".join(state.get("doc_headings", []))
        result: QuerySpecification = await chain.ainvoke(
            {
                "doc_source": state.get("doc_source", "unknown document"),
                "headings": headings_text,
                "doc_text": state.get("doc_text", ""),
                "format_instructions": format_instructions,
            }
        )
        await self._log_event(
            "query_agent",
            {
                "doc_source": state.get("doc_source"),
                "document_type": result.document_type,
                "queries": result.queries,
                "summary": result.summary,
            },
        )
        return {
            "queries": result.queries,
            "document_summary": result.summary,
            "document_type": result.document_type,
        }

    async def _retrieval_node(self, state: AgenticState) -> AgenticState:
        retrieved_chunks: List[Dict] = []
        for query in state.get("queries", []):
            payload = await self.mcp_client.call_tool("retrieve_chunks", {"query": query, "top_k": self.retrieval_top_k})
            data = json.loads(payload)
            deduped_results = self._dedupe_chunks(data.get("results", []))
            chunk_ids: List[str] = []
            for row in deduped_results:
                chunk = RetrievalChunk.model_validate(row)
                retrieved_chunks.append(chunk.model_dump())
                chunk_ids.append(chunk.chunk_id)
            await self._log_event(
                "retrieval",
                {
                    "doc_source": state.get("doc_source"),
                    "query": query,
                    "result_count": len(chunk_ids),
                    "chunk_ids": chunk_ids,
                },
            )
        return {"retrieval_results": retrieved_chunks}

    @staticmethod
    def _dedupe_chunks(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Collapse duplicate chunks from the same source/page while keeping the best score."""

        best_by_location: Dict[Tuple[Any, Any], Dict[str, Any]] = {}
        order: List[Tuple[Any, Any]] = []

        for row in rows:
            key = (row.get("source"), row.get("page"))
            current_best = best_by_location.get(key)
            if current_best is None:
                best_by_location[key] = row
                order.append(key)
                continue

            if row.get("score", 0) > current_best.get("score", 0):
                best_by_location[key] = row

        return [best_by_location[key] for key in order]

    async def _context_node(self, state: AgenticState) -> AgenticState:
        parser = PydanticOutputParser(pydantic_object=ContextAssessment)
        format_instructions = parser.get_format_instructions()
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are the context assessor for Agent 1. Evaluate whether the retrieved chunks alone let a bank draft precise requirements."
                    " When gaps remain (e.g., unclear thresholds, missing ESRS references, external timelines), recommend up to three follow-up open-web queries."
                    " Remember you can call the MCP tools `web_search` (metadata only) and `fetch_web_page`"
                    " (for shortlisted URLs) so phrase queries that surface high-signal regulatory explainers."
                    " Respond strictly with the JSON schema described in the instructions and justify why more context is or is not needed.",
                ),
                (
                    "human",
                    "Document summary: {summary}\nDocument type: {doc_type}\nRetrieved context count: {ctx_count}\n"
                    "{format_instructions}",
                ),
            ]
        )
        chain = prompt | self.query_model | parser
        assessment: ContextAssessment = await chain.ainvoke(
            {
                "summary": state.get("document_summary", ""),
                "doc_type": state.get("document_type", "unknown"),
                "ctx_count": len(state.get("retrieval_results", [])),
                "format_instructions": format_instructions,
            }
        )
        await self._log_event(
            "context_assessment",
            {
                "doc_source": state.get("doc_source"),
                "document_type": state.get("document_type"),
                "retrieval_count": len(state.get("retrieval_results", [])),
                "needs_additional_context": assessment.needs_additional_context,
                "missing_queries": assessment.missing_information_queries,
                "explanation": assessment.explanation,
            },
        )

        web_context: List[Dict] = []
        if assessment.needs_additional_context:
            for query in assessment.missing_information_queries[:3]:
                payload = await self.mcp_client.call_tool(
                    "web_search",
                    {"query": query, "num_results": 5, "include_content": False},
                )
                data = json.loads(payload)
                raw_candidates = data.get("results", [])
                candidates: List[WebSourceCandidate] = []
                for idx, row in enumerate(raw_candidates, start=1):
                    href = row.get("href")
                    if not href:
                        continue
                    candidates.append(
                        WebSourceCandidate(
                            identifier=f"result_{idx}",
                            title=row.get("title"),
                            snippet=row.get("snippet"),
                            href=href,
                        )
                    )

                if not candidates:
                    continue

                selections = await self._screen_web_candidates(
                    query=query,
                    document_summary=state.get("document_summary", ""),
                    document_type=state.get("document_type", "unknown"),
                    candidates=candidates,
                )
                await self._log_event(
                    "web_candidate_screen",
                    {
                        "doc_source": state.get("doc_source"),
                        "query": query,
                        "candidates": [cand.model_dump() for cand in candidates],
                        "decisions": [selection.model_dump() for selection in selections],
                    },
                )
                candidate_map = {candidate.identifier: candidate for candidate in candidates}

                for decision in selections:
                    if not decision.fetch:
                        continue

                    candidate = candidate_map.get(decision.identifier)
                    if not candidate or not candidate.href:
                        continue

                    page_payload = await self.mcp_client.call_tool("fetch_web_page", {"url": candidate.href})
                    page_data = json.loads(page_payload)
                    content = page_data.get("content", "")

                    content_decision = await self._evaluate_web_content(
                        query=query,
                        candidate=candidate,
                        content=content,
                        document_summary=state.get("document_summary", ""),
                        document_type=state.get("document_type", "unknown"),
                    )

                    if not content_decision.include:
                        await self._log_event(
                            "web_content_rejected",
                            {
                                "doc_source": state.get("doc_source"),
                                "query": query,
                                "url": candidate.href,
                                "rationale": content_decision.rationale,
                                "summary": content_decision.summary,
                            },
                        )
                        continue

                    result = WebResult(
                        title=candidate.title,
                        snippet=candidate.snippet,
                        href=candidate.href,
                        content=content,
                        selection_reason=decision.rationale,
                        inclusion_reason=content_decision.rationale,
                        summary=content_decision.summary,
                    )
                    await self._log_event(
                        "web_content_accepted",
                        {
                            "doc_source": state.get("doc_source"),
                            "query": query,
                            "url": candidate.href,
                            "summary": content_decision.summary,
                            "selection_reason": decision.rationale,
                            "inclusion_reason": content_decision.rationale,
                        },
                    )
                    web_context.append(result.model_dump())

        return {"web_context": web_context}

    async def _screen_web_candidates(
        self,
        *,
        query: str,
        document_summary: str,
        document_type: str,
        candidates: List[WebSourceCandidate],
    ) -> List[WebSourceSelection]:
        parser = PydanticOutputParser(pydantic_object=WebSelectionResponse)
        format_instructions = parser.get_format_instructions()
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are triaging open-web search hits for a regulatory analyst."
                    " Only approve fetching pages likely to contain official or well-sourced regulatory insight"
                    " beyond the provided document. Keep the shortlist lean and justify every approval or rejection.",
                ),
                (
                    "human",
                    "Document summary: {summary}\nDocument type: {doc_type}\nQuery: {query}\n"
                    "Candidates (JSON list): {candidates}\n\n{format_instructions}",
                ),
            ]
        )
        chain = prompt | self.query_model | parser
        summary_snippet = (document_summary or "")[:80]
        try:
            response: WebSelectionResponse = await chain.ainvoke(
                {
                    "summary": document_summary,
                    "doc_type": document_type,
                    "query": query,
                    "candidates": json.dumps([cand.model_dump() for cand in candidates]),
                    "format_instructions": format_instructions,
                }
            )
            return response.selections or []
        except OutputParserException as exc:
            await self._log_event(
                "web_candidate_screen_error",
                {
                    "doc_source": summary_snippet,
                    "query": query,
                    "error": str(exc),
                },
            )
        except Exception as exc:  # noqa: BLE001 - keep pipeline resilient
            await self._log_event(
                "web_candidate_screen_error",
                {
                    "doc_source": summary_snippet,
                    "query": query,
                    "error": f"Unexpected failure: {exc}",
                },
            )

        # Default to rejecting all candidates when parsing fails to avoid pipeline crashes.
        return [
            WebSourceSelection(
                identifier=candidate.identifier,
                fetch=False,
                rationale="Skipped: unable to parse selection response",
            )
            for candidate in candidates
        ]

    async def _evaluate_web_content(
        self,
        *,
        query: str,
        candidate: WebSourceCandidate,
        content: str,
        document_summary: str,
        document_type: str,
    ) -> WebContentDecision:
        parser = PydanticOutputParser(pydantic_object=WebContentDecision)
        format_instructions = parser.get_format_instructions()
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are vetting fetched web pages for a regulatory requirements agent."
                    " Include the page only if it adds concrete obligations, thresholds, or authoritative context"
                    " that complements the base document. Provide a focused summary when you approve it.",
                ),
                (
                    "human",
                    "Document summary: {summary}\nDocument type: {doc_type}\nQuery: {query}\n"
                    "URL: {url}\nTitle: {title}\nSnippet: {snippet}\nContent: {content}\n\n{format_instructions}",
                ),
            ]
        )
        chain = prompt | self.query_model | parser
        decision: WebContentDecision = await chain.ainvoke(
            {
                "summary": document_summary,
                "doc_type": document_type,
                "query": query,
                "url": candidate.href or "unknown",
                "title": candidate.title or "",
                "snippet": candidate.snippet or "",
                "content": content,
                "format_instructions": format_instructions,
            }
        )
        return decision

    async def _requirements_node(self, state: AgenticState) -> AgenticState:
        parser = PydanticOutputParser(pydantic_object=RequirementBundle)
        format_instructions = parser.get_format_instructions()
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are Agent 2, a senior regulatory implementation architect for a major European bank."
                    " Use the retrieved document chunks (and any enriched web context) to translate legal obligations into actionable business and data requirements."
                    " Address governance, risk management, assurance, reporting formats, and data collection duties relevant to lending, investment, and underwriting activities."
                    " Cite chunk IDs/pages for every document-driven statement and include online citations whenever external context informed the requirement."
                    " Format every entry in `document_sources` exactly as `{{source}}, page {{page_number}}, chunk {{chunk_id}}` using the values from the provided retrieval chunks."
                    " If a page number is missing, write `page unknown` but always keep the chunk identifier in the same format."
                    " Respond strictly with the JSON schema described in the instructions so downstream systems can ingest your output.",
                ),
                (
                    "human",
                    "Document: {doc}\nType: {doc_type}\nSummary: {summary}\n\nContext from retrieval: {chunks}\n\n"
                    "Online insights: {web}\n\n{format_instructions}",
                ),
            ]
        )
        chain = prompt | self.requirements_model | parser

        chunk_text = json.dumps(state.get("retrieval_results", []))
        web_text = json.dumps(state.get("web_context", []))

        bundle: RequirementBundle = await chain.ainvoke(
            {
                "doc": state.get("doc_source"),
                "doc_type": state.get("document_type", "unknown"),
                "summary": state.get("document_summary", ""),
                "chunks": chunk_text,
                "web": web_text,
                "format_instructions": format_instructions,
            }
        )
        await self._log_event(
            "requirements",
            {
                "doc_source": state.get("doc_source"),
                "document_type": state.get("document_type"),
                "business_requirement_count": len(bundle.business_requirements),
                "data_requirement_count": len(bundle.data_requirements),
                "assumption_count": len(bundle.assumptions),
            },
        )
        return {"requirements": bundle.model_dump()}

    async def arun(self, state: AgenticState) -> AgenticState:
        result = await self.graph.ainvoke(state)
        return cast(AgenticState, result)

    async def process_document(self, state: AgenticState) -> AgenticState:
        """Run query, retrieval, and context nodes for a single document."""

        query_state = await self._query_agent_node(state)
        state.update(query_state)

        retrieval_state = await self._retrieval_node(state)
        state.update(retrieval_state)

        context_state = await self._context_node(state)
        state.update(context_state)

        return state

    async def generate_requirements(self, state: AgenticState) -> AgenticState:
        """Invoke only the requirements node with the provided state."""

        return await self._requirements_node(state)

    async def _log_event(self, event_type: str, payload: Dict[str, Any]) -> None:
        if not self.decision_logger:
            return
        await self.decision_logger.log({"event_type": event_type, **payload})
