from __future__ import annotations

import json
from typing import Dict, List, TypedDict

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, StateGraph

from AgenticAI.agentic.models import (
    ContextAssessment,
    QuerySpecification,
    RequirementBundle,
    RetrievalChunk,
    WebResult,
)
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
    ):
        self.query_model = query_model
        self.requirements_model = requirements_model
        self.mcp_client = mcp_client
        self.retrieval_top_k = retrieval_top_k
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
            for row in data.get("results", []):
                chunk = RetrievalChunk.model_validate(row)
                retrieved_chunks.append(chunk.model_dump())
        return {"retrieval_results": retrieved_chunks}

    async def _context_node(self, state: AgenticState) -> AgenticState:
        parser = PydanticOutputParser(pydantic_object=ContextAssessment)
        format_instructions = parser.get_format_instructions()
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are the context assessor for Agent 1. Evaluate whether the retrieved chunks alone let a bank draft precise requirements."
                    " When gaps remain (e.g., unclear thresholds, missing ESRS references, external timelines), recommend up to three follow-up open-web queries."
                    " Remember you can call the MCP tool `web_search` which returns both metadata and full-page text, so phrase queries that will surface high-signal regulatory explainers."
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

        web_context: List[Dict] = []
        if assessment.needs_additional_context:
            for query in assessment.missing_information_queries[:3]:
                payload = await self.mcp_client.call_tool("web_search", {"query": query, "num_results": 5})
                data = json.loads(payload)
                for row in data.get("results", []):
                    web_result = WebResult.model_validate(row)
                    web_context.append(web_result.model_dump())

        return {"web_context": web_context}

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
        return {"requirements": bundle.model_dump()}

    async def arun(self, state: AgenticState) -> AgenticState:
        return await self.graph.ainvoke(state)

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
