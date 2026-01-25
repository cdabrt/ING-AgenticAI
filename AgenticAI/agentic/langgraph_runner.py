from __future__ import annotations

import asyncio
import json
import os
import random
import re
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, TypedDict, cast

from langchain_core.exceptions import OutputParserException
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, StateGraph

from AgenticAI.agentic.models import (
    ContextAssessment,
    EvidenceCard,
    EvidenceDigest,
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
    evidence_cards: List[Dict]
    requirements: Dict


class AsyncRateLimiter:
    def __init__(self, rpm: float):
        self.min_interval = 60.0 / rpm if rpm and rpm > 0 else 0.0
        self._lock = asyncio.Lock()
        self._last_call = 0.0

    async def wait(self) -> None:
        if self.min_interval <= 0:
            return
        async with self._lock:
            now = time.monotonic()
            elapsed = now - self._last_call
            delay = self.min_interval - elapsed
            if delay > 0:
                await asyncio.sleep(delay)
            self._last_call = time.monotonic()


class AgenticGraphRunner:
    def __init__(
        self,
        query_model: BaseChatModel,
        requirements_model: BaseChatModel,
        mcp_client: MCPToolClient,
        retrieval_top_k: int = 8,
        web_search_enabled: bool | None = None,
        max_web_queries_per_doc: int | None = None,
        decision_logger: DecisionLogger | None = None,
        throttle_enabled: bool = True,
        llm_status_callback: Optional[Callable[[Dict[str, Optional[str] | int]], None]] = None,
    ):
        self.query_model = query_model
        self.requirements_model = requirements_model
        self.mcp_client = mcp_client
        self.retrieval_top_k = retrieval_top_k
        self.web_search_enabled = self._resolve_web_search_enabled(web_search_enabled)
        self.max_web_queries_per_doc = self._resolve_web_search_limit(max_web_queries_per_doc)
        self.decision_logger = decision_logger
        self.throttle_enabled = throttle_enabled
        self.query_rate_limiter = AsyncRateLimiter(self._get_query_rpm())
        self.requirements_rate_limiter = AsyncRateLimiter(self._get_requirements_rpm())
        self.graph = self._build_graph()
        self.llm_status_callback = llm_status_callback

    @staticmethod
    def _parse_rpm(value: str | None, default: float) -> float:
        if not value:
            return default
        try:
            rpm = float(value)
        except ValueError:
            return default
        return rpm if rpm > 0 else 0.0

    @staticmethod
    def _parse_bool(value: str | None) -> Optional[bool]:
        if value is None:
            return None
        lowered = value.strip().lower()
        if lowered in ("1", "true", "yes", "y", "on"):
            return True
        if lowered in ("0", "false", "no", "n", "off"):
            return False
        return None

    def _resolve_web_search_enabled(self, value: bool | str | None) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            parsed = self._parse_bool(value)
            if parsed is not None:
                return parsed
        env_value = self._parse_bool(os.getenv("WEB_SEARCH_ENABLED"))
        if env_value is None:
            return True
        return env_value

    @staticmethod
    def _resolve_web_search_limit(value: int | str | None) -> int:
        if isinstance(value, int):
            return max(0, value)
        if isinstance(value, str):
            try:
                return max(0, int(value))
            except ValueError:
                pass
        env_value = os.getenv("WEB_SEARCH_MAX_QUERIES_PER_DOC", "3")
        try:
            return max(0, int(env_value))
        except ValueError:
            return 3

    def _get_query_rpm(self) -> float:
        if not self.throttle_enabled:
            return 0.0
        default_rpm = self._parse_rpm(
            os.getenv("GEMINI_RPM_LIMIT") or os.getenv("GEMINI_REQUESTS_PER_MINUTE"),
            8.0,
        )
        return self._parse_rpm(os.getenv("QUERY_MODEL_RPM"), default_rpm)

    def _get_requirements_rpm(self) -> float:
        if not self.throttle_enabled:
            return 0.0
        default_rpm = self._parse_rpm(
            os.getenv("GEMINI_RPM_LIMIT") or os.getenv("GEMINI_REQUESTS_PER_MINUTE"),
            8.0,
        )
        return self._parse_rpm(os.getenv("REQUIREMENTS_MODEL_RPM"), default_rpm)

    @staticmethod
    def _parse_retry_delay(message: str) -> float | None:
        match = re.search(r"retryDelay['\"]?: ['\"]?([0-9.]+)s", message)
        if match:
            return float(match.group(1))
        match = re.search(r"retry in ([0-9.]+)s", message)
        if match:
            return float(match.group(1))
        return None

    @staticmethod
    def _is_daily_quota(message: str) -> bool:
        lowered = message.lower()
        return "perday" in lowered or "per day" in lowered or "requestsperday" in lowered

    async def _ainvoke_with_retry(
        self,
        *,
        chain,
        payload: Dict[str, Any],
        limiter: AsyncRateLimiter,
        model_label: str,
        max_retries: int,
    ):
        attempt = 0
        while True:
            await limiter.wait()
            try:
                return await chain.ainvoke(payload)
            except Exception as exc:  # noqa: BLE001 - surface quota errors
                message = str(exc)
                if "RESOURCE_EXHAUSTED" not in message and "429" not in message:
                    raise

                if self._is_daily_quota(message):
                    raise RuntimeError(
                        f"Quota exceeded for {model_label}. Daily limit reached; try again later or use another key."
                    ) from exc

                retry_delay = self._parse_retry_delay(message)
                if retry_delay is None:
                    retry_delay = min(60.0, 2.0 ** min(attempt, 5))
                retry_delay += random.uniform(0.2, 0.8)

                await self._log_event(
                    "llm_backoff",
                    {
                        "model": model_label,
                        "attempt": attempt + 1,
                        "retry_delay_seconds": round(retry_delay, 2),
                        "error": message,
                    },
                )

                attempt += 1
                if attempt > max_retries:
                    raise RuntimeError(
                        f"Quota exceeded for {model_label}. Retry limit reached; wait before trying again."
                    ) from exc
                await asyncio.sleep(retry_delay)

    def _build_graph(self):
        workflow = StateGraph(AgenticState)
        workflow.add_node("query_agent", self._query_agent_node)
        workflow.add_node("retrieval", self._retrieval_node)
        workflow.add_node("context", self._context_node)
        workflow.add_node("evidence_distill", self._distill_evidence_node)
        workflow.add_node("requirements", self._requirements_node)
        workflow.add_node("requirements_refinement", self._requirements_refinement_node)

        workflow.set_entry_point("query_agent")
        workflow.add_edge("query_agent", "retrieval")
        workflow.add_edge("retrieval", "context")
        workflow.add_edge("context", "evidence_distill")
        workflow.add_edge("evidence_distill", "requirements")
        workflow.add_edge("requirements", "requirements_refinement")
        workflow.add_edge("requirements_refinement", END)

        return workflow.compile()

    @staticmethod
    def _build_structured_chain(model: BaseChatModel, schema):
        try:
            return model.with_structured_output(schema, include_raw=True)
        except Exception:
            return None

    @staticmethod
    def _sanitize_json_text(text: str) -> str:
        if not text:
            return text
        cleaned = text.strip()
        cleaned = re.sub(r",\s*([}\]])", r"\1", cleaned)
        return cleaned

    @classmethod
    def _extract_json_payload(cls, text: str) -> str | None:
        if not text:
            return None
        if "```" in text:
            fence_start = text.find("```")
            fence_end = text.find("```", fence_start + 3)
            if fence_end != -1:
                fenced_body = text[fence_start + 3:fence_end]
                brace_start = fenced_body.find("{")
                brace_end = fenced_body.rfind("}")
                if brace_start != -1 and brace_end != -1 and brace_end > brace_start:
                    return cls._sanitize_json_text(fenced_body[brace_start:brace_end + 1])
        fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL | re.IGNORECASE)
        if fenced:
            return cls._sanitize_json_text(fenced.group(1))
        decoder = json.JSONDecoder()
        best_payload: str | None = None
        best_len = 0
        for match in re.finditer(r"{", text):
            start = match.start()
            try:
                _, end = decoder.raw_decode(text[start:])
            except json.JSONDecodeError:
                continue
            candidate = cls._sanitize_json_text(text[start:start + end])
            if len(candidate) > best_len:
                best_payload = candidate
                best_len = len(candidate)
        if best_payload:
            return best_payload
        last_open = text.rfind("{")
        last_close = text.rfind("}")
        if last_open != -1 and last_close != -1 and last_close > last_open:
            return cls._sanitize_json_text(text[last_open:last_close + 1])
        return None

    @staticmethod
    def _extract_message_text(response: Any) -> str:
        def _flatten(value: Any) -> List[str]:
            if isinstance(value, str):
                return [value]
            if isinstance(value, list):
                parts: List[str] = []
                for item in value:
                    parts.extend(_flatten(item))
                return parts
            if isinstance(value, dict):
                value_type = value.get("type")
                if "encrypted_content" in value:
                    return []
                if isinstance(value_type, str) and value_type.lower() in ("reasoning", "thinking", "analysis"):
                    return []
                if isinstance(value.get("text"), str):
                    return [value["text"]]
                if "content" in value:
                    return _flatten(value.get("content"))
                return []
            return [str(value)]

        content = response.content if hasattr(response, "content") else response
        return "".join(_flatten(content))

    def _emit_llm_status(
        self,
        *,
        step: str,
        doc_source: Optional[str] = None,
        detail: Optional[str] = None,
        increment: int = 1,
    ) -> None:
        if not self.llm_status_callback:
            return
        payload: Dict[str, Optional[str] | int] = {
            "llm_step": step,
            "llm_detail": detail,
            "current_doc": doc_source,
            "llm_increment": increment,
        }
        self.llm_status_callback(payload)

    async def _coerce_model_output(
        self,
        parsed: Any,
        schema,
        *,
        parse_label: str,
        log_context: Optional[Dict[str, Any]] = None,
    ):
        if isinstance(parsed, schema):
            return parsed
        if isinstance(parsed, str):
            recovered = self._extract_json_payload(parsed) or parsed
            recovered = self._sanitize_json_text(recovered)
            try:
                return schema.model_validate_json(recovered)
            except Exception:
                try:
                    return schema.model_validate(json.loads(recovered, strict=False))
                except Exception as exc:
                    await self._log_event(
                        "structured_output_parse_error",
                        {
                            "stage": parse_label,
                            "error": str(exc),
                            "response_excerpt": recovered[:1200],
                            **(log_context or {}),
                        },
                    )
                    raise
        try:
            return schema.model_validate(parsed)
        except Exception as exc:
            await self._log_event(
                "structured_output_parse_error",
                {
                    "stage": parse_label,
                    "error": str(exc),
                    "response_excerpt": str(parsed)[:1200],
                    **(log_context or {}),
                },
            )
            raise

    async def _ainvoke_with_parser(
        self,
        *,
        chain,
        parser: PydanticOutputParser,
        payload: Dict[str, Any],
        limiter: AsyncRateLimiter,
        model_label: str,
        max_retries: int,
        parse_label: str,
        log_context: Optional[Dict[str, Any]] = None,
    ):
        response = await self._ainvoke_with_retry(
            chain=chain,
            payload=payload,
            limiter=limiter,
            model_label=model_label,
            max_retries=max_retries,
        )
        content = self._extract_message_text(response)
        try:
            return parser.parse(content)
        except OutputParserException as exc:
            sanitized_content = self._sanitize_json_text(content)
            if sanitized_content != content:
                try:
                    return parser.parse(sanitized_content)
                except OutputParserException:
                    pass
            recovered = self._extract_json_payload(content)
            if recovered:
                recovered = self._sanitize_json_text(recovered)
                try:
                    return parser.parse(recovered)
                except OutputParserException:
                    try:
                        raw = json.loads(recovered, strict=False)
                        return parser.parse_obj(raw)
                    except Exception:
                        pass
            log_payload = {
                "model": model_label,
                "stage": parse_label,
                "error": str(exc),
                "response_excerpt": content[:1200],
            }
            if log_context:
                log_payload.update(log_context)
            await self._log_event("output_parse_error", log_payload)
            raise

    async def _query_agent_node(self, state: AgenticState) -> AgenticState:
        parser = PydanticOutputParser(pydantic_object=QuerySpecification)
        format_instructions = parser.get_format_instructions()
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are Agent 1, a regulatory discovery strategist supporting ING Bank (a large publicly listed EU bank)."
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
        chain = prompt | self.query_model
        headings_text = " | ".join(state.get("doc_headings", []))
        result: QuerySpecification = await self._ainvoke_with_parser(
            chain=chain,
            parser=parser,
            payload={
                "doc_source": state.get("doc_source", "unknown document"),
                "headings": headings_text,
                "doc_text": state.get("doc_text", ""),
                "format_instructions": format_instructions,
            },
            limiter=self.query_rate_limiter,
            model_label="query_model",
            max_retries=int(os.getenv("GEMINI_MAX_RETRIES", "3")),
            parse_label="query_agent",
            log_context={"doc_source": state.get("doc_source")},
        )
        self._emit_llm_status(step="query_agent", doc_source=state.get("doc_source"))
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

    @staticmethod
    def _normalize_text(text: str) -> str:
        return " ".join(text.split())

    @staticmethod
    def _trim_text(text: str, max_chars: int) -> str:
        if max_chars <= 0 or len(text) <= max_chars:
            return text
        return text[:max_chars].rstrip() + "..."

    @classmethod
    def _prepare_chunks_for_distillation(
        cls,
        rows: List[Dict[str, Any]],
        max_chunk_chars: int,
    ) -> List[Dict[str, Any]]:
        prepared: List[Dict[str, Any]] = []
        for row in rows:
            raw_text = row.get("text") or ""
            normalized = cls._normalize_text(raw_text)
            trimmed = cls._trim_text(normalized, max_chunk_chars)
            prepared.append(
                {
                    "chunk_id": row.get("chunk_id"),
                    "source": row.get("source"),
                    "page": row.get("page"),
                    "parent_heading": row.get("parent_heading"),
                    "text": trimmed,
                }
            )
        return prepared

    @staticmethod
    def _batch_by_char_budget(rows: List[Dict[str, Any]], char_budget: int) -> List[List[Dict[str, Any]]]:
        if not rows:
            return []
        if char_budget <= 0:
            return [rows]
        batches: List[List[Dict[str, Any]]] = []
        current: List[Dict[str, Any]] = []
        current_size = 0
        for row in rows:
            text_size = len(row.get("text") or "")
            entry_size = text_size + 200
            if current and current_size + entry_size > char_budget:
                batches.append(current)
                current = []
                current_size = 0
            current.append(row)
            current_size += entry_size
        if current:
            batches.append(current)
        return batches

    @staticmethod
    def _split_batches_by_count(batches: List[List[Dict[str, Any]]], max_chunks: int) -> List[List[Dict[str, Any]]]:
        if max_chunks <= 0:
            return batches
        split: List[List[Dict[str, Any]]] = []
        for batch in batches:
            if len(batch) <= max_chunks:
                split.append(batch)
                continue
            for start in range(0, len(batch), max_chunks):
                split.append(batch[start:start + max_chunks])
        return split

    @staticmethod
    def _should_split_distill_error(exc: Exception) -> bool:
        message = str(exc)
        indicators = (
            "Invalid JSON",
            "EOF while parsing",
            "JSONDecodeError",
            "model_validate_json",
            "parse",
            "ValidationError",
        )
        return any(indicator in message for indicator in indicators)

    @staticmethod
    def _first_sentence(text: str) -> str:
        if not text:
            return ""
        for sep in (". ", "? ", "! "):
            if sep in text:
                return text.split(sep, 1)[0].strip() + sep.strip()
        return text.strip()

    def _fallback_evidence_cards_from_chunks(self, batch: List[Dict[str, Any]]) -> List[EvidenceCard]:
        cards: List[EvidenceCard] = []
        for row in batch:
            raw_text = self._normalize_text(row.get("text") or "")
            summary = self._trim_text(self._first_sentence(raw_text), 240)
            support = self._trim_text(raw_text, 360)
            cards.append(
                EvidenceCard(
                    origin="document",
                    claim=summary or support or "Evidence extracted from document.",
                    source=row.get("source") or "document",
                    page=row.get("page"),
                    chunk_id=row.get("chunk_id"),
                    certainty="low",
                    support=support or None,
                )
            )
        return cards

    @staticmethod
    def _dedupe_evidence_cards(cards: List[EvidenceCard]) -> List[EvidenceCard]:
        seen: set[tuple] = set()
        deduped: List[EvidenceCard] = []
        for card in cards:
            key = (
                card.origin,
                card.source,
                card.page,
                card.chunk_id,
                card.claim.strip().lower(),
            )
            if key in seen:
                continue
            seen.add(key)
            deduped.append(card)
        return deduped

    @staticmethod
    def _web_context_to_evidence(web_context: List[Dict[str, Any]]) -> List[EvidenceCard]:
        cards: List[EvidenceCard] = []
        for row in web_context:
            summary = (row.get("summary") or "").strip()
            if not summary:
                continue
            cards.append(
                EvidenceCard(
                    origin="web",
                    claim=summary,
                    source=row.get("href") or row.get("title") or "web",
                    certainty="medium",
                    support=(row.get("snippet") or "").strip() or None,
                )
            )
        return cards

    async def _context_node(
        self,
        state: AgenticState,
        status_callback: Optional[Callable[[str, float], None]] = None,
    ) -> AgenticState:
        retrieval_chunks = state.get("retrieval_results", [])
        # Pass a trimmed JSON payload so the assessor can inspect actual snippets without blowing up the prompt.
        chunk_preview = json.dumps(retrieval_chunks[:20]) if retrieval_chunks else "[]"
        parser = PydanticOutputParser(pydantic_object=ContextAssessment)
        format_instructions = parser.get_format_instructions()
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are the context assessor for Agent 1 at ING Bank. Evaluate whether the retrieved chunks alone let a bank draft precise requirements."
                    " When gaps remain (e.g., unclear thresholds, missing ESRS references, external timelines), recommend up to three follow-up open-web queries."
                    " You will receive the actual retrieved chunks (JSON objects with source/page/chunk_id/text) so ground your judgment in what is already available, and ask for more meaningful context if needed."
                    " Remember you can call the MCP tools `web_search` (metadata only) and `fetch_web_page`"
                    " (for shortlisted URLs) so phrase queries that surface high-signal regulatory explainers."
                    " Respond strictly with the JSON schema described in the instructions and justify why more context is or is not needed.",
                ),
                (
                    "human",
                    "Document summary: {summary}\nDocument type: {doc_type}\nRetrieved context count: {ctx_count}\nRetrieved context (JSON list): {chunks}\n"
                    "{format_instructions}",
                ),
            ]
        )
        chain = prompt | self.query_model
        assessment: ContextAssessment = await self._ainvoke_with_parser(
            chain=chain,
            parser=parser,
            payload={
                "summary": state.get("document_summary", ""),
                "doc_type": state.get("document_type", "unknown"),
                "ctx_count": len(state.get("retrieval_results", [])),
                "chunks": chunk_preview,
                "format_instructions": format_instructions,
            },
            limiter=self.query_rate_limiter,
            model_label="query_model",
            max_retries=int(os.getenv("GEMINI_MAX_RETRIES", "3")),
            parse_label="context_assessment",
            log_context={"doc_source": state.get("doc_source")},
        )
        self._emit_llm_status(step="context_assessment", doc_source=state.get("doc_source"))
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
        if assessment.needs_additional_context and self.web_search_enabled and self.max_web_queries_per_doc > 0:
            if status_callback:
                status_callback("Shortlisting web sources", 0.85)
            for query in assessment.missing_information_queries[: self.max_web_queries_per_doc]:
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
                    doc_source=state.get("doc_source"),
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

                    if status_callback:
                        status_callback("Fetching web sources", 0.9)
                    page_payload = await self.mcp_client.call_tool("fetch_web_page", {"url": candidate.href})
                    page_data = json.loads(page_payload)
                    content = page_data.get("content", "")

                    content_decision = await self._evaluate_web_content(
                        query=query,
                        candidate=candidate,
                        content=content,
                        document_summary=state.get("document_summary", ""),
                        document_type=state.get("document_type", "unknown"),
                        doc_source=state.get("doc_source"),
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
        elif assessment.needs_additional_context and not self.web_search_enabled:
            await self._log_event(
                "web_search_skipped",
                {
                    "doc_source": state.get("doc_source"),
                    "reason": "disabled",
                    "missing_queries": assessment.missing_information_queries,
                },
            )
        elif assessment.needs_additional_context and self.max_web_queries_per_doc <= 0:
            await self._log_event(
                "web_search_skipped",
                {
                    "doc_source": state.get("doc_source"),
                    "reason": "max_queries_zero",
                    "missing_queries": assessment.missing_information_queries,
                },
            )

        return {"web_context": web_context}

    async def _distill_evidence_node(
        self,
        state: AgenticState,
        status_callback: Optional[Callable[[str, float], None]] = None,
    ) -> AgenticState:
        retrieval_chunks = state.get("retrieval_results", [])
        web_context = state.get("web_context", [])
        if not retrieval_chunks and not web_context:
            return {"evidence_cards": []}

        char_budget = int(os.getenv("EVIDENCE_DISTILL_CHAR_BUDGET", "12000"))
        max_chunk_chars = int(os.getenv("EVIDENCE_DISTILL_MAX_CHUNK_CHARS", "4000"))
        max_chunks_per_batch = int(os.getenv("EVIDENCE_DISTILL_MAX_CHUNKS_PER_BATCH", "0"))

        prepared_chunks = self._prepare_chunks_for_distillation(retrieval_chunks, max_chunk_chars)
        batches = self._batch_by_char_budget(prepared_chunks, char_budget)
        batches = self._split_batches_by_count(batches, max_chunks_per_batch)

        parser = PydanticOutputParser(pydantic_object=EvidenceDigest)
        format_instructions = parser.get_format_instructions()
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are an evidence distiller for regulatory compliance."
                    " Extract atomic obligation statements from each chunk."
                    " Each card must be grounded in the provided chunk text."
                    " Return only the claims that represent concrete obligations, thresholds, timelines, or reporting duties."
                    " Keep each claim to one short sentence."
                    " Set `origin` to `document` for every card."
                    " Use certainty tags: high, medium, or low.",
                ),
                (
                    "human",
                    "Document: {doc}\nDocument type: {doc_type}\n"
                    "Chunks (JSON list): {chunks}\n\n{format_instructions}",
                ),
            ]
        )
        structured = self._build_structured_chain(self.query_model, EvidenceDigest)
        structured_chain = prompt | structured if structured else None
        base_chain = prompt | self.query_model

        distilled_cards: List[EvidenceCard] = []
        total_batches = max(1, len(batches))

        async def distill_batch(batch: List[Dict[str, Any]], label: str) -> List[EvidenceCard]:
            payload = {
                "doc": state.get("doc_source"),
                "doc_type": state.get("document_type", "unknown"),
                "chunks": json.dumps(batch),
                "format_instructions": format_instructions,
            }
            try:
                if structured_chain:
                    response = await self._ainvoke_with_retry(
                        chain=structured_chain,
                        payload=payload,
                        limiter=self.query_rate_limiter,
                        model_label="query_model",
                        max_retries=int(os.getenv("GEMINI_MAX_RETRIES", "3")),
                    )
                    parsed = response.get("parsed") if isinstance(response, dict) else None
                    if parsed is None:
                        parsing_error = response.get("parsing_error") if isinstance(response, dict) else None
                        await self._log_event(
                            "evidence_distill_parse_error",
                            {
                                "doc_source": state.get("doc_source"),
                                "batch": label,
                                "error": str(parsing_error) if parsing_error else "Structured output missing",
                            },
                        )
                        parsed = await self._ainvoke_with_parser(
                            chain=base_chain,
                            parser=parser,
                            payload=payload,
                            limiter=self.query_rate_limiter,
                            model_label="query_model",
                            max_retries=int(os.getenv("GEMINI_MAX_RETRIES", "3")),
                            parse_label="evidence_distill",
                            log_context={"doc_source": state.get("doc_source")},
                        )
                    digest = await self._coerce_model_output(
                        parsed,
                        EvidenceDigest,
                        parse_label="evidence_distill",
                        log_context={"doc_source": state.get("doc_source")},
                    )
                else:
                    digest = await self._ainvoke_with_parser(
                        chain=base_chain,
                        parser=parser,
                        payload=payload,
                        limiter=self.query_rate_limiter,
                        model_label="query_model",
                        max_retries=int(os.getenv("GEMINI_MAX_RETRIES", "3")),
                        parse_label="evidence_distill",
                        log_context={"doc_source": state.get("doc_source")},
                    )
            except Exception as exc:
                if structured_chain:
                    await self._log_event(
                        "evidence_distill_structured_failure",
                        {
                            "doc_source": state.get("doc_source"),
                            "batch": label,
                            "error": str(exc),
                        },
                    )
                    try:
                        digest = await self._ainvoke_with_parser(
                            chain=base_chain,
                            parser=parser,
                            payload=payload,
                            limiter=self.query_rate_limiter,
                            model_label="query_model",
                            max_retries=int(os.getenv("GEMINI_MAX_RETRIES", "3")),
                            parse_label="evidence_distill",
                            log_context={"doc_source": state.get("doc_source")},
                        )
                        self._emit_llm_status(
                            step="evidence_distill",
                            doc_source=state.get("doc_source"),
                            detail=f"{label} (fallback)",
                        )
                        return digest.cards or []
                    except Exception as fallback_exc:
                        exc = fallback_exc
                if len(batch) > 1 and self._should_split_distill_error(exc):
                    mid = len(batch) // 2
                    left = await distill_batch(batch[:mid], f"{label}.1")
                    right = await distill_batch(batch[mid:], f"{label}.2")
                    return left + right
                await self._log_event(
                    "evidence_distill_fallback",
                    {
                        "doc_source": state.get("doc_source"),
                        "batch": label,
                        "error": str(exc),
                        "strategy": "heuristic_cards",
                    },
                )
                return self._fallback_evidence_cards_from_chunks(batch)

            self._emit_llm_status(
                step="evidence_distill",
                doc_source=state.get("doc_source"),
                detail=label,
            )
            return digest.cards or []

        for idx, batch in enumerate(batches, start=1):
            if status_callback:
                status_callback(f"Distilling evidence ({idx}/{total_batches})", 0.82)
            distilled_cards.extend(await distill_batch(batch, f"batch {idx}/{total_batches}"))

        web_cards = self._web_context_to_evidence(web_context)
        combined_cards = self._dedupe_evidence_cards(distilled_cards + web_cards)

        await self._log_event(
            "evidence_distilled",
            {
                "doc_source": state.get("doc_source"),
                "retrieval_count": len(retrieval_chunks),
                "batch_count": len(batches),
                "card_count": len(combined_cards),
                "web_card_count": len(web_cards),
            },
        )
        return {"evidence_cards": [card.model_dump() for card in combined_cards]}

    async def _screen_web_candidates(
        self,
        *,
        query: str,
        document_summary: str,
        document_type: str,
        candidates: List[WebSourceCandidate],
        doc_source: Optional[str] = None,
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
        chain = prompt | self.query_model
        summary_snippet = (document_summary or "")[:80]
        try:
            response: WebSelectionResponse = await self._ainvoke_with_parser(
                chain=chain,
                parser=parser,
                payload={
                    "summary": document_summary,
                    "doc_type": document_type,
                    "query": query,
                    "candidates": json.dumps([cand.model_dump() for cand in candidates]),
                    "format_instructions": format_instructions,
                },
                limiter=self.query_rate_limiter,
                model_label="query_model",
                max_retries=int(os.getenv("GEMINI_MAX_RETRIES", "3")),
                parse_label="web_candidate_screen",
                log_context={"doc_source": summary_snippet, "query": query},
            )
            self._emit_llm_status(
                step="web_candidate_screen",
                doc_source=doc_source,
                detail=query[:120] if query else None,
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
        doc_source: Optional[str] = None,
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
        chain = prompt | self.query_model
        decision: WebContentDecision = await self._ainvoke_with_parser(
            chain=chain,
            parser=parser,
            payload={
                "summary": document_summary,
                "doc_type": document_type,
                "query": query,
                "url": candidate.href or "unknown",
                "title": candidate.title or "",
                "snippet": candidate.snippet or "",
                "content": content,
                "format_instructions": format_instructions,
            },
            limiter=self.query_rate_limiter,
            model_label="query_model",
            max_retries=int(os.getenv("GEMINI_MAX_RETRIES", "3")),
            parse_label="web_content_decision",
            log_context={"doc_source": document_summary[:80], "query": query},
        )
        self._emit_llm_status(
            step="web_content_eval",
            doc_source=doc_source,
            detail=candidate.href or candidate.title,
        )
        return decision

    async def _requirements_node(
        self,
        state: AgenticState,
        status_callback: Optional[Callable[[str, float], None]] = None,
    ) -> AgenticState:
        parser = PydanticOutputParser(pydantic_object=RequirementBundle)
        format_instructions = parser.get_format_instructions()
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are Agent 2, a senior regulatory implementation architect for ING Bank."
                    " Use the distilled evidence cards to translate legal obligations into actionable business and data requirements."
                    " Address governance, risk management, assurance, reporting formats, and data collection duties relevant to lending, investment, and underwriting activities."
                    " Cite chunk IDs/pages for every document-driven statement and include online citations whenever external context informed the requirement."
                    " Format every entry in `document_sources` exactly as `{{source}}, page {{page_number}}, chunk {{chunk_id}}` using the values from the evidence cards."
                    " If a page number is missing, write `page unknown` but always keep the chunk identifier in the same format."
                    " Evidence cards with origin `web` should be cited in `online_sources` using their `source` value."
                    " Respond strictly with the JSON schema described in the instructions so downstream systems can ingest your output.",
                ),
                (
                    "human",
                    "Document: {doc}\nType: {doc_type}\nSummary: {summary}\n\nEvidence cards: {evidence}\n\n"
                    "{format_instructions}",
                ),
            ]
        )
        structured = self._build_structured_chain(self.requirements_model, RequirementBundle)
        chain = prompt | (structured or self.requirements_model)

        evidence_cards = state.get("evidence_cards")
        if not evidence_cards:
            evidence_cards = [
                {
                    "origin": "document",
                    "claim": self._trim_text(self._normalize_text(row.get("text") or ""), 600),
                    "source": row.get("source"),
                    "page": row.get("page"),
                    "chunk_id": row.get("chunk_id"),
                    "certainty": "low",
                }
                for row in state.get("retrieval_results", [])
            ]
            evidence_cards.extend(
                [card.model_dump() for card in self._web_context_to_evidence(state.get("web_context", []))]
            )
        evidence_text = json.dumps(evidence_cards)

        if status_callback:
            status_callback("Drafting requirement bundle", 0.2)
        if structured:
            response = None
            try:
                response = await self._ainvoke_with_retry(
                    chain=chain,
                    payload={
                        "doc": state.get("doc_source"),
                        "doc_type": state.get("document_type", "unknown"),
                        "summary": state.get("document_summary", ""),
                        "evidence": evidence_text,
                        "format_instructions": format_instructions,
                    },
                    limiter=self.requirements_rate_limiter,
                    model_label="requirements_model",
                    max_retries=int(os.getenv("GEMINI_MAX_RETRIES", "3")),
                )
            except Exception as exc:
                await self._log_event(
                    "requirements_structured_failure",
                    {
                        "doc_source": state.get("doc_source"),
                        "error": str(exc),
                    },
                )
            parsed = response.get("parsed") if isinstance(response, dict) else None
            if parsed is None:
                parsing_error = response.get("parsing_error") if isinstance(response, dict) else None
                await self._log_event(
                    "requirements_parse_error",
                    {
                        "doc_source": state.get("doc_source"),
                        "error": str(parsing_error) if parsing_error else "Structured output missing",
                    },
                )
                parsed = await self._ainvoke_with_parser(
                    chain=prompt | self.requirements_model,
                    parser=parser,
                    payload={
                        "doc": state.get("doc_source"),
                        "doc_type": state.get("document_type", "unknown"),
                        "summary": state.get("document_summary", ""),
                        "evidence": evidence_text,
                        "format_instructions": format_instructions,
                    },
                    limiter=self.requirements_rate_limiter,
                    model_label="requirements_model",
                    max_retries=int(os.getenv("GEMINI_MAX_RETRIES", "3")),
                    parse_label="requirements",
                    log_context={"doc_source": state.get("doc_source")},
                )
            bundle = await self._coerce_model_output(
                parsed,
                RequirementBundle,
                parse_label="requirements",
                log_context={"doc_source": state.get("doc_source")},
            )
        else:
            bundle = await self._ainvoke_with_parser(
                chain=chain,
                parser=parser,
                payload={
                    "doc": state.get("doc_source"),
                    "doc_type": state.get("document_type", "unknown"),
                    "summary": state.get("document_summary", ""),
                    "evidence": evidence_text,
                    "format_instructions": format_instructions,
                },
                limiter=self.requirements_rate_limiter,
                model_label="requirements_model",
                max_retries=int(os.getenv("GEMINI_MAX_RETRIES", "3")),
                parse_label="requirements",
                log_context={"doc_source": state.get("doc_source")},
            )
        self._emit_llm_status(step="requirements_draft", doc_source=state.get("doc_source"))
        if status_callback:
            status_callback("Validating requirement bundle", 0.85)
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

    async def _requirements_refinement_node(self, state: AgenticState) -> AgenticState:
        existing_requirements = state.get("requirements")
        if not existing_requirements:
            return {}

        prior_bundle = RequirementBundle.model_validate(existing_requirements)
        parser = PydanticOutputParser(pydantic_object=RequirementBundle)
        format_instructions = parser.get_format_instructions()
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are Agent 2R, a self-reviewing regulatory implementation architect for ING Bank."
                    " Re-read the previously generated requirements bundle and tighten it."
                    " Revise existing entries by updating their description, rationale, or sources as needed,"
                    " keeping the original `id` whenever you edit an item."
                    " Add new business or data requirements whenever the document or web context mandates coverage that is missing."
                    " Elevate actionability: spell out concrete owner actions, controls, data flows, and timing so a reporting team can operationalize the requirement in the bank's yearly report."
                    " Strengthen coherence by explaining how requirements interact or build on one another, flagging dependencies or sequencing when relevant."
                    " Use richer detail where the source material allows (e.g., explicit thresholds, scenario coverage, assurance cadence) without inventing unsupported facts."
                    " Ensure every change is grounded in the provided evidence cards, citing them in the same format as before, and highlight why the refinement improves usefulness for the reporting audience."
                    " Respond strictly with the JSON schema described in the instructions so downstream systems can ingest your output.",
                ),
                (
                    "human",
                    "Document: {doc}\nType: {doc_type}\nSummary: {summary}\n\n"
                    "Prior requirements bundle: {existing}\n\nEvidence cards: {evidence}\n\n{format_instructions}",
                ),
            ]
        )
        structured = self._build_structured_chain(self.requirements_model, RequirementBundle)
        chain = prompt | (structured or self.requirements_model)

        evidence_text = json.dumps(state.get("evidence_cards", []))
        if not evidence_text or evidence_text == "[]":
            evidence_text = json.dumps(state.get("retrieval_results", []))
        existing_text = json.dumps(prior_bundle.model_dump())

        if structured:
            response = None
            try:
                response = await self._ainvoke_with_retry(
                    chain=chain,
                    payload={
                        "doc": state.get("doc_source"),
                        "doc_type": state.get("document_type", "unknown"),
                        "summary": state.get("document_summary", ""),
                        "existing": existing_text,
                        "evidence": evidence_text,
                        "format_instructions": format_instructions,
                    },
                    limiter=self.requirements_rate_limiter,
                    model_label="requirements_model",
                    max_retries=int(os.getenv("GEMINI_MAX_RETRIES", "3")),
                )
            except Exception as exc:
                await self._log_event(
                    "requirements_refinement_structured_failure",
                    {
                        "doc_source": state.get("doc_source"),
                        "error": str(exc),
                    },
                )
            parsed = response.get("parsed") if isinstance(response, dict) else None
            if parsed is None:
                parsing_error = response.get("parsing_error") if isinstance(response, dict) else None
                await self._log_event(
                    "requirements_refinement_parse_error",
                    {
                        "doc_source": state.get("doc_source"),
                        "error": str(parsing_error) if parsing_error else "Structured output missing",
                    },
                )
                parsed = await self._ainvoke_with_parser(
                    chain=prompt | self.requirements_model,
                    parser=parser,
                    payload={
                        "doc": state.get("doc_source"),
                        "doc_type": state.get("document_type", "unknown"),
                        "summary": state.get("document_summary", ""),
                        "existing": existing_text,
                        "evidence": evidence_text,
                        "format_instructions": format_instructions,
                    },
                    limiter=self.requirements_rate_limiter,
                    model_label="requirements_model",
                    max_retries=int(os.getenv("GEMINI_MAX_RETRIES", "3")),
                    parse_label="requirements_refinement",
                    log_context={"doc_source": state.get("doc_source")},
                )
            refined_bundle = await self._coerce_model_output(
                parsed,
                RequirementBundle,
                parse_label="requirements_refinement",
                log_context={"doc_source": state.get("doc_source")},
            )
        else:
            refined_bundle = await self._ainvoke_with_parser(
                chain=chain,
                parser=parser,
                payload={
                    "doc": state.get("doc_source"),
                    "doc_type": state.get("document_type", "unknown"),
                    "summary": state.get("document_summary", ""),
                    "existing": existing_text,
                    "evidence": evidence_text,
                    "format_instructions": format_instructions,
                },
                limiter=self.requirements_rate_limiter,
                model_label="requirements_model",
                max_retries=int(os.getenv("GEMINI_MAX_RETRIES", "3")),
                parse_label="requirements_refinement",
                log_context={"doc_source": state.get("doc_source")},
            )
        self._emit_llm_status(step="requirements_refine", doc_source=state.get("doc_source"))
        await self._log_event(
            "requirements_refinement",
            {
                "doc_source": state.get("doc_source"),
                "document_type": state.get("document_type"),
                "previous_business_requirement_count": len(prior_bundle.business_requirements),
                "previous_data_requirement_count": len(prior_bundle.data_requirements),
                "business_requirement_count": len(refined_bundle.business_requirements),
                "data_requirement_count": len(refined_bundle.data_requirements),
            },
        )
        return {"requirements": refined_bundle.model_dump()}

    async def arun(self, state: AgenticState) -> AgenticState:
        result = await self.graph.ainvoke(state)
        return cast(AgenticState, result)

    async def process_document(
        self,
        state: AgenticState,
        status_callback: Optional[Callable[[str, float], None]] = None,
    ) -> AgenticState:
        """Run query, retrieval, and context nodes for a single document."""

        if status_callback:
            status_callback("Drafting retrieval queries", 0.15)
        query_state = await self._query_agent_node(state)
        state.update(query_state)

        if status_callback:
            status_callback("Retrieving relevant chunks", 0.45)
        retrieval_state = await self._retrieval_node(state)
        state.update(retrieval_state)

        if status_callback:
            status_callback("Assessing context gaps", 0.7)
        context_state = await self._context_node(state, status_callback=status_callback)
        state.update(context_state)

        return state

    async def distill_evidence(
        self,
        state: AgenticState,
        status_callback: Optional[Callable[[str, float], None]] = None,
    ) -> AgenticState:
        """Distill retrieval chunks into compact evidence cards."""

        return await self._distill_evidence_node(state, status_callback=status_callback)

    async def generate_requirements(
        self,
        state: AgenticState,
        status_callback: Optional[Callable[[str, float], None]] = None,
    ) -> AgenticState:
        """Invoke only the requirements node with the provided state."""

        return await self._requirements_node(state, status_callback=status_callback)

    async def merge_requirements(
        self,
        bundles: List[RequirementBundle],
        doc_sources: List[str],
        status_callback: Optional[Callable[[str, float], None]] = None,
    ) -> RequirementBundle:
        """Merge per-document requirement bundles into a consolidated bundle."""

        parser = PydanticOutputParser(pydantic_object=RequirementBundle)
        format_instructions = parser.get_format_instructions()
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are consolidating multiple requirement bundles produced from individual regulatory documents."
                    " Merge overlapping requirements, preserve citations, and produce a single coherent bundle."
                    " Reindex business requirements as BR-001, BR-002... and data requirements as DR-001, DR-002..."
                    " Keep document_sources and online_sources as arrays, merging citations when requirements overlap."
                    " Produce a deduplicated assumptions list.",
                ),
                (
                    "human",
                    "Documents: {docs}\n\nPer-document bundles (JSON list): {bundles}\n\n{format_instructions}",
                ),
            ]
        )
        structured = self._build_structured_chain(self.requirements_model, RequirementBundle)
        chain = prompt | (structured or self.requirements_model)

        if status_callback:
            status_callback("Merging requirement bundles", 0.4)

        payload = {
            "docs": ", ".join(doc_sources) if doc_sources else "unknown",
            "bundles": json.dumps([bundle.model_dump() for bundle in bundles]),
            "format_instructions": format_instructions,
        }
        if structured:
            response = None
            try:
                response = await self._ainvoke_with_retry(
                    chain=chain,
                    payload=payload,
                    limiter=self.requirements_rate_limiter,
                    model_label="requirements_model",
                    max_retries=int(os.getenv("GEMINI_MAX_RETRIES", "3")),
                )
            except Exception as exc:
                await self._log_event(
                    "requirements_merge_structured_failure",
                    {
                        "doc_source": ",".join(doc_sources),
                        "error": str(exc),
                    },
                )
            parsed = response.get("parsed") if isinstance(response, dict) else None
            if parsed is None:
                parsing_error = response.get("parsing_error") if isinstance(response, dict) else None
                await self._log_event(
                    "requirements_merge_parse_error",
                    {
                        "doc_source": ",".join(doc_sources),
                        "error": str(parsing_error) if parsing_error else "Structured output missing",
                    },
                )
                parsed = await self._ainvoke_with_parser(
                    chain=prompt | self.requirements_model,
                    parser=parser,
                    payload=payload,
                    limiter=self.requirements_rate_limiter,
                    model_label="requirements_model",
                    max_retries=int(os.getenv("GEMINI_MAX_RETRIES", "3")),
                    parse_label="requirements_merge",
                    log_context={"doc_source": ",".join(doc_sources)},
                )
            merged_bundle = await self._coerce_model_output(
                parsed,
                RequirementBundle,
                parse_label="requirements_merge",
                log_context={"doc_source": ",".join(doc_sources)},
            )
        else:
            merged_bundle = await self._ainvoke_with_parser(
                chain=chain,
                parser=parser,
                payload=payload,
                limiter=self.requirements_rate_limiter,
                model_label="requirements_model",
                max_retries=int(os.getenv("GEMINI_MAX_RETRIES", "3")),
                parse_label="requirements_merge",
                log_context={"doc_source": ",".join(doc_sources)},
            )

        self._emit_llm_status(step="requirements_merge", doc_source=",".join(doc_sources))
        await self._log_event(
            "requirements_merge",
            {
                "doc_source": ",".join(doc_sources),
                "bundle_count": len(bundles),
                "business_requirement_count": len(merged_bundle.business_requirements),
                "data_requirement_count": len(merged_bundle.data_requirements),
                "assumption_count": len(merged_bundle.assumptions),
            },
        )
        return merged_bundle

    async def _log_event(self, event_type: str, payload: Dict[str, Any]) -> None:
        if not self.decision_logger:
            return
        await self.decision_logger.log({"event_type": event_type, **payload})
