import asyncio

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field

from AgenticAI.agentic.langgraph_runner import AgenticGraphRunner
from AgenticAI.agentic.models import RequirementBundle


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
    async def call_tool(self, *args, **kwargs):
        raise AssertionError("call_tool should not be invoked in this test")


def test_query_agent_recovers_prefixed_json():
    response = (
        'Here is JSON: {"document_type":"Directive","summary":"Scope overview","queries":["q1","q2"]}'
    )
    model = FakeChatModel(responses=[response])
    runner = AgenticGraphRunner(
        query_model=model,
        requirements_model=model,
        mcp_client=FakeMCPClient(),
        throttle_enabled=False,
    )
    state = {"doc_source": "doc.pdf", "doc_text": "body", "doc_headings": ["Heading"]}

    result = asyncio.run(runner._query_agent_node(state))

    assert result["queries"] == ["q1", "q2"]
    assert result["document_type"] == "Directive"


def test_query_agent_recovers_list_content():
    response = [
        {"type": "text", "text": "Here is JSON: "},
        {"type": "text", "text": '{"document_type":"Directive","summary":"Scope","queries":["q1"]}'},
    ]
    model = FakeChatModel(responses=[response])
    runner = AgenticGraphRunner(
        query_model=model,
        requirements_model=model,
        mcp_client=FakeMCPClient(),
        throttle_enabled=False,
    )
    state = {"doc_source": "doc.pdf", "doc_text": "body", "doc_headings": ["Heading"]}

    result = asyncio.run(runner._query_agent_node(state))

    assert result["queries"] == ["q1"]
    assert result["document_type"] == "Directive"


def test_query_agent_ignores_reasoning_payload_with_braces():
    response = (
        "Reasoning with placeholder {source} and {chunk_id} before output. "
        '{"document_type":"Directive","summary":"Scope","queries":["q1"]}'
    )
    model = FakeChatModel(responses=[response])
    runner = AgenticGraphRunner(
        query_model=model,
        requirements_model=model,
        mcp_client=FakeMCPClient(),
        throttle_enabled=False,
    )
    state = {"doc_source": "doc.pdf", "doc_text": "body", "doc_headings": ["Heading"]}

    result = asyncio.run(runner._query_agent_node(state))

    assert result["queries"] == ["q1"]
    assert result["document_type"] == "Directive"


def test_requirements_agent_recovers_prefixed_json():
    response = (
        "Here is JSON: "
        '{"document":"doc.pdf","document_type":"Directive","business_requirements":[],"data_requirements":[],"assumptions":[]}'
    )
    model = FakeChatModel(responses=[response])
    runner = AgenticGraphRunner(
        query_model=model,
        requirements_model=model,
        mcp_client=FakeMCPClient(),
        throttle_enabled=False,
    )
    state = {
        "doc_source": "doc.pdf",
        "document_type": "Directive",
        "document_summary": "Summary",
        "retrieval_results": [],
        "web_context": [],
    }

    result = asyncio.run(runner._requirements_node(state))

    assert result["requirements"]["document"] == "doc.pdf"


def test_requirements_agent_recovers_list_content():
    response = [
        {"type": "text", "text": "Here is JSON: "},
        {"type": "text", "text": '{"document":"doc.pdf","document_type":"Directive","business_requirements":[],"data_requirements":[],"assumptions":[]}'},
    ]
    model = FakeChatModel(responses=[response])
    runner = AgenticGraphRunner(
        query_model=model,
        requirements_model=model,
        mcp_client=FakeMCPClient(),
        throttle_enabled=False,
    )
    state = {
        "doc_source": "doc.pdf",
        "document_type": "Directive",
        "document_summary": "Summary",
        "retrieval_results": [],
        "web_context": [],
    }

    result = asyncio.run(runner._requirements_node(state))

    assert result["requirements"]["document"] == "doc.pdf"


def test_requirements_agent_accepts_unescaped_newlines():
    response = (
        '{\"document\":\"doc.pdf\",\"document_type\":\"Directive\",\"business_requirements\":['
        '{\"id\":\"BR-1\",\"description\":\"Line 1\\nLine 2\",\"rationale\":\"Why\",'
        '\"document_sources\":[],\"online_sources\":[]}'
        '],\"data_requirements\":[],\"assumptions\":[]}'
    )
    model = FakeChatModel(responses=[response])
    runner = AgenticGraphRunner(
        query_model=model,
        requirements_model=model,
        mcp_client=FakeMCPClient(),
        throttle_enabled=False,
    )
    state = {
        "doc_source": "doc.pdf",
        "document_type": "Directive",
        "document_summary": "Summary",
        "retrieval_results": [],
        "web_context": [],
    }

    result = asyncio.run(runner._requirements_node(state))

    assert result["requirements"]["document"] == "doc.pdf"


def test_requirements_agent_ignores_reasoning_with_braces_and_trailing_text():
    response = (
        "Reasoning with placeholders {source} and {chunk_id}. "
        "Now output JSON."
        "{"
        "\"document\":\"doc.pdf\","
        "\"document_type\":\"Directive\","
        "\"business_requirements\":[],"
        "\"data_requirements\":[],"
        "\"assumptions\":[]"
        "}"
        " For troubleshooting, visit: https://example.com"
    )
    model = FakeChatModel(responses=[response])
    runner = AgenticGraphRunner(
        query_model=model,
        requirements_model=model,
        mcp_client=FakeMCPClient(),
        throttle_enabled=False,
    )
    state = {
        "doc_source": "doc.pdf",
        "document_type": "Directive",
        "document_summary": "Summary",
        "retrieval_results": [],
        "web_context": [],
    }

    result = asyncio.run(runner._requirements_node(state))

    assert result["requirements"]["document"] == "doc.pdf"


def test_coerce_structured_output_strips_fenced_json():
    response = (
        "```json\n"
        '{"document":"doc.pdf","document_type":"Directive","business_requirements":[],"data_requirements":[],"assumptions":[]}'
        "\n```"
    )
    model = FakeChatModel(responses=[response])
    runner = AgenticGraphRunner(
        query_model=model,
        requirements_model=model,
        mcp_client=FakeMCPClient(),
        throttle_enabled=False,
    )
    bundle = asyncio.run(
        runner._coerce_model_output(
            response,
            RequirementBundle,
            parse_label="requirements",
            log_context={"doc_source": "doc.pdf"},
        )
    )

    assert bundle.document == "doc.pdf"


def test_distill_fallback_cards_from_chunks():
    runner = AgenticGraphRunner(
        query_model=FakeChatModel(responses=[]),
        requirements_model=FakeChatModel(responses=[]),
        mcp_client=FakeMCPClient(),
        throttle_enabled=False,
    )
    batch = [
        {
            "chunk_id": "c1",
            "source": "doc.pdf",
            "page": 2,
            "text": "Companies must report annually. Additional context follows.",
        }
    ]
    cards = runner._fallback_evidence_cards_from_chunks(batch)

    assert cards[0].chunk_id == "c1"
    assert cards[0].origin == "document"
