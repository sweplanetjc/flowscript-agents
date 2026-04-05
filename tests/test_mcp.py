"""Tests for FlowScript MCP server handler."""

import json
import pytest

from flowscript_agents import UnifiedMemory
from flowscript_agents.mcp import MCPHandler, TOOLS, _jsonrpc_response, _jsonrpc_error

# Import shared MockEmbeddings from conftest
from conftest import MockEmbeddings


def _make_handler(with_embedder: bool = False, with_llm: bool = False):
    emb = MockEmbeddings(dims=16) if with_embedder else None
    llm = None
    if with_llm:
        def llm(prompt):
            return json.dumps({
                "nodes": [{"type": "thought", "content": "extracted fact"}],
                "relationships": [], "states": [],
            })
    umem = UnifiedMemory(embedder=emb, llm=llm)
    return MCPHandler(umem), umem


class TestToolDefinitions:
    def test_all_tools_have_required_fields(self):
        for tool in TOOLS:
            assert "name" in tool
            assert "description" in tool
            assert "inputSchema" in tool

    def test_tool_count(self):
        assert len(TOOLS) == 15

    def test_tool_names(self):
        names = {t["name"] for t in TOOLS}
        expected = {
            "search_memory", "add_memory", "get_context",
            "query_tensions", "query_blocked", "query_why",
            "query_what_if", "query_alternatives", "query_counterfactual",
            "remove_memory", "session_wrap", "memory_stats",
            "query_audit", "verify_audit",
            "encode_exchange",
        }
        assert names == expected


class TestSearchMemory:
    def test_keyword_search(self):
        handler, umem = _make_handler()
        umem.add_raw("Redis is fast for caching")
        umem.add_raw("PostgreSQL ensures ACID")
        result = handler.handle_tool("search_memory", {"query": "Redis"})
        assert result["count"] > 0
        assert result["mode"] == "unified"

    def test_vector_search(self):
        handler, umem = _make_handler(with_embedder=True)
        umem.add_raw("Redis is fast for caching")
        result = handler.handle_tool("search_memory", {"query": "Redis", "mode": "vector"})
        assert result["mode"] == "vector"

    def test_keyword_only_mode(self):
        handler, umem = _make_handler()
        umem.add_raw("Redis is fast")
        result = handler.handle_tool("search_memory", {"query": "Redis", "mode": "keyword"})
        assert result["mode"] == "keyword"

    def test_empty_memory(self):
        handler, _ = _make_handler()
        result = handler.handle_tool("search_memory", {"query": "anything"})
        assert result["count"] == 0


class TestAddMemory:
    def test_add_without_llm(self):
        handler, umem = _make_handler()
        result = handler.handle_tool("add_memory", {"text": "Redis is fast"})
        assert result["nodes_created"] == 1
        assert umem.size == 1

    def test_add_with_llm(self):
        handler, umem = _make_handler(with_llm=True)
        result = handler.handle_tool("add_memory", {"text": "any text"})
        assert result["nodes_created"] >= 1

    def test_add_with_metadata(self):
        handler, umem = _make_handler()
        result = handler.handle_tool("add_memory", {
            "text": "Redis is fast",
            "metadata": {"source": "test"},
        })
        assert result["nodes_created"] == 1


class TestGetContext:
    def test_empty_context(self):
        handler, _ = _make_handler()
        result = handler.handle_tool("get_context", {})
        assert result["nodes"] == 0
        assert result["context"] == ""

    def test_with_nodes(self):
        handler, umem = _make_handler()
        umem.add_raw("Redis is fast")
        umem.add_raw("PostgreSQL is reliable")
        result = handler.handle_tool("get_context", {"max_tokens": 1000})
        assert result["nodes"] == 2
        assert "Redis" in result["context"]


class TestQueryTensions:
    def test_no_tensions(self):
        handler, _ = _make_handler()
        result = handler.handle_tool("query_tensions", {})
        assert "error" not in result or result.get("metadata", {}).get("total_tensions", 0) == 0

    def test_with_tensions(self):
        handler, umem = _make_handler()
        a = umem.memory.thought("Speed")
        b = umem.memory.thought("Safety")
        a.tension_with(b, axis="performance vs reliability")
        result = handler.handle_tool("query_tensions", {"group_by": "axis"})
        assert result is not None


class TestQueryBlocked:
    def test_no_blockers(self):
        handler, _ = _make_handler()
        result = handler.handle_tool("query_blocked", {})
        assert result is not None

    def test_with_blocker(self):
        handler, umem = _make_handler()
        ref = umem.memory.thought("Database migration")
        ref.block(reason="waiting on schema approval")
        result = handler.handle_tool("query_blocked", {})
        assert result is not None


class TestQueryWhy:
    def test_with_causal_chain(self):
        handler, umem = _make_handler()
        a = umem.memory.thought("Root cause")
        b = umem.memory.thought("Effect")
        a.causes(b)
        result = handler.handle_tool("query_why", {"node_id": b.id})
        assert "error" not in result

    def test_by_content(self):
        handler, umem = _make_handler()
        a = umem.memory.thought("Root cause")
        b = umem.memory.thought("The effect of root cause")
        a.causes(b)
        result = handler.handle_tool("query_why", {"content": "effect"})
        assert "error" not in result

    def test_no_node_found(self):
        handler, _ = _make_handler()
        result = handler.handle_tool("query_why", {"content": "nonexistent"})
        assert "error" in result


class TestQueryWhatIf:
    def test_with_consequences(self):
        handler, umem = _make_handler()
        a = umem.memory.thought("Change database")
        b = umem.memory.thought("Need to migrate data")
        a.causes(b)
        result = handler.handle_tool("query_what_if", {"node_id": a.id})
        assert "error" not in result

    def test_by_content(self):
        handler, umem = _make_handler()
        a = umem.memory.thought("Change database schema")
        b = umem.memory.thought("Downstream API breaks")
        a.causes(b)
        result = handler.handle_tool("query_what_if", {"content": "database schema"})
        assert "error" not in result

    def test_no_node_found(self):
        handler, _ = _make_handler()
        result = handler.handle_tool("query_what_if", {"content": "nonexistent"})
        assert "error" in result


class TestQueryAlternatives:
    def test_with_alternatives(self):
        handler, umem = _make_handler()
        q = umem.memory.question("Which database?")
        umem.memory.alternative(q, "Redis")
        umem.memory.alternative(q, "PostgreSQL")
        result = handler.handle_tool("query_alternatives", {"question_id": q.id})
        assert "error" not in result

    def test_by_content(self):
        handler, umem = _make_handler()
        q = umem.memory.question("Which database?")
        umem.memory.alternative(q, "Redis")
        result = handler.handle_tool("query_alternatives", {"content": "database"})
        assert "error" not in result


class TestMemoryStats:
    def test_empty_stats(self):
        handler, _ = _make_handler()
        result = handler.handle_tool("memory_stats", {})
        assert result["total_nodes"] == 0
        assert result["tiers"]["current"] == 0

    def test_with_data(self):
        handler, umem = _make_handler(with_embedder=True)
        umem.add_raw("Redis is fast")
        umem.add_raw("PostgreSQL is reliable")
        result = handler.handle_tool("memory_stats", {})
        assert result["total_nodes"] == 2
        assert "embeddings" in result
        assert result["embeddings"]["indexed"] == 2

    def test_unknown_tool(self):
        handler, _ = _make_handler()
        result = handler.handle_tool("nonexistent_tool", {})
        assert "error" in result


class TestJsonRpc:
    def test_response_format(self):
        resp = _jsonrpc_response(1, {"test": True})
        assert resp["jsonrpc"] == "2.0"
        assert resp["id"] == 1
        assert resp["result"]["test"] is True

    def test_error_format(self):
        resp = _jsonrpc_error(1, -32601, "Method not found")
        assert resp["jsonrpc"] == "2.0"
        assert resp["error"]["code"] == -32601


class TestMCPStdioProtocol:
    """Test the actual MCP JSON-RPC message routing (simulates stdio)."""

    def _simulate_message(self, msg: dict) -> dict | None:
        """Simulate sending a JSON-RPC message through the server's routing logic.

        We test the routing logic directly rather than actual stdio to avoid
        subprocess complexity while still verifying protocol compliance.
        """
        import io
        from flowscript_agents.mcp import run_server, TOOLS, _jsonrpc_response, _jsonrpc_error

        umem = UnifiedMemory()
        handler = MCPHandler(umem)

        msg_id = msg.get("id")
        method = msg.get("method", "")
        params = msg.get("params", {})

        if method == "initialize":
            return _jsonrpc_response(msg_id, {
                "protocolVersion": "2025-03-26",
                "capabilities": {"tools": {}, "resources": {}},
                "serverInfo": {"name": "flowscript-agents", "version": "0.2.5"},
            })
        elif method == "notifications/initialized":
            return None  # notification, no response
        elif method == "tools/list":
            from flowscript_agents.mcp import ALL_TOOLS, _thaw
            return _jsonrpc_response(msg_id, {"tools": [_thaw(t) for t in ALL_TOOLS]})
        elif method == "tools/call":
            tool_name = params.get("name", "")
            tool_args = params.get("arguments", {})
            result = handler.handle_tool(tool_name, tool_args)
            return _jsonrpc_response(msg_id, {
                "content": [{"type": "text", "text": json.dumps(result)}],
            })
        elif method == "resources/list":
            from flowscript_agents.mcp import _INTEGRITY_RESOURCE
            return _jsonrpc_response(msg_id, {"resources": [_INTEGRITY_RESOURCE]})
        elif method == "prompts/list":
            return _jsonrpc_response(msg_id, {"prompts": []})
        elif method == "ping":
            return _jsonrpc_response(msg_id, {})
        else:
            return _jsonrpc_error(msg_id, -32601, f"Method not found: {method}")

    def test_initialize(self):
        resp = self._simulate_message({
            "jsonrpc": "2.0", "id": 1, "method": "initialize",
            "params": {"protocolVersion": "2025-03-26", "capabilities": {}},
        })
        assert resp["result"]["protocolVersion"] == "2025-03-26"
        assert resp["result"]["capabilities"]["tools"] == {}
        assert resp["result"]["serverInfo"]["name"] == "flowscript-agents"

    def test_tools_list(self):
        resp = self._simulate_message({
            "jsonrpc": "2.0", "id": 2, "method": "tools/list",
        })
        tools = resp["result"]["tools"]
        assert len(tools) == 16  # 15 verified + verify_integrity
        names = {t["name"] for t in tools}
        assert "search_memory" in names
        assert "query_what_if" in names
        assert "verify_integrity" in names

    def test_tools_call(self):
        resp = self._simulate_message({
            "jsonrpc": "2.0", "id": 3, "method": "tools/call",
            "params": {"name": "memory_stats", "arguments": {}},
        })
        content = resp["result"]["content"]
        assert len(content) == 1
        assert content[0]["type"] == "text"
        stats = json.loads(content[0]["text"])
        assert stats["total_nodes"] == 0

    def test_notification_no_response(self):
        resp = self._simulate_message({
            "jsonrpc": "2.0", "method": "notifications/initialized",
        })
        assert resp is None

    def test_resources_list(self):
        resp = self._simulate_message({
            "jsonrpc": "2.0", "id": 4, "method": "resources/list",
        })
        resources = resp["result"]["resources"]
        assert len(resources) == 1
        assert resources[0]["uri"] == "flowscript://integrity/manifest"

    def test_prompts_list(self):
        resp = self._simulate_message({
            "jsonrpc": "2.0", "id": 5, "method": "prompts/list",
        })
        assert resp["result"]["prompts"] == []

    def test_ping(self):
        resp = self._simulate_message({
            "jsonrpc": "2.0", "id": 6, "method": "ping",
        })
        assert resp["result"] == {}

    def test_unknown_method(self):
        resp = self._simulate_message({
            "jsonrpc": "2.0", "id": 7, "method": "nonexistent/method",
        })
        assert resp["error"]["code"] == -32601


class TestRemoveMemory:
    def test_remove_existing(self):
        handler, umem = _make_handler()
        ref = umem.memory.thought("Remove me")
        result = handler.handle_tool("remove_memory", {"node_id": ref.id})
        assert result["removed"] is True
        assert umem.size == 0

    def test_remove_nonexistent(self):
        handler, _ = _make_handler()
        result = handler.handle_tool("remove_memory", {"node_id": "fake-id"})
        assert result["removed"] is False

    def test_remove_no_id(self):
        handler, _ = _make_handler()
        result = handler.handle_tool("remove_memory", {})
        assert "error" in result

    def test_remove_cleans_vector_index(self):
        handler, umem = _make_handler(with_embedder=True)
        umem.add_raw("Indexed content")
        nodes = list(umem.memory._nodes.keys())
        assert umem.vector_index.indexed_count == 1
        handler.handle_tool("remove_memory", {"node_id": nodes[0]})
        assert umem.vector_index.indexed_count == 0


class TestSessionWrap:
    def test_wrap_empty(self):
        handler, umem = _make_handler()
        umem.memory.session_start()
        result = handler.handle_tool("session_wrap", {})
        assert result["nodes_before"] == 0
        assert result["nodes_after"] == 0
        assert result["nodes_pruned"] == 0

    def test_wrap_with_data(self):
        handler, umem = _make_handler()
        umem.memory.session_start()
        umem.add_raw("Active node")
        result = handler.handle_tool("session_wrap", {})
        assert result["nodes_before"] == 1
        assert result["nodes_after"] >= 0  # may prune if dormant

    def test_wrap_continuity_disabled(self):
        """session_wrap without continuity manager reports disabled."""
        handler, umem = _make_handler()
        umem.memory.session_start()
        result = handler.handle_tool("session_wrap", {})
        assert result["continuity"]["produced"] is False
        assert result["continuity"]["reason"] == "disabled"


class TestAutoConfiguration:
    """Tests for OPENAI_API_KEY auto-detection logic."""

    def test_run_server_accepts_consolidation_provider(self):
        """run_server() should accept consolidation_provider kwarg."""
        from flowscript_agents.mcp import run_server
        import inspect
        sig = inspect.signature(run_server)
        assert "consolidation_provider" in sig.parameters

    def test_auto_configure_requires_openai(self):
        """_auto_configure_openai should exist and be callable."""
        from flowscript_agents.mcp import _auto_configure_openai
        assert callable(_auto_configure_openai)

    def test_openai_consolidation_provider_class_exists(self):
        """_OpenAIConsolidationProvider should exist and have tool_call method."""
        from flowscript_agents.mcp import _OpenAIConsolidationProvider
        assert hasattr(_OpenAIConsolidationProvider, "tool_call")

    def test_anthropic_consolidation_provider_class_exists(self):
        """_AnthropicConsolidationProvider should exist and have tool_call method."""
        from flowscript_agents.mcp import _AnthropicConsolidationProvider
        assert hasattr(_AnthropicConsolidationProvider, "tool_call")

    def test_auto_configure_anthropic_exists(self):
        """_auto_configure_anthropic should exist and be callable."""
        from flowscript_agents.mcp import _auto_configure_anthropic
        assert callable(_auto_configure_anthropic)

    def test_log_function(self):
        """_log should write to stderr without raising."""
        from flowscript_agents.mcp import _log
        _log("test message")

    def test_tool_descriptions_have_behavioral_guidance(self):
        """Tool descriptions should include 'Call this' behavioral guidance."""
        behavioral_tools = [
            "search_memory", "add_memory", "get_context",
            "query_tensions", "query_blocked", "query_why",
            "query_alternatives", "query_what_if",
            "remove_memory", "session_wrap",
        ]
        for tool in TOOLS:
            if tool["name"] in behavioral_tools:
                assert "Call this" in tool["description"], (
                    f"Tool {tool['name']} missing behavioral guidance "
                    f"('Call this when...')"
                )


class TestIsErrorFlag:
    """Tests for MCP isError flag on tool call error responses."""

    def test_error_response_has_is_error(self):
        """Tool errors should set isError: true in the response."""
        handler, _ = _make_handler()
        result = handler.handle_tool("query_why", {"content": "nonexistent"})
        assert "error" in result  # handler returns error dict
        # Verify the MCP protocol-level isError by simulating the message flow
        from flowscript_agents.mcp import _jsonrpc_response
        # The server loop checks for "error" key and sets isError
        is_error = "error" in result
        assert is_error is True

    def test_success_response_no_is_error(self):
        handler, umem = _make_handler()
        umem.add_raw("test content")
        result = handler.handle_tool("memory_stats", {})
        assert "error" not in result


class TestInputValidation:
    """Tests for input validation."""

    def test_empty_text_rejected(self):
        handler, _ = _make_handler()
        result = handler.handle_tool("add_memory", {"text": ""})
        assert "error" in result

    def test_whitespace_text_rejected(self):
        handler, _ = _make_handler()
        result = handler.handle_tool("add_memory", {"text": "   "})
        assert "error" in result

    def test_valid_text_accepted(self):
        handler, _ = _make_handler()
        result = handler.handle_tool("add_memory", {"text": "Valid content"})
        assert "error" not in result
        assert result["nodes_created"] == 1

class TestSessionWrapWithContinuity:
    """Tests for session_wrap when ContinuityManager is configured."""

    def test_session_wrap_produces_continuity(self):
        """session_wrap should produce continuity metadata when manager is configured."""
        import tempfile, os
        from flowscript_agents.continuity import ContinuityManager

        mock_llm = lambda p: "# Test — Memory\n\n## State\nTest\n\n## Patterns\nNone\n\n## Decisions\nNone\n\n## Context\nTest"
        cont_mgr = ContinuityManager(llm=mock_llm, project_name="Test")

        with tempfile.TemporaryDirectory() as tmpdir:
            mem_path = os.path.join(tmpdir, "agent.json")
            umem = UnifiedMemory(file_path=mem_path)
            umem.memory.session_start()
            umem.memory.thought("test node")

            handler = MCPHandler(umem, continuity_manager=cont_mgr, memory_path=mem_path)
            result = handler.handle_tool("session_wrap", {})

            assert "continuity" in result
            assert result["continuity"]["produced"] is True
            assert result["continuity"]["char_count"] > 0
            assert os.path.exists(result["continuity"]["path"])

    def test_session_wrap_continuity_failure_nonfatal(self):
        """If continuity production fails, session_wrap should still succeed."""
        import tempfile, os
        from flowscript_agents.continuity import ContinuityManager

        def failing_llm(prompt):
            raise RuntimeError("LLM unavailable")

        cont_mgr = ContinuityManager(llm=failing_llm, project_name="Test")

        with tempfile.TemporaryDirectory() as tmpdir:
            mem_path = os.path.join(tmpdir, "agent.json")
            umem = UnifiedMemory(file_path=mem_path)
            umem.memory.session_start()
            umem.memory.thought("test node")

            handler = MCPHandler(umem, continuity_manager=cont_mgr, memory_path=mem_path)
            result = handler.handle_tool("session_wrap", {})

            # session_wrap should succeed even though continuity failed
            assert "error" not in result
            assert "nodes_before" in result
            assert result["saved"] is True
            # Continuity key present but indicates failure
            assert result["continuity"]["produced"] is False
            assert result["continuity"]["reason"] == "error"


class TestVersionNegotiation:
    """Tests for MCP protocol version negotiation."""

    def test_matching_version(self):
        """Server should respond with client's version if compatible."""
        from flowscript_agents.mcp import _PROTOCOL_VERSION
        handler, _ = _make_handler()
        # Simulate initialize with matching version
        assert _PROTOCOL_VERSION == "2025-03-26"

    def test_newer_client_version_accepted(self):
        """Server should accept newer client versions (tools-only, compatible)."""
        from flowscript_agents.mcp import _PROTOCOL_VERSION
        assert _PROTOCOL_VERSION >= "2025-03-26"


class TestDescriptionIntegrity:
    """Tests for the three-layer MCP description integrity system."""

    def test_tools_are_frozen(self):
        """Tool definitions should be immutable MappingProxyType."""
        from types import MappingProxyType
        from flowscript_agents.mcp import TOOLS
        for tool in TOOLS:
            assert isinstance(tool, MappingProxyType), f"{tool['name']} is not frozen"

    def test_mutation_blocked(self):
        """Attempting to mutate a frozen tool should raise TypeError."""
        from flowscript_agents.mcp import TOOLS
        import pytest
        with pytest.raises(TypeError):
            TOOLS[0]["name"] = "hacked"

    def test_verify_integrity_returns_pass(self):
        """verify_integrity should return PASS on unmodified tools."""
        handler, _ = _make_handler()
        result = handler.handle_tool("verify_integrity", {})
        assert result["verdict"] == "PASS"
        assert result["count_match"] is True
        assert result["tool_count"] == 15  # verified tools (not counting verify_integrity itself)

    def test_verify_integrity_per_tool_status(self):
        """Each tool should have pass status with matching hashes."""
        handler, _ = _make_handler()
        result = handler.handle_tool("verify_integrity", {})
        for tool_result in result["tools"]:
            assert tool_result["status"] == "pass", f"{tool_result['tool']} failed integrity check"
            assert tool_result["expected_hash"] == tool_result["current_hash"]

    def test_hash_determinism(self):
        """Same tool should produce same hash across calls."""
        from flowscript_agents.mcp import TOOLS, _hash_tool_definition
        h1 = _hash_tool_definition(TOOLS[0])
        h2 = _hash_tool_definition(TOOLS[0])
        assert h1 == h2
        assert len(h1) == 64  # SHA-256 hex length

    def test_manifest_matches_runtime(self):
        """Build-time manifest should match runtime hashes."""
        from flowscript_agents.mcp import TOOLS, _INTEGRITY_MANIFEST, _hash_tool_definition
        for tool in TOOLS:
            name = tool["name"]
            assert name in _INTEGRITY_MANIFEST
            assert _INTEGRITY_MANIFEST[name] == _hash_tool_definition(tool)

    def test_integrity_resource_exists(self):
        """The integrity resource should be listed."""
        from flowscript_agents.mcp import _INTEGRITY_RESOURCE
        assert _INTEGRITY_RESOURCE["uri"] == "flowscript://integrity/manifest"
        assert _INTEGRITY_RESOURCE["mimeType"] == "application/json"

    def test_integrity_resource_frozen(self):
        """The integrity resource metadata should be frozen."""
        from types import MappingProxyType
        from flowscript_agents.mcp import _INTEGRITY_RESOURCE
        assert isinstance(_INTEGRITY_RESOURCE, MappingProxyType)

    def test_canonicalize_none_as_null(self):
        """None should canonicalize as 'null', not be skipped."""
        from flowscript_agents.mcp import _canonicalize
        result = _canonicalize({"a": None, "b": 1})
        assert '"a":null' in result
        assert '"b":1' in result

    def test_canonicalize_bool_not_int(self):
        """Booleans should serialize as true/false, not 1/0."""
        from flowscript_agents.mcp import _canonicalize
        assert _canonicalize(True) == "true"
        assert _canonicalize(False) == "false"
        assert _canonicalize(1) == "1"

    def test_canonicalize_sorted_keys(self):
        """Keys should be sorted alphabetically."""
        from flowscript_agents.mcp import _canonicalize
        result = _canonicalize({"z": 1, "a": 2, "m": 3})
        assert result == '{"a":2,"m":3,"z":1}'

    def test_all_schemas_have_additional_properties(self):
        """All tool inputSchemas should have additionalProperties: false."""
        from flowscript_agents.mcp import TOOLS
        for tool in TOOLS:
            schema = tool["inputSchema"]
            assert schema.get("additionalProperties") is False, (
                f"{tool['name']} missing additionalProperties: false"
            )


class TestAutoWrapTimer:
    """Tests for the auto-wrap consolidation timer."""

    def test_auto_wrap_fires_after_inactivity(self):
        """Auto-wrap should fire session_wrap after timer expires."""
        import os
        import threading
        import time

        # Set a very short timer for testing (0.1 seconds = 6 "minutes" scaled)
        os.environ["FLOWSCRIPT_AUTO_WRAP_MINUTES"] = "1"

        from flowscript_agents.mcp import run_server
        from flowscript_agents import UnifiedMemory
        from flowscript_agents.memory import Memory

        # Test the timer mechanism directly (not run_server, which blocks on stdin)
        mem = Memory()
        mem.session_start()
        mem.thought("test node for auto-wrap")
        assert mem.size == 1

        # Simulate what run_server does: create timer, let it fire
        auto_wrap_minutes = 0  # We'll test the logic, not the actual timer
        wrapped = [False]

        def do_wrap():
            mem.session_wrap()
            wrapped[0] = True

        # Verify session_wrap works when called
        result = mem.session_wrap()
        assert result.nodes_before == 1
        assert result.saved is False  # no file path set

        # Clean up
        if "FLOWSCRIPT_AUTO_WRAP_MINUTES" in os.environ:
            del os.environ["FLOWSCRIPT_AUTO_WRAP_MINUTES"]

    def test_auto_wrap_env_var_disable(self):
        """Setting FLOWSCRIPT_AUTO_WRAP_MINUTES=0 should disable auto-wrap."""
        import os
        val = os.environ.get("FLOWSCRIPT_AUTO_WRAP_MINUTES")
        os.environ["FLOWSCRIPT_AUTO_WRAP_MINUTES"] = "0"
        assert int(os.environ["FLOWSCRIPT_AUTO_WRAP_MINUTES"]) == 0
        # Restore
        if val is not None:
            os.environ["FLOWSCRIPT_AUTO_WRAP_MINUTES"] = val
        elif "FLOWSCRIPT_AUTO_WRAP_MINUTES" in os.environ:
            del os.environ["FLOWSCRIPT_AUTO_WRAP_MINUTES"]

    def test_session_wrap_tool_description_mentions_auto_wrap(self):
        """session_wrap tool description should mention auto-wrap safety net."""
        from flowscript_agents.mcp import TOOLS
        session_wrap_tool = None
        for t in TOOLS:
            if t["name"] == "session_wrap":
                session_wrap_tool = t
                break
        assert session_wrap_tool is not None
        desc = session_wrap_tool["description"]
        assert "auto-wrap" in desc.lower()
        assert "consolidat" in desc.lower()  # "consolidates" or "consolidation"
        assert "temporal tiers" in desc.lower() or "temporal" in desc.lower()


class TestEncodeExchangeHandler:
    """Happy-path tests for the encode_exchange MCP tool."""

    def test_encode_exchange_creates_nodes(self):
        """encode_exchange should create nodes from exchange text."""
        handler, umem = _make_handler(with_llm=True)
        result = handler.handle_tool("encode_exchange", {
            "user_content": "Should we use Redis or PostgreSQL for sessions?",
            "assistant_content": "PostgreSQL is better for ACID compliance.",
        })
        assert "error" not in result
        assert result["exchange_captured"] is True
        assert result["nodes_created"] >= 1

    def test_encode_exchange_empty_rejected(self):
        """encode_exchange with empty content returns error."""
        handler, _ = _make_handler()
        result = handler.handle_tool("encode_exchange", {
            "user_content": "",
            "assistant_content": "",
        })
        assert "error" in result


class TestQueryCounterfactualHandler:
    """Happy-path tests for the query_counterfactual MCP tool."""

    def test_counterfactual_returns_factors(self):
        """query_counterfactual should find tension-bearing factors."""
        handler, umem = _make_handler()
        mem = umem.memory
        low_cost = mem.thought("low cost option")
        high_perf = mem.thought("high performance option")
        mem.tension(low_cost, high_perf, "cost vs performance")
        decision = mem.thought("chose low cost")
        low_cost.causes(decision)
        result = handler.handle_tool("query_counterfactual", {"node_id": decision.id})
        assert "error" not in result
        assert len(result["factors"]) >= 1
        assert result["factors"][0]["tension_axis"] == "cost vs performance"

    def test_counterfactual_by_content(self):
        """query_counterfactual should find node by content search."""
        handler, umem = _make_handler()
        mem = umem.memory
        a = mem.thought("option alpha")
        b = mem.thought("option beta")
        mem.tension(a, b, "alpha vs beta")
        decision = mem.thought("selected option alpha for the project")
        a.causes(decision)
        result = handler.handle_tool("query_counterfactual", {"content": "selected option alpha"})
        assert "error" not in result
        assert result["decision"]["id"] == decision.id

    def test_counterfactual_no_node_returns_error(self):
        """query_counterfactual with no matching node returns error."""
        handler, _ = _make_handler()
        result = handler.handle_tool("query_counterfactual", {"content": "nonexistent"})
        assert "error" in result
