"""
FlowScript Unified Memory MCP Server.

Minimal MCP server (JSON-RPC over stdio) that wraps UnifiedMemory.
No external MCP SDK required — implements the protocol directly.

Zero-config quick start (auto-detects OPENAI_API_KEY):
    python -m flowscript_agents.mcp --memory ./agent.json
    # or, if installed via pip:
    flowscript-mcp --memory ./agent.json

Full config:
    python -m flowscript_agents.mcp --memory ./agent.json \\
        --embedder openai --llm-model gpt-4o-mini

Configure in your editor's MCP settings:

Claude Code — .claude/settings.json (project or ~/.claude/settings.json global):
{
  "mcpServers": {
    "flowscript": {
      "command": "flowscript-mcp",
      "args": ["--memory", "./agent-memory.json"],
      "env": { "OPENAI_API_KEY": "sk-..." }
    }
  }
}

Cursor / Windsurf / VS Code — .mcp.json in project root:
{
  "mcpServers": {
    "flowscript": {
      "type": "stdio",
      "command": "flowscript-mcp",
      "args": ["--memory", "./agent-memory.json"],
      "env": { "OPENAI_API_KEY": "sk-..." }
    }
  }
}

When OPENAI_API_KEY is set, the server auto-configures:
- OpenAI embeddings (text-embedding-3-small) for vector search
- LLM extraction (gpt-4o-mini) for typed reasoning extraction
- Consolidation (gpt-4o-mini) for memory management (UPDATE/RELATE/RESOLVE)

Session lifecycle:
- Auto-wrap safety net: consolidates memory after inactivity (default 5 min)
  Configure: FLOWSCRIPT_AUTO_WRAP_MINUTES=10 (or 0 to disable)
- Explicit session_wrap: LLM or user triggers consolidation at session end
- atexit wrap: final consolidation when process exits

Tools exposed (16: 15 verified + verify_integrity):
- search_memory: Unified search (vector + keyword + temporal)
- add_memory: Auto-extract reasoning from text with consolidation
- get_context: Get formatted memory for prompt injection
- query_tensions: Find all tensions/tradeoffs in memory
- query_blocked: Find all blocked items with impact analysis
- query_why: Trace causal chain for a node (returns structured data)
- query_what_if: Trace downstream impact
- query_alternatives: Reconstruct decision from options
- query_counterfactual: Counterfactual analysis (CJEU C-203/22)
- encode_exchange: Per-response exchange capture for AutoExtract pipeline
- remove_memory: Remove a node from memory
- session_wrap: Session consolidation (graduation, pruning, audit trail, save)
- memory_stats: Get memory statistics
- query_audit: Search the audit trail with filters
- verify_audit: Verify hash chain integrity
- verify_integrity: Verify tool description integrity (SRI for LLM prompts)
"""

from __future__ import annotations

import argparse
import atexit
import datetime
import hashlib
import json
import os
import sys
import threading
import time
from types import MappingProxyType
from typing import Any, Optional

from .memory import Memory
from .unified import UnifiedMemory
from .continuity import ContinuityManager
from .embeddings.providers import EmbeddingProvider
from .embeddings.consolidate import ConsolidationProvider


# =============================================================================
# MCP Protocol (JSON-RPC over stdio)
# =============================================================================

def _log(msg: str) -> None:
    """Log to stderr (stdout is reserved for JSON-RPC protocol)."""
    sys.stderr.write(f"[flowscript] {msg}\n")
    sys.stderr.flush()


_PROTOCOL_VERSION = "2025-03-26"
_SERVER_NAME = "flowscript-agents"
_SERVER_VERSION = "0.4.0"


def _jsonrpc_response(id: Any, result: Any) -> dict:
    return {"jsonrpc": "2.0", "id": id, "result": result}


def _jsonrpc_error(id: Any, code: int, message: str) -> dict:
    return {"jsonrpc": "2.0", "id": id, "error": {"code": code, "message": message}}


# =============================================================================
# Description Integrity — "SRI for LLM tool descriptions"
# =============================================================================
# Reference implementation: deterministic integrity verification for MCP servers.
# See: github.com/modelcontextprotocol/modelcontextprotocol/discussions/2402
#
# THREE-LAYER ARCHITECTURE:
#   1. Tool: verify_integrity — LLM-callable, detects in-process mutation
#   2. Resource: flowscript://integrity/manifest — Host-verifiable manifest
#      (enables Claude Code/Cursor to verify descriptions WITHOUT LLM involvement,
#       moving the security boundary to the correct layer)
#   3. Build-time manifest: tool-integrity.json — root of trust independent of
#      running process (generated via --generate-manifest)
#
# DETECTS:
#   - In-process description mutation (malicious dependency, monkey-patching,
#     or middleware that modifies tool dicts in the same Python process)
#   - Accidental mutation (buggy wrapper that string-replaces descriptions)
#
# DOES NOT DETECT (requires ecosystem-level changes):
#   - Supply chain attacks (poisoned before startup — manifest captures poisoned state)
#   - Transport-layer attacks (MITM between server and client — hashes never leave process)
#   - Client-side injection (host manipulates descriptions after receiving them)
#   - Reflection-based bypass: gc.get_referents() can reach the underlying dict
#     behind MappingProxyType. ctypes can write to arbitrary memory. Deep-freeze
#     is best-effort against casual/accidental mutation. For determined in-process
#     attackers, the build-time manifest is the correct verification layer.
#   - Filesystem manifest replacement: if an attacker can write to the package
#     directory, they can replace tool-integrity.json to match poisoned definitions.
#     In high-security deployments, sign the manifest or distribute via separate
#     trust channel.
#
# This is a reference implementation. Full integrity requires client-side verification
# against an out-of-band manifest (build-time hashes, package signatures, etc.).


def _canonicalize(obj: Any) -> str:
    """Canonicalize a JSON-serializable value for deterministic hashing.

    Sorted keys, no whitespace, deterministic primitive serialization.
    Matches the TypeScript MCP server's canonicalize() for cross-language
    consistency (though hash comparison is per-server, not cross-language).
    """
    if obj is None:
        return "null"
    if isinstance(obj, bool):
        return "true" if obj else "false"
    if isinstance(obj, (int, float)):
        return json.dumps(obj)
    if isinstance(obj, str):
        return json.dumps(obj, ensure_ascii=True)
    if isinstance(obj, (list, tuple)):
        return "[" + ",".join(_canonicalize(v) for v in obj) + "]"
    if isinstance(obj, (dict, MappingProxyType)):
        entries = []
        for k in sorted(obj.keys()):
            v = obj[k]
            # Include None as "null" (matches TS which keeps null but skips undefined).
            # Python dicts don't have "undefined" — all present keys are serialized.
            entries.append(json.dumps(k, ensure_ascii=True) + ":" + _canonicalize(v))
        return "{" + ",".join(entries) + "}"
    return json.dumps(str(obj), ensure_ascii=True)


def _hash_tool_definition(tool: dict | MappingProxyType) -> str:
    """Compute SHA-256 hash of a canonical JSON representation of a tool definition."""
    canonical = _canonicalize(tool)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _thaw(obj: Any) -> Any:
    """Recursively convert MappingProxyType back to plain dicts for JSON serialization."""
    if isinstance(obj, MappingProxyType):
        return {k: _thaw(v) for k, v in obj.items()}
    if isinstance(obj, tuple):
        return [_thaw(x) for x in obj]
    if isinstance(obj, list):
        return [_thaw(x) for x in obj]
    return obj


def _deep_freeze(obj: dict) -> MappingProxyType:
    """Recursively convert a dict tree to immutable MappingProxyType.

    Any attempt to mutate a frozen dict raises TypeError.
    Lists inside are converted to tuples (also immutable).
    """
    frozen = {}
    for k, v in obj.items():
        if isinstance(v, dict):
            frozen[k] = _deep_freeze(v)
        elif isinstance(v, list):
            frozen[k] = tuple(_deep_freeze(x) if isinstance(x, dict) else x for x in v)
        else:
            frozen[k] = v
    return MappingProxyType(frozen)


# =============================================================================
# Tool definitions
# =============================================================================

# Defined as plain dicts first, then frozen after definition.
# The verify_integrity tool is NOT in this list (it verifies, it isn't verified).
_TOOL_DEFS_RAW = [
    {
        "name": "search_memory",
        "description": (
            "Search agent memory using unified ranking (vector similarity + keyword "
            "matching + temporal intelligence). Call this to recall prior context before "
            "making decisions, or whenever the conversation touches topics that may "
            "have prior reasoning context. "
            "Use mode='vector' for pure semantic search, 'keyword' for exact matching, "
            "or 'unified' (default) for combined ranking."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query text"},
                "top_k": {"type": "integer", "description": "Max results (default 10)", "default": 10},
                "mode": {
                    "type": "string",
                    "enum": ["unified", "vector", "keyword"],
                    "description": "Search mode: unified (default), vector (semantic only), keyword (exact only)",
                    "default": "unified",
                },
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    },
    {
        "name": "add_memory",
        "description": (
            "Add information to agent memory. Capture reasoning, not just conclusions — "
            "WHY you decided matters more than WHAT you decided. Call this when important "
            "decisions are made, architectural tradeoffs are discussed, blockers are "
            "identified, causal relationships are established, or any reasoning worth "
            "preserving occurs in conversation. Include full context — the extraction "
            "layer automatically identifies and types the reasoning structures (decisions "
            "with rationale, tensions with axes, causal chains, blockers with reasons). "
            "Do NOT store routine code changes, transient debugging steps, or "
            "information already tracked in git. "
            "Returns extraction results including node count and dedup info."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "Text to add to memory — include full reasoning context"},
                "metadata": {
                    "type": "object",
                    "description": "Optional metadata to attach to created nodes",
                },
            },
            "required": ["text"],
            "additionalProperties": False,
        },
    },
    {
        "name": "get_context",
        "description": (
            "Get formatted memory content for prompt context. Call this at the start "
            "of sessions to load relevant memory, or periodically during long sessions "
            "to check what reasoning has been preserved. Returns nodes sorted by "
            "tier and frequency, with tier labels. Use max_tokens to control size."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "max_tokens": {
                    "type": "integer",
                    "description": "Maximum tokens for context (default 4000)",
                    "default": 4000,
                },
            },
            "additionalProperties": False,
        },
    },
    {
        "name": "query_tensions",
        "description": (
            "Find all tensions and tradeoffs in memory. Call this when evaluating "
            "tradeoffs, before making decisions that might conflict with prior choices, "
            "or when the user asks about competing concerns. Returns tension pairs "
            "grouped by axis."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "group_by": {
                    "type": "string",
                    "enum": ["axis", "node", "flat"],
                    "description": "How to group tensions (default: axis)",
                    "default": "axis",
                },
            },
            "additionalProperties": False,
        },
    },
    {
        "name": "query_blocked",
        "description": (
            "Find all blocked items in memory with impact analysis. Call this when "
            "planning work, when progress stalls, or to check what's waiting on "
            "external dependencies. Returns blockers sorted by impact score "
            "(downstream effects), with reason, duration, and transitive causes."
        ),
        "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False},
    },
    {
        "name": "query_why",
        "description": (
            "Trace the causal chain for a specific memory node. Call this when "
            "the user asks 'why did we decide X' or when you need to understand "
            "the reasoning behind a prior decision. Returns root cause, intermediate "
            "steps, and the full chain. Search by content to find the node, or "
            "provide a node_id if known from a prior search."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "node_id": {"type": "string", "description": "Node ID to trace"},
                "content": {"type": "string", "description": "Search for node by content (alternative to node_id)"},
            },
            "additionalProperties": False,
        },
    },
    {
        "name": "query_alternatives",
        "description": (
            "Reconstruct a decision from its alternatives. Call this when revisiting "
            "decisions or when the user asks what options were considered. Shows all "
            "options, which was chosen, rejection rationale, and consequences. "
            "Search by content to find the question, or provide a question_id if known."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "question_id": {"type": "string", "description": "Question node ID"},
                "content": {"type": "string", "description": "Search for question by content (alternative to question_id)"},
            },
            "additionalProperties": False,
        },
    },
    {
        "name": "query_what_if",
        "description": (
            "Forward impact analysis: what happens if a node changes? Call this "
            "when considering changes to understand downstream consequences before "
            "committing. Traces direct and indirect effects, finds tensions in "
            "the impact zone."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "node_id": {"type": "string", "description": "Node ID to analyze"},
                "content": {"type": "string", "description": "Search for node by content (alternative to node_id)"},
            },
            "additionalProperties": False,
        },
    },
    {
        "name": "remove_memory",
        "description": (
            "Remove a specific memory node by ID. Call this to correct mistakes — "
            "if something was stored incorrectly, a decision was reversed, or "
            "information is no longer relevant. Also removes associated relationships "
            "and states. Use search_memory first to find the node_id to remove."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "node_id": {"type": "string", "description": "ID of the node to remove"},
            },
            "required": ["node_id"],
            "additionalProperties": False,
        },
    },
    {
        "name": "session_wrap",
        "description": (
            "Run memory lifecycle maintenance — compress with judgment, not just storage. "
            "Prune dormant nodes to audit trail, graduate frequently-accessed "
            "knowledge through temporal tiers, extract patterns and principles from "
            "session data, save to disk. Call this at the end of a work session or when "
            "the user says to wrap up. Just like sleep consolidates human memory, session "
            "wraps let the reasoning graph mature: knowledge that keeps getting queried "
            "earns its place, one-off observations fade naturally. The act of compression "
            "is itself a form of thinking — patterns emerge that weren't visible in the "
            "raw data. An auto-wrap safety net runs after inactivity, but explicit wraps "
            "at session boundaries produce the best results. Archived nodes are preserved "
            "in the audit trail with full provenance — never destroyed."
        ),
        "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False},
    },
    {
        "name": "memory_stats",
        "description": (
            "Get memory statistics: node count, tier distribution, garden health, "
            "embedding status. Call this to understand the current state of memory."
        ),
        "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False},
    },
    {
        "name": "query_audit",
        "description": (
            "Search the audit trail for reasoning provenance. Call this to understand "
            "how memory evolved — what was extracted, what consolidation decided, "
            "which adapter made changes, or what happened in a specific session. "
            "Returns hash-chained audit entries matching the filters."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "after": {"type": "string", "description": "Only entries after this ISO timestamp"},
                "before": {"type": "string", "description": "Only entries before this ISO timestamp"},
                "events": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Filter by event types. Available: node_create, relationship_create, "
                        "state_change, graduation, prune, session_start, session_end, "
                        "session_wrap, consolidation, consolidation_batch, transcript_extract, "
                        "node_remove, update_node, update_node_merge, audit_cleanup"
                    ),
                },
                "node_id": {"type": "string", "description": "Filter by node involvement"},
                "session_id": {"type": "string", "description": "Filter by session ID"},
                "adapter": {"type": "string", "description": "Filter by adapter framework name"},
                "limit": {"type": "integer", "description": "Maximum entries (default 100)", "default": 100},
                "verify_chain": {
                    "type": "boolean",
                    "description": "Also verify hash chain integrity of matched entries",
                    "default": False,
                },
            },
            "additionalProperties": False,
        },
    },
    {
        "name": "verify_audit",
        "description": (
            "Verify hash chain integrity of the entire audit trail. Call this to "
            "confirm the audit trail has not been tampered with. Returns chain "
            "validity status, total entries verified, and location of any break."
        ),
        "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False},
    },
    {
        "name": "query_counterfactual",
        "description": (
            "Counterfactual analysis: what would need to change for a different "
            "outcome? Satisfies CJEU Case C-203/22 requirement for counterfactual "
            "explanations — not just 'why this' but 'why not that.' Walks backward "
            "through the causal chain from a decision, finds tension-bearing "
            "ancestors (pivotal factors), and identifies what conditions, if "
            "different, would have led to a different outcome. Deterministic — "
            "pure graph traversal, no LLM. Returns pivotal factors ranked "
            "deepest-first (most fundamental first)."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "node_id": {"type": "string", "description": "Decision node ID to analyze"},
                "content": {"type": "string", "description": "Search for node by content (alternative to node_id)"},
                "max_depth": {
                    "type": "integer",
                    "description": "Maximum causal chain depth to traverse (default: unlimited)",
                },
            },
            "additionalProperties": False,
        },
    },
    {
        "name": "encode_exchange",
        "description": (
            "Encode a user-assistant exchange into the reasoning memory graph "
            "with typed extraction (decisions, tensions, causal chains) and "
            "hash-chained audit trail. When your instructions say to call this "
            "after every response, do so — it captures the full reasoning chain "
            "from each exchange. Pass the user's message and your response. The "
            "extraction engine automatically identifies reasoning structures. "
            "Lightweight call — heavy lifting happens server-side."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "user_content": {
                    "type": "string",
                    "description": "The user's message from this exchange",
                },
                "assistant_content": {
                    "type": "string",
                    "description": "Your (the assistant's) response to the user",
                },
            },
            "required": ["user_content", "assistant_content"],
            "additionalProperties": False,
        },
    },
]

# Deep-freeze all tool definitions — any in-process mutation raises TypeError.
TOOLS: list[MappingProxyType] = [_deep_freeze(t) for t in _TOOL_DEFS_RAW]

# Compute integrity manifest at startup — captures the "intended" state.
_INTEGRITY_MANIFEST: dict[str, str] = {}
_EXPECTED_TOOL_COUNT = len(TOOLS)
for _t in TOOLS:
    _INTEGRITY_MANIFEST[_t["name"]] = _hash_tool_definition(_t)
_INTEGRITY_MANIFEST = MappingProxyType(_INTEGRITY_MANIFEST)  # type: ignore[assignment]

# Load build-time manifest if available (generated via --generate-manifest).
_BUILD_TIME_MANIFEST: dict[str, str] | None = None
try:
    _manifest_path = os.path.join(os.path.dirname(__file__), "tool-integrity.json")
    with open(_manifest_path) as _f:
        _BUILD_TIME_MANIFEST = json.load(_f)
    _log(f"Integrity: loaded build-time manifest ({len(_BUILD_TIME_MANIFEST)} tools)")
except (FileNotFoundError, json.JSONDecodeError):
    pass  # No build-time manifest — startup-only verification

# The verify_integrity tool — separate from the verified tools.
_VERIFY_INTEGRITY_TOOL = _deep_freeze({
    "name": "verify_integrity",
    "description": (
        "Verify that tool descriptions have not been mutated in-process since "
        "server startup. Detects description modifications by malicious dependencies, "
        "middleware, or monkey-patching. Returns per-tool SHA-256 hashes (expected vs "
        "current) and a pass/fail verdict. NOTE: This verifies the server's own state "
        "— transport-layer integrity requires host-level verification via the "
        "flowscript://integrity/manifest resource. "
        "Reference implementation: "
        "github.com/modelcontextprotocol/modelcontextprotocol/discussions/2402"
    ),
    "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False},
})

# Full tool list exposed to clients: verified tools + the verifier
ALL_TOOLS: list[MappingProxyType] = [*TOOLS, _VERIFY_INTEGRITY_TOOL]

# Integrity resource definition (frozen for consistency)
_INTEGRITY_RESOURCE = _deep_freeze({
    "uri": "flowscript://integrity/manifest",
    "name": "Tool Integrity Manifest",
    "description": (
        "SHA-256 hashes of all tool definitions for client-side integrity "
        "verification. Compare these hashes against the tool definitions you "
        "received to detect transport-layer description mutation."
    ),
    "mimeType": "application/json",
})


# =============================================================================
# Tool handlers
# =============================================================================


class MCPHandler:
    """Handles MCP tool calls against a UnifiedMemory instance."""

    def __init__(
        self,
        umem: UnifiedMemory,
        continuity_manager: Optional[ContinuityManager] = None,
        memory_path: Optional[str] = None,
    ) -> None:
        self._umem = umem
        self._continuity_mgr = continuity_manager
        self._memory_path = memory_path

    def handle_tool(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        handlers = {
            "search_memory": self._search_memory,
            "add_memory": self._add_memory,
            "get_context": self._get_context,
            "query_tensions": self._query_tensions,
            "query_blocked": self._query_blocked,
            "query_why": self._query_why,
            "query_what_if": self._query_what_if,
            "query_alternatives": self._query_alternatives,
            "remove_memory": self._remove_memory,
            "session_wrap": self._session_wrap,
            "memory_stats": self._memory_stats,
            "query_audit": self._query_audit,
            "verify_audit": self._verify_audit,
            "query_counterfactual": self._query_counterfactual,
            "encode_exchange": self._encode_exchange,
            "verify_integrity": self._verify_integrity,
        }
        handler = handlers.get(name)
        if handler is None:
            return {"error": f"Unknown tool: {name}"}
        try:
            return handler(args)
        except (KeyError, ValueError) as e:
            return {"error": f"{type(e).__name__}: {e}"}
        except Exception as e:
            _log(f"Tool error in {name}: {type(e).__name__}: {e}")
            return {"error": f"Internal error in {name} — check server logs"}

    def _search_memory(self, args: dict) -> dict:
        query = args.get("query", "")
        top_k = args.get("top_k", 10)
        mode = args.get("mode", "unified")

        if mode == "vector":
            results = self._umem.vector_search(query, top_k=top_k)
            return {
                "results": [
                    {
                        "content": r.content,
                        "score": round(r.score, 4),
                        "node_id": r.node_id,
                        "type": r.node_type,
                        "tier": r.tier,
                        "frequency": r.frequency,
                    }
                    for r in results
                ],
                "mode": "vector",
                "count": len(results),
            }
        elif mode == "keyword":
            results = self._umem.search(
                query, top_k=top_k,
                vector_weight=0.0, keyword_weight=0.8, temporal_weight=0.2,
            )
        else:  # unified
            results = self._umem.search(query, top_k=top_k)

        return {
            "results": [
                {
                    "content": r.content,
                    "score": round(r.combined_score, 4),
                    "node_id": r.node_id,
                    "type": r.node_type,
                    "tier": r.tier,
                    "frequency": r.frequency,
                    "sources": r.sources,
                }
                for r in results
            ],
            "mode": mode,
            "count": len(results),
        }

    def _add_memory(self, args: dict) -> dict:
        # Accept both "text" and "content" — LLMs frequently use "content"
        text = args.get("text", "") or args.get("content", "")
        if not text or not text.strip():
            return {"error": "text is required and must not be empty"}
        metadata = args.get("metadata")
        result = self._umem.add(text, metadata=metadata, actor="agent")
        resp: dict[str, Any] = {
            "nodes_created": result.nodes_created,
            "nodes_deduplicated": result.nodes_deduplicated,
            "relationships_created": result.relationships_created,
            "states_created": result.states_created,
            "node_ids": result.node_ids,
        }
        # Hint when running without LLM (raw storage only)
        if self._umem.extractor is None and result.nodes_created == 1:
            resp["note"] = (
                "Running without LLM — stored as raw text (no typed extraction). "
                "Install openai and set OPENAI_API_KEY for automatic reasoning extraction."
            )
        return resp

    def _get_context(self, args: dict) -> dict:
        max_tokens = args.get("max_tokens", 4000)
        context = self._umem.get_context(max_tokens=max_tokens)

        result: dict[str, Any] = {"context": context, "nodes": self._umem.size}

        # Include continuity file when Layer 1 is enabled
        if self._continuity_mgr and self._memory_path:
            continuity_text = ContinuityManager.load(self._memory_path)
            if continuity_text:
                result["continuity"] = continuity_text
                result["continuity_chars"] = len(continuity_text)

        return result

    def _query_tensions(self, args: dict) -> dict:
        group_by = args.get("group_by", "axis")
        result = self._umem.memory.query.tensions(group_by=group_by)
        # Serialize the result
        return _serialize_query_result(result)

    def _query_blocked(self, args: dict) -> dict:
        result = self._umem.memory.query.blocked()
        return _serialize_query_result(result)

    def _query_why(self, args: dict) -> dict:
        node_id = args.get("node_id")
        content = args.get("content")
        if not node_id and content:
            refs = self._umem.memory.find_nodes(content)
            if refs:
                node_id = refs[0].id
        if not node_id:
            return {"error": "No node found. Provide node_id or searchable content."}
        result = self._umem.memory.query.why(node_id)
        return _serialize_query_result(result)

    def _query_what_if(self, args: dict) -> dict:
        node_id = args.get("node_id")
        content = args.get("content")
        if not node_id and content:
            refs = self._umem.memory.find_nodes(content)
            if refs:
                node_id = refs[0].id
        if not node_id:
            return {"error": "No node found. Provide node_id or searchable content."}
        result = self._umem.memory.query.what_if(node_id)
        return _serialize_query_result(result)

    def _query_alternatives(self, args: dict) -> dict:
        question_id = args.get("question_id")
        content = args.get("content")
        if not question_id and content:
            refs = self._umem.memory.find_nodes(content)
            if refs:
                question_id = refs[0].id
        if not question_id:
            return {"error": "No question found. Provide question_id or searchable content."}
        result = self._umem.memory.query.alternatives(question_id)
        return _serialize_query_result(result)

    def _query_counterfactual(self, args: dict) -> dict:
        node_id = args.get("node_id")
        content = args.get("content")
        if not node_id and content:
            refs = self._umem.memory.find_nodes(content)
            if refs:
                node_id = refs[0].id
        if not node_id:
            return {"error": "No node found. Provide node_id or searchable content."}
        max_depth = args.get("max_depth")
        kwargs = {}
        if max_depth is not None:
            kwargs["max_depth"] = max_depth
        result = self._umem.memory.query.counterfactual(node_id, **kwargs)
        return _serialize_query_result(result)

    def _remove_memory(self, args: dict) -> dict:
        node_id = args.get("node_id", "")
        if not node_id:
            return {"error": "node_id is required"}
        # Remove from graph first, then clean up vector index
        removed = self._umem.memory.remove_node(node_id)
        if removed and self._umem.vector_index:
            self._umem.vector_index.remove_node(node_id)
        return {"removed": removed, "node_id": node_id}

    def _session_wrap(self, args: dict) -> dict:
        # Produce continuity file BEFORE session_wrap prunes nodes
        # (we want the full session data for compression)
        continuity_result = None
        if self._continuity_mgr and self._memory_path:
            try:
                meta = ContinuityManager.load_meta(self._memory_path)
                existing = ContinuityManager.load(self._memory_path)
                continuity_result = self._continuity_mgr.produce(
                    self._umem.memory,
                    existing_continuity=existing,
                    citations_seen=meta.get("citations_seen", False),
                )
                self._continuity_mgr.save(continuity_result.text, self._memory_path)
                # Update metadata
                meta["sessions_produced"] = meta.get("sessions_produced", 0) + 1
                if continuity_result.graduations_validated > 0:
                    meta["citations_seen"] = True
                ContinuityManager.save_meta(meta, self._memory_path)
            except Exception as e:
                _log(f"Continuity production failed: {e}")
                # Non-fatal — session_wrap still proceeds

        result = self._umem.memory.session_wrap()

        response: dict[str, Any] = {
            "nodes_before": result.nodes_before,
            "tiers_before": result.tiers_before,
            "nodes_after": result.nodes_after,
            "tiers_after": result.tiers_after,
            "nodes_pruned": result.pruned.count,
            "pruned_node_ids": result.pruned.archived,
            "garden_after": {
                "growing": len(result.garden_after.growing),
                "resting": len(result.garden_after.resting),
                "dormant": len(result.garden_after.dormant),
            },
            "saved": result.saved,
            "path": result.path,
        }

        # Always include continuity key so callers can distinguish disabled/error/success.
        if continuity_result:
            response["continuity"] = {
                "produced": True,
                "char_count": continuity_result.char_count,
                "section_sizes": continuity_result.section_sizes,
                "patterns_extracted": continuity_result.patterns_extracted,
                "truncated": continuity_result.truncated,
                "path": ContinuityManager.continuity_path(self._memory_path),
            }
        elif self._continuity_mgr:
            response["continuity"] = {"produced": False, "reason": "error"}
        else:
            response["continuity"] = {"produced": False, "reason": "disabled"}

        return response

    def _memory_stats(self, args: dict) -> dict:
        mem = self._umem.memory
        tiers = mem.count_tiers()
        garden = mem.garden()
        stats = {
            "total_nodes": mem.size,
            "tiers": tiers,
            "garden": {
                "growing": len(garden.growing),
                "resting": len(garden.resting),
                "dormant": len(garden.dormant),
            },
            "relationships": mem.relationship_count,
            "states": mem.state_count,
        }
        if self._umem.vector_index:
            stats["embeddings"] = {
                "indexed": self._umem.vector_index.indexed_count,
                "provider": repr(self._umem.vector_index._provider),
            }
        return stats


    def _get_audit_path(self) -> str | None:
        """Derive audit trail path from memory file path."""
        mem_path = self._umem.memory._file_path
        if not mem_path:
            return None
        from pathlib import Path as _P
        return str(_P(mem_path).parent / (_P(mem_path).stem + ".audit.jsonl"))

    def _query_audit(self, args: dict) -> dict:
        audit_path = self._get_audit_path()
        if not audit_path:
            return {"error": "No memory file path — audit trail requires file-based persistence"}
        try:
            result = Memory.query_audit(
                audit_path,
                after=args.get("after"),
                before=args.get("before"),
                events=args.get("events"),
                node_id=args.get("node_id"),
                session_id=args.get("session_id"),
                adapter=args.get("adapter"),
                limit=args.get("limit", 100),
                verify_chain=args.get("verify_chain", False),
            )
            resp: dict[str, Any] = {
                "entries": result.entries,
                "total_scanned": result.total_scanned,
                "files_searched": result.files_searched,
                "count": len(result.entries),
            }
            if result.chain_valid is not None:
                resp["chain_valid"] = result.chain_valid
                if result.chain_break_at is not None:
                    resp["chain_break_at"] = result.chain_break_at
            return resp
        except FileNotFoundError:
            return {"entries": [], "total_scanned": 0, "files_searched": 0, "count": 0,
                    "note": "No audit trail file found — audit may not be configured"}

    def _verify_audit(self, args: dict) -> dict:
        audit_path = self._get_audit_path()
        if not audit_path:
            return {"error": "No memory file path — audit trail requires file-based persistence"}
        try:
            result = Memory.verify_audit(audit_path)
            resp: dict[str, Any] = {
                "valid": result.valid,
                "total_entries": result.total_entries,
                "files_verified": result.files_verified,
                "legacy_entries": result.legacy_entries,
            }
            if result.valid is False:
                if result.chain_break_at is not None:
                    resp["chain_break_at"] = result.chain_break_at
                if result.chain_break_file is not None:
                    resp["chain_break_file"] = result.chain_break_file
            return resp
        except FileNotFoundError:
            return {"valid": None, "total_entries": 0, "files_verified": 0,
                    "status": "no_audit_trail",
                    "note": "No audit trail file found — auditing may not be configured"}

    def _encode_exchange(self, args: dict) -> dict:
        """Encode a user-assistant exchange into the reasoning memory graph."""
        user_content = args.get("user_content", "").strip()
        assistant_content = args.get("assistant_content", "").strip()

        if not user_content and not assistant_content:
            return {"error": "At least one of user_content or assistant_content must be non-empty"}

        # Format as exchange — same format used by the SDK client wrapper
        parts = []
        if user_content:
            parts.append(f"User: {user_content}")
        if assistant_content:
            parts.append(f"Assistant: {assistant_content}")
        exchange_text = "\n".join(parts)

        # Feed through the existing extraction pipeline
        result = self._umem.add(exchange_text, actor="agent")

        return {
            "nodes_created": result.nodes_created,
            "nodes_deduplicated": result.nodes_deduplicated,
            "relationships_created": result.relationships_created,
            "states_created": result.states_created,
            "node_ids": result.node_ids,
            "exchange_captured": True,
        }

    def _verify_integrity(self, args: dict) -> dict:
        """Verify in-process description integrity of all tool definitions."""
        results = []
        all_passed = True

        # Check: has the tool count changed? (detect additions/removals)
        count_match = len(TOOLS) == _EXPECTED_TOOL_COUNT
        if not count_match:
            all_passed = False

        # Per-tool hash verification
        for tool in TOOLS:
            tool_name = tool["name"]
            expected = _INTEGRITY_MANIFEST[tool_name]
            current = _hash_tool_definition(tool)
            passed = expected == current
            if not passed:
                all_passed = False

            entry: dict[str, Any] = {
                "tool": tool_name,
                "expected_hash": expected,
                "current_hash": current,
                "status": "pass" if passed else "fail",
            }

            # Compare against build-time manifest if available
            if _BUILD_TIME_MANIFEST:
                build_hash = _BUILD_TIME_MANIFEST.get(tool_name)
                if build_hash:
                    build_match = build_hash == current
                    if not build_match:
                        all_passed = False
                    entry["build_time_status"] = "pass" if build_match else "fail"
                else:
                    entry["build_time_status"] = "no_manifest"

            results.append(entry)

        verdict = "PASS" if all_passed else "FAIL"
        return {
            "success": True,
            "verdict": verdict,
            "tool_count": len(TOOLS),
            "expected_tool_count": _EXPECTED_TOOL_COUNT,
            "count_match": count_match,
            "algorithm": "SHA-256",
            "canonicalization": "deterministic sorted-keys JSON",
            "build_time_manifest": "verified" if _BUILD_TIME_MANIFEST else "not available",
            "tools": results,
            "scope": (
                "Verifies in-process description integrity (detects mutation by "
                "dependencies, middleware, or monkey-patching). Transport-layer "
                "integrity requires host-side verification via "
                "flowscript://integrity/manifest resource."
            ),
            "description": (
                "All tool descriptions match their startup hashes. "
                "No in-process mutation detected."
                if all_passed else
                "WARNING: Tool description integrity violation detected. "
                "One or more definitions have been modified since server startup."
            ),
        }


def _serialize_query_result(result: Any, _seen: set | None = None) -> dict:
    """Best-effort serialization of query result dataclasses."""
    if _seen is None:
        _seen = set()
    obj_id = id(result)
    if obj_id in _seen:
        return {"_circular": str(type(result).__name__)}
    _seen.add(obj_id)
    if hasattr(result, "__dict__"):
        d = {}
        for k, v in result.__dict__.items():
            if k.startswith("_"):
                continue
            d[k] = _serialize_value(v, _seen)
        return d
    return {"result": str(result)}


def _serialize_value(v: Any, _seen: set | None = None) -> Any:
    if _seen is None:
        _seen = set()
    if isinstance(v, (str, int, float, bool, type(None))):
        return v
    if isinstance(v, (list, tuple)):
        return [_serialize_value(x, _seen) for x in v]
    if isinstance(v, dict):
        return {str(k): _serialize_value(val, _seen) for k, val in v.items()}
    if hasattr(v, "__dict__"):
        return _serialize_query_result(v, _seen)
    return str(v)


# =============================================================================
# MCP Server loop
# =============================================================================


def _create_embedder(provider: str, **kwargs: Any) -> Optional[EmbeddingProvider]:
    """Create an embedding provider by name."""
    if provider == "openai":
        from .embeddings.providers import OpenAIEmbeddings
        return OpenAIEmbeddings(**kwargs)
    elif provider == "sentence-transformers":
        from .embeddings.providers import SentenceTransformerEmbeddings
        return SentenceTransformerEmbeddings(**kwargs)
    elif provider == "ollama":
        from .embeddings.providers import OllamaEmbeddings
        return OllamaEmbeddings(**kwargs)
    return None


class _OpenAIConsolidationProvider:
    """OpenAI consolidation provider using tool calling for memory management."""

    def __init__(self, model: str = "gpt-4o-mini", client: Any = None) -> None:
        if client is not None:
            self._client = client
        else:
            try:
                import openai
            except ImportError:
                raise ImportError(
                    "Auto-configuration requires the openai package. "
                    "Install with: pip install openai"
                )
            self._client = openai.OpenAI()
        self._model = model

    def tool_call(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        resp = self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            tools=tools,
            tool_choice="required",
            temperature=0.1,
        )
        results = []
        for tc in resp.choices[0].message.tool_calls or []:
            results.append({
                "name": tc.function.name,
                "arguments": json.loads(tc.function.arguments),
            })
        return results


class _AnthropicConsolidationProvider:
    """Anthropic consolidation provider using Claude tool calling.

    Translates between the OpenAI-format ConsolidationProvider protocol
    and Anthropic's native tool calling API.
    """

    def __init__(self, model: str = "claude-haiku-4-5-20251001", client: Any = None) -> None:
        if client is not None:
            self._client = client
        else:
            try:
                import anthropic
            except ImportError:
                raise ImportError(
                    "Anthropic auto-configuration requires the anthropic package. "
                    "Install with: pip install anthropic"
                )
            self._client = anthropic.Anthropic()
        self._model = model

    def tool_call(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        # Translate OpenAI-format tools to Anthropic format
        anthropic_tools = []
        for t in tools:
            fn = t.get("function", t)  # handle both wrapped and flat formats
            anthropic_tools.append({
                "name": fn.get("name", t.get("name")),
                "description": fn.get("description", t.get("description", "")),
                "input_schema": fn.get("parameters", t.get("inputSchema", t.get("parameters", {}))),
            })

        # Separate system message from user messages
        system_msg = None
        user_messages = []
        for m in messages:
            if m.get("role") == "system":
                system_msg = m["content"]
            else:
                user_messages.append({"role": m["role"], "content": m["content"]})

        kwargs: dict[str, Any] = {
            "model": self._model,
            "max_tokens": 4096,
            "tools": anthropic_tools,
            "tool_choice": {"type": "any"},
            "messages": user_messages,
        }
        if system_msg:
            kwargs["system"] = system_msg

        resp = self._client.messages.create(**kwargs)

        results = []
        for block in resp.content:
            if hasattr(block, "type") and block.type == "tool_use":
                results.append({
                    "name": block.name,
                    "arguments": block.input,
                })
        return results


def _auto_configure_anthropic(
    model: str = "claude-haiku-4-5-20251001",
) -> tuple[None, Any, _AnthropicConsolidationProvider]:
    """Auto-configure Anthropic extraction LLM + consolidation provider.

    Returns (None, llm_fn, consolidation_provider).
    Embedder is None — Anthropic has no embeddings API.
    Uses keyword-only search (still functional, just no vector similarity).
    """
    try:
        import anthropic
    except ImportError:
        raise ImportError(
            "Anthropic auto-configuration requires the anthropic package. "
            "Install with: pip install anthropic"
        )

    client = anthropic.Anthropic()

    def llm_extract(prompt: str) -> str:
        resp = client.messages.create(
            model=model,
            max_tokens=4096,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.content[0].text if resp.content else ""

    consolidation = _AnthropicConsolidationProvider(model=model, client=client)

    return None, llm_extract, consolidation


def _auto_configure_openai(
    model: str = "gpt-4o-mini",
    embedding_model: str | None = None,
) -> tuple[EmbeddingProvider, Any, _OpenAIConsolidationProvider]:
    """Auto-configure OpenAI embedder + extraction LLM + consolidation provider.

    Returns (embedder, llm_fn, consolidation_provider).
    Requires OPENAI_API_KEY in environment and the openai package installed.
    Uses a single shared OpenAI client for extraction and consolidation.
    """
    try:
        import openai
    except ImportError:
        raise ImportError(
            "Auto-configuration requires the openai package. "
            "Install with: pip install openai"
        )

    from .embeddings.providers import OpenAIEmbeddings

    client = openai.OpenAI()

    emb_kwargs: dict[str, Any] = {}
    if embedding_model:
        emb_kwargs["model"] = embedding_model
    embedder = OpenAIEmbeddings(**emb_kwargs)

    def llm_extract(prompt: str) -> str:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
        )
        return resp.choices[0].message.content or ""

    consolidation = _OpenAIConsolidationProvider(model=model, client=client)

    return embedder, llm_extract, consolidation


def run_server(
    memory_path: str,
    embedder: Optional[EmbeddingProvider] = None,
    llm: Optional[Any] = None,
    consolidation_provider: Optional[Any] = None,
) -> None:
    """Run the MCP server over stdio."""
    umem = UnifiedMemory(
        file_path=memory_path,
        embedder=embedder,
        llm=llm,
        consolidation_provider=consolidation_provider,
    )
    # Start session tracking — enables touch deduplication and temporal
    # intelligence across the lifetime of this MCP server instance.
    umem.memory.session_start()
    umem.memory.set_adapter_context("mcp", "FlowScriptMCP", "server")

    # Layer 1: Continuity (LLM-compressed session boundary file)
    continuity_mgr: Optional[ContinuityManager] = None
    if os.environ.get("FLOWSCRIPT_CONTINUITY", "").lower() in ("true", "1", "yes"):
        if llm is not None:
            continuity_mgr = ContinuityManager(llm=llm)
            _log("Layer 1 (Continuity) enabled — session wraps produce compressed memory file")
        else:
            _log("Warning: FLOWSCRIPT_CONTINUITY=true but no LLM configured — continuity disabled")

    handler = MCPHandler(umem, continuity_manager=continuity_mgr, memory_path=memory_path)

    # -------------------------------------------------------------------------
    # Auto-wrap timer: consolidation safety net for when the LLM or user
    # doesn't explicitly call session_wrap. Just like sleep consolidates
    # human memory, auto-wrap ensures the reasoning graph matures even if
    # the session boundary isn't explicitly marked.
    #
    # - Resets on every tool call (activity = timer restart)
    # - Fires after FLOWSCRIPT_AUTO_WRAP_MINUTES of inactivity (default 5)
    # - Set to 0 to disable
    # - atexit handler provides a final wrap on process exit
    # -------------------------------------------------------------------------
    auto_wrap_minutes = int(os.environ.get("FLOWSCRIPT_AUTO_WRAP_MINUTES", "5"))
    _auto_wrap_timer: list[Optional[threading.Timer]] = [None]  # mutable container for closure
    _session_wrapped: list[bool] = [False]  # track if wrap already happened
    _continuity_produced: list[bool] = [False]  # track if continuity was produced this session (F2 fix)
    _last_node_count: list[int] = [umem.memory.size]  # track node count to skip no-op wraps (F6 fix)
    _wrap_lock = threading.Lock()  # protects _session_wrapped check-then-act

    def _do_auto_wrap() -> None:
        """Execute auto-wrap. Called by timer thread or atexit.

        Uses _wrap_lock to prevent race between timer thread and main thread
        both calling session_wrap() simultaneously. The lock protects the
        check-then-act on _session_wrapped — without it, both threads could
        read False, set True, and proceed to concurrent session_wrap() calls
        that corrupt the audit hash chain.
        """
        with _wrap_lock:
            if _session_wrapped[0]:
                return
            _session_wrapped[0] = True
        # Lock released — flag prevents re-entry, and session_wrap() is now
        # safe to run without holding the lock (main thread won't enter).
        try:
            # Produce continuity file BEFORE pruning (capture full session data)
            # Skip if: already produced this session (F2), or no new nodes since last wrap (F6)
            if continuity_mgr and memory_path and not _continuity_produced[0]:
                current_nodes = umem.memory.size
                if current_nodes > _last_node_count[0]:
                    try:
                        meta = ContinuityManager.load_meta(memory_path)
                        existing = ContinuityManager.load(memory_path)
                        cont_result = continuity_mgr.produce(
                            umem.memory, existing_continuity=existing,
                            citations_seen=meta.get("citations_seen", False),
                        )
                        continuity_mgr.save(cont_result.text, memory_path)
                        meta["sessions_produced"] = meta.get("sessions_produced", 0) + 1
                        if cont_result.graduations_validated > 0:
                            meta["citations_seen"] = True
                        ContinuityManager.save_meta(meta, memory_path)
                        _continuity_produced[0] = True
                        _last_node_count[0] = current_nodes
                        _log(f"Auto-wrap: continuity produced ({cont_result.char_count} chars)")
                    except Exception as e:
                        _log(f"Auto-wrap: continuity production failed: {e}")
                else:
                    _log("Auto-wrap: skipping continuity (no new nodes since last wrap)")

            umem.memory.session_wrap()
            # session_wrap() calls session_end() which calls save() internally,
            # so no additional umem.save() needed here.
            _log("Auto-wrap: session consolidated after inactivity")
        except Exception as e:
            _log(f"Auto-wrap failed: {e}")

    def _reset_auto_wrap_timer() -> None:
        """Cancel pending timer and start a new one. Called on each tool call."""
        if auto_wrap_minutes <= 0:
            return
        # Cancel existing timer
        if _auto_wrap_timer[0] is not None:
            _auto_wrap_timer[0].cancel()
        # If a previous auto-wrap fired, restart session for the new activity
        with _wrap_lock:
            if _session_wrapped[0]:
                _session_wrapped[0] = False
                _continuity_produced[0] = False  # Reset for new session
                try:
                    umem.memory.session_start()
                except Exception:
                    pass
        # Schedule new timer
        timer = threading.Timer(auto_wrap_minutes * 60, _do_auto_wrap)
        timer.daemon = True
        timer.start()
        _auto_wrap_timer[0] = timer

    def _atexit_wrap() -> None:
        """Final wrap on process exit — save state + consolidate."""
        if _auto_wrap_timer[0] is not None:
            _auto_wrap_timer[0].cancel()
        if not _session_wrapped[0]:
            _do_auto_wrap()

    atexit.register(_atexit_wrap)

    # Start the initial timer
    if auto_wrap_minutes > 0:
        _reset_auto_wrap_timer()
        _log(f"Auto-wrap enabled: {auto_wrap_minutes}m inactivity threshold")

    try:
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue

            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                continue

            msg_id = msg.get("id")
            method = msg.get("method", "")
            params = msg.get("params", {})

            if method == "initialize":
                # Version negotiation: respond with client's version if we
                # support it, otherwise our latest. Our tools-only server is
                # compatible with all versions from 2024-11-05 onward.
                client_version = params.get("protocolVersion", _PROTOCOL_VERSION)
                resp = _jsonrpc_response(msg_id, {
                    "protocolVersion": client_version if client_version >= _PROTOCOL_VERSION else _PROTOCOL_VERSION,
                    "capabilities": {"tools": {}, "resources": {}},
                    "serverInfo": {
                        "name": _SERVER_NAME,
                        "version": _SERVER_VERSION,
                    },
                })
            elif method == "notifications/initialized":
                continue  # notification, no response
            elif method == "tools/list":
                resp = _jsonrpc_response(msg_id, {"tools": [json.loads(json.dumps(_thaw(t))) for t in ALL_TOOLS]})
            elif method == "resources/list":
                resp = _jsonrpc_response(msg_id, {"resources": [_thaw(_INTEGRITY_RESOURCE)]})
            elif method == "resources/read":
                uri = params.get("uri", "")
                if uri == "flowscript://integrity/manifest":
                    manifest = {
                        "version": _SERVER_VERSION,
                        "algorithm": "SHA-256",
                        "canonicalization": "deterministic sorted-keys JSON",
                        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                        "tool_count": _EXPECTED_TOOL_COUNT,
                        "tools": dict(_INTEGRITY_MANIFEST),
                        "build_time_manifest": "available" if _BUILD_TIME_MANIFEST else "not generated",
                        "usage": (
                            "Hash each tool definition (sorted keys, no whitespace, SHA-256) "
                            "and compare against the hashes in this manifest. Mismatches "
                            "indicate description mutation between server and client."
                        ),
                    }
                    resp = _jsonrpc_response(msg_id, {
                        "contents": [{
                            "uri": uri,
                            "mimeType": "application/json",
                            "text": json.dumps(manifest, indent=2),
                        }],
                    })
                else:
                    resp = _jsonrpc_error(msg_id, -32602, f"Unknown resource: {uri}")
            elif method == "prompts/list":
                resp = _jsonrpc_response(msg_id, {"prompts": []})
            elif method == "tools/call":
                tool_name = params.get("name", "")
                tool_args = params.get("arguments", {})
                _reset_auto_wrap_timer()  # Activity detected — reset consolidation timer
                result = handler.handle_tool(tool_name, tool_args)
                # If explicit session_wrap was called, mark it so auto-wrap doesn't double-fire
                if tool_name == "session_wrap":
                    with _wrap_lock:
                        _session_wrapped[0] = True
                    _continuity_produced[0] = True  # F2: prevent auto-wrap from overwriting
                # Save after modifications (session_wrap saves internally, but
                # add_memory/remove_memory need explicit save for vector index)
                if tool_name in ("add_memory", "remove_memory", "encode_exchange"):
                    try:
                        umem.save()
                    except ValueError:
                        pass  # in-memory mode, no file path — expected
                    except OSError as e:
                        _log(f"Warning: save failed: {e}")
                is_error = "error" in result
                call_result: dict[str, Any] = {
                    "content": [{"type": "text", "text": json.dumps(result, indent=2)}],
                }
                if is_error:
                    call_result["isError"] = True
                resp = _jsonrpc_response(msg_id, call_result)
            elif method == "ping":
                resp = _jsonrpc_response(msg_id, {})
            else:
                resp = _jsonrpc_error(msg_id, -32601, f"Method not found: {method}")

            sys.stdout.write(json.dumps(resp) + "\n")
            sys.stdout.flush()
    finally:
        # Clean shutdown: cancel timer, run final wrap (via atexit if not
        # already done), then clear adapter context.
        if _auto_wrap_timer[0] is not None:
            _auto_wrap_timer[0].cancel()
        if not _session_wrapped[0]:
            _do_auto_wrap()
        umem.memory.clear_adapter_context()


# =============================================================================
# CLI entry point
# =============================================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description="FlowScript Unified Memory MCP Server",
        epilog=(
            "Zero-config: if OPENAI_API_KEY is set and no --embedder is specified, "
            "the server auto-configures OpenAI embeddings, extraction, and consolidation. "
            "Just run: python -m flowscript_agents.mcp --memory ./agent.json"
        ),
    )
    parser.add_argument(
        "--memory",
        help="Path to memory JSON file (created if doesn't exist). Required unless --generate-manifest.",
    )
    parser.add_argument(
        "--embedder", choices=["openai", "sentence-transformers", "ollama"],
        help="Embedding provider (overrides auto-detection)",
    )
    parser.add_argument(
        "--embedding-model",
        help="Embedding model name (provider-specific, default: text-embedding-3-small)",
    )
    parser.add_argument(
        "--llm-model",
        default="gpt-4o-mini",
        help="LLM model for extraction and consolidation (default: gpt-4o-mini)",
    )
    parser.add_argument(
        "--no-auto",
        action="store_true",
        help="Disable auto-configuration from OPENAI_API_KEY",
    )
    parser.add_argument(
        "--generate-manifest",
        action="store_true",
        help="Generate tool-integrity.json and exit (build-time integrity manifest)",
    )
    args = parser.parse_args()

    # Generate build-time manifest and exit (no --memory needed)
    if args.generate_manifest:
        manifest = dict(_INTEGRITY_MANIFEST)
        out_path = os.path.join(os.path.dirname(__file__), "tool-integrity.json")
        with open(out_path, "w") as f:
            json.dump(manifest, f, indent=2, sort_keys=True)
            f.write("\n")
        print(f"Generated {out_path} ({len(manifest)} tools)")
        sys.exit(0)

    if not args.memory:
        parser.error("--memory is required (unless using --generate-manifest)")

    embedder = None
    llm = None
    consolidation = None

    if args.embedder:
        # Explicit embedder specified — use it, no auto-config
        kwargs = {}
        if args.embedding_model:
            if args.embedder == "openai":
                kwargs["model"] = args.embedding_model
            elif args.embedder == "sentence-transformers":
                kwargs["model_name"] = args.embedding_model
            elif args.embedder == "ollama":
                kwargs["model"] = args.embedding_model
        embedder = _create_embedder(args.embedder, **kwargs)
    elif not args.no_auto and os.environ.get("OPENAI_API_KEY"):
        # Auto-configure from OPENAI_API_KEY (full stack: embeddings + extraction + consolidation)
        try:
            embedder, llm, consolidation = _auto_configure_openai(
                model=args.llm_model,
                embedding_model=args.embedding_model,
            )
            _log("Auto-configured: OpenAI embeddings + extraction + consolidation "
                 f"(model: {args.llm_model})")
        except ImportError as e:
            _log(f"OpenAI auto-configuration skipped: {e}")
        except Exception as e:
            _log(f"OpenAI auto-configuration failed: {e}")
    elif not args.no_auto and os.environ.get("ANTHROPIC_API_KEY"):
        # Auto-configure from ANTHROPIC_API_KEY (extraction + consolidation, no embeddings)
        try:
            embedder, llm, consolidation = _auto_configure_anthropic(
                model=args.llm_model if args.llm_model != "gpt-4o-mini" else "claude-haiku-4-5-20251001",
            )
            _log("Auto-configured: Anthropic extraction + consolidation "
                 f"(no embeddings — Anthropic has no embedding API, using keyword search)")
        except ImportError as e:
            _log(f"Anthropic auto-configuration skipped: {e}")
        except Exception as e:
            _log(f"Anthropic auto-configuration failed: {e}")

    # Validate API key at startup with a lightweight probe
    if embedder is not None:
        try:
            embedder.embed(["startup validation"])
            _log("Embedding provider validated successfully")
        except Exception as e:
            _log(f"ERROR: Embedding provider failed validation: {e}")
            _log("Check your API key and network connection. "
                 "Falling back to keyword-only search.")
            embedder = None

    # Validate LLM at startup (lightweight probe — extraction, not full call)
    if llm is not None:
        try:
            test_result = llm("Respond with OK.")
            if test_result:
                _log("LLM extraction provider validated successfully")
        except Exception as e:
            _log(f"ERROR: LLM extraction provider failed validation: {e}")
            _log("Check your API key and network connection. "
                 "Falling back to raw text storage (no typed extraction).")
            llm = None
            consolidation = None

    # Warn about degraded mode so developers know what they're getting
    if embedder is None:
        _log("Warning: No embedding provider configured — vector search disabled. "
             "Set OPENAI_API_KEY for full auto-configuration, or use --embedder.")
    if llm is None:
        _log("Warning: No LLM configured — add_memory stores raw text only "
             "(no typed extraction). Set OPENAI_API_KEY or ANTHROPIC_API_KEY "
             "for auto-configuration.")

    run_server(
        memory_path=args.memory,
        embedder=embedder,
        llm=llm,
        consolidation_provider=consolidation,
    )


if __name__ == "__main__":
    main()
