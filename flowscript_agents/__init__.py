"""
FlowScript Agents — Complete agent memory: reasoning + vector + auto-extraction.

Decision intelligence memory for AI agent frameworks. Drop-in provider for
LangGraph, CrewAI, Google ADK, OpenAI Agents SDK, Pydantic AI, smolagents,
LlamaIndex, Haystack, and CAMEL-AI.

What Mem0 does, plus what Mem0 can't: typed semantic queries (why, tensions,
blocked, alternatives, whatIf) over agent reasoning with temporal intelligence.

Usage:
    from flowscript_agents import Memory, UnifiedMemory
    from flowscript_agents.embeddings import OpenAIEmbeddings

    # Reasoning memory (no embeddings needed)
    mem = Memory()
    q = mem.question("Which database?")
    mem.alternative(q, "Redis").decide(rationale="speed critical")
    mem.alternative(q, "SQLite").block(reason="no concurrent writes")
    print(mem.query.tensions())

    # Complete memory (reasoning + vector + auto-extraction)
    umem = UnifiedMemory("./agent.json", embedder=OpenAIEmbeddings(), llm=my_llm)
    umem.add("User chose PostgreSQL for ACID compliance")
    results = umem.search("database decisions")
    umem.memory.query.why(results[0].node_id)
"""

from .audit import AuditConfig, AuditQueryResult, AuditVerifyResult
from .continuity import ContinuityManager, ContinuityResult
from .memory import (
    Memory,
    MemoryOptions,
    NodeRef,
    TemporalConfig,
    TemporalMeta,
    TemporalTierConfig,
    DormancyConfig,
    GardenReport,
    PruneReport,
    SessionStartResult,
    SessionEndResult,
    SessionWrapResult,
)
from .unified import UnifiedMemory

__version__ = "0.4.1"
__all__ = [
    "AuditConfig",
    "AuditQueryResult",
    "AuditVerifyResult",
    "ContinuityManager",
    "ContinuityResult",
    "Memory",
    "MemoryOptions",
    "NodeRef",
    "TemporalConfig",
    "TemporalMeta",
    "TemporalTierConfig",
    "DormancyConfig",
    "GardenReport",
    "PruneReport",
    "SessionStartResult",
    "SessionEndResult",
    "SessionWrapResult",
    "UnifiedMemory",
]
