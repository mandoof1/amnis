"""MCP Server for Amnis — exposes memory, RAG, and wiki tools to Hermes Agent.

Run with:
  python -m amnis.server.mcp

Or via Hermes config.yaml:
  mcp_servers:
    amnis:
      command: /path/to/amnis/env/bin/python
      args: ["-m", "amnis.server.mcp"]
"""
import sys
import json
from pathlib import Path
from typing import Optional

# Ensure amnis is importable
sys.path.insert(0, str(Path.home() / "amnis"))

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import (
    Tool,
    TextContent,
    CallToolResult,
)

from ..memory import store as memory_store
from ..memory import consolidation as memory_consolidation
from ..memory import episodic as memory_episodic
from ..memory import pruning as memory_pruning
from ..rag.engine import engine as rag_engine
from ..wiki.compiler import compiler as wiki_compiler
from ..config import config

server = Server("amnis")


# ─── Tool Definitions ──────────────────────────────────────────────────

@server.list_tools()
async def list_tools() -> list[Tool]:
    return [
        # ── Memory Tools ──
        Tool(
            name="amnis_remember",
            description="Store a persistent fact so I can remember it across sessions. Use this for anything LO tells me about himself, his preferences, his setup, or things I should never forget.",
            inputSchema={
                "type": "object",
                "properties": {
                    "fact": {
                        "type": "string",
                        "description": "The fact/memory to store (e.g., 'LO prefers dark mode in all his editors')",
                    },
                    "category": {
                        "type": "string",
                        "enum": ["preference", "fact", "event", "procedure", "concept", "meta", "general"],
                        "description": "Category of memory",
                        "default": "general",
                    },
                    "importance": {
                        "type": "integer",
                        "description": "Importance 1-10 (10 = critical, never forget)",
                        "default": 5,
                        "minimum": 1,
                        "maximum": 10,
                    },
                    "tags": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Tags for filtering",
                    },
                    "context": {
                        "type": "string",
                        "description": "Surrounding context for better recall",
                    },
                },
                "required": ["fact"],
            },
        ),
        Tool(
            name="amnis_recall",
            description="Recall stored memories. Use this when you need to remember something LO told you in a past session, or check if you already know something.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "What to search for",
                    },
                    "category": {
                        "type": "string",
                        "description": "Filter by category",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max results (default: 10)",
                        "default": 10,
                    },
                    "min_importance": {
                        "type": "integer",
                        "description": "Minimum importance filter 1-10",
                        "default": 0,
                    },
                    "tags": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Filter by tags",
                    },
                },
            },
        ),
        Tool(
            name="amnis_forget",
            description="Delete a specific memory by its ID.",
            inputSchema={
                "type": "object",
                "properties": {
                    "memory_id": {
                        "type": "string",
                        "description": "ID of the memory to delete",
                    },
                },
                "required": ["memory_id"],
            },
        ),
        Tool(
            name="amnis_memory_stats",
            description="Get statistics about the memory store — how many memories, by category, etc.",
            inputSchema={"type": "object", "properties": {}},
        ),
        Tool(
            name="amnis_consolidate",
            description="Run memory consolidation — extracts structured facts from recent conversation logs.",
            inputSchema={
                "type": "object",
                "properties": {
                    "limit": {
                        "type": "integer",
                        "description": "Number of recent logs to scan",
                        "default": 50,
                    },
                },
            },
        ),

        # ── RAG Tools ──
        Tool(
            name="amnis_search",
            description="Semantic search over indexed documents (notes, code, markdown).",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "What to search for",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max results (default: 5)",
                        "default": 5,
                    },
                    "where": {
                        "type": "object",
                        "description": "Metadata filter (e.g., {'source': {'$contains': 'memory'}})",
                    },
                },
                "required": ["query"],
            },
        ),
        Tool(
            name="amnis_index_file",
            description="Index a file into the RAG vector store so it becomes searchable.",
            inputSchema={
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Path to the file to index",
                    },
                },
                "required": ["file_path"],
            },
        ),
        Tool(
            name="amnis_index_notes",
            description="Index (or re-index) all notes in the Amnis notes directory for semantic search.",
            inputSchema={"type": "object", "properties": {}},
        ),
        Tool(
            name="amnis_rag_stats",
            description="Get RAG engine statistics — indexed chunks, sources, embedding model, keyword index.",
            inputSchema={"type": "object", "properties": {}},
        ),
        Tool(
            name="amnis_hybrid_search",
            description="Hybrid search combining semantic (ChromaDB) + keyword (FTS5) with weighted ranking. Best for finding documents when you need both meaning and exact keyword matches.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "limit": {"type": "integer", "description": "Max results", "default": 5},
                    "semantic_weight": {"type": "number", "description": "Weight for semantic search (0.0-1.0). Lower = more keyword-heavy.", "default": 0.7},
                    "where": {
                        "type": "object",
                        "description": "Metadata filter (e.g., {'source': {'$contains': 'memory'}})",
                    },
                },
                "required": ["query"],
            },
        ),

        # ── Wiki Tools ──
        Tool(
            name="amnis_compile_wiki",
            description="Compile the knowledge wiki — reads indexed notes + memory, generates structured markdown wiki pages with cross-references.",
            inputSchema={
                "type": "object",
                "properties": {
                    "topics": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Specific topics to compile (omit to compile all)",
                    },
                },
            },
        ),
        Tool(
            name="amnis_wiki_query",
            description="Ask the compiled wiki a question — searches across all wiki pages and RAG index.",
            inputSchema={
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "Your question",
                    },
                },
                "required": ["question"],
            },
        ),
        Tool(
            name="amnis_wiki_lint",
            description="Check the wiki for stale content, missing sources, and other issues.",
            inputSchema={"type": "object", "properties": {}},
        ),
        Tool(
            name="amnis_wiki_stats",
            description="Get wiki statistics — page count, source count, directory.",
            inputSchema={"type": "object", "properties": {}},
        ),

        # ── Episodic Memory Tools ──
        Tool(
            name="amnis_episodic_log",
            description="Log a conversation turn as an episodic memory. Stores the interaction with extracted topics and a summary for later recall.",
            inputSchema={
                "type": "object",
                "properties": {
                    "session_id": {"type": "string", "description": "Session identifier"},
                    "role": {"type": "string", "enum": ["user", "assistant"], "description": "Who spoke"},
                    "content": {"type": "string", "description": "The interaction content"},
                    "summary": {"type": "string", "description": "Optional summary (auto-generated if not provided)"},
                    "topics": {"type": "array", "items": {"type": "string"}, "description": "Optional topic tags (auto-extracted if not provided)"},
                    "outcome": {"type": "string", "enum": ["success", "failure", "neutral"], "description": "Optional outcome of the interaction"},
                    "results": {"type": "object", "description": "Optional structured outcome data"},
                },
                "required": ["session_id", "role", "content"],
            },
        ),
        Tool(
            name="amnis_episodic_recall",
            description="Recall episodic memories with filters — by session, topic, or role.",
            inputSchema={
                "type": "object",
                "properties": {
                    "session_id": {"type": "string", "description": "Filter by session ID"},
                    "topic": {"type": "string", "description": "Filter by topic keyword"},
                    "role": {"type": "string", "enum": ["user", "assistant"], "description": "Filter by role"},
                    "limit": {"type": "integer", "description": "Max results", "default": 20},
                },
            },
        ),
        Tool(
            name="amnis_episodic_stats",
            description="Get episodic memory statistics — total episodes, unique sessions, date range.",
            inputSchema={"type": "object", "properties": {}},
        ),
        Tool(
            name="amnis_episodic_prune",
            description="Delete episodic memories older than the retention period (default: 30 days). Frees up space without affecting permanent semantic memory.",
            inputSchema={
                "type": "object",
                "properties": {
                    "days": {"type": "integer", "description": "Retention period in days", "default": 30},
                },
            },
        ),

        # ── Pruning Tools ──
        Tool(
            name="amnis_prune",
            description="Run automatic memory pruning — removes expired, low-importance/stale, and duplicate memories. Pass dry_run=true to preview without deleting anything.",
            inputSchema={
                "type": "object",
                "properties": {
                    "dry_run": {"type": "boolean", "description": "If true, only report what would be pruned without deleting", "default": false},
                },
            },
        ),

        # ── Meta Tools ──
        Tool(
            name="amnis_status",
            description="Get the overall status of Amnis — all three layers' health and stats.",
            inputSchema={"type": "object", "properties": {}},
        ),
    ]


# ─── Tool Handlers ─────────────────────────────────────────────────────

@server.call_tool()
async def call_tool(name: str, arguments: dict) -> CallToolResult:
    try:
        if name == "amnis_remember":
            result = memory_store.store(
                fact=arguments["fact"],
                category=arguments.get("category", "general"),
                importance=arguments.get("importance", 5),
                source="agent",
                tags=arguments.get("tags"),
                context=arguments.get("context"),
            )
            return _ok(f"✅ Remembered: {result['fact'][:80]}...", result)

        elif name == "amnis_recall":
            results = memory_store.recall(
                query=arguments.get("query", ""),
                category=arguments.get("category"),
                limit=arguments.get("limit", 10),
                min_importance=arguments.get("min_importance", 0),
                tags=arguments.get("tags"),
            )
            if not results:
                return _ok("🤔 No memories found matching that query.")
            summary = f"📋 Found {len(results)} memory(ies):\n"
            for m in results:
                summary += f"\n  [{m['category']}] (⭐{m['importance']}) {m['fact'][:120]}"
                summary += f"\n  ID: {m['id'][:8]}... | {m['timestamp'][:10]}"
            return _ok(summary, results)

        elif name == "amnis_forget":
            success = memory_store.forget(arguments["memory_id"])
            if success:
                return _ok(f"🗑️ Deleted memory {arguments['memory_id'][:8]}...")
            return _ok(f"⚠️ Memory {arguments['memory_id'][:8]}... not found.")

        elif name == "amnis_memory_stats":
            stats = memory_store.stats()
            return _ok(f"📊 Memory Stats: {stats['total_memories']} total, avg importance {stats['avg_importance']}", stats)

        elif name == "amnis_consolidate":
            result = memory_consolidation.run_pipeline()
            return _ok(f"🔄 Consolidation: scanned {result['logs_scanned']} logs, extracted {result['memories_extracted']} memories", result)

        elif name == "amnis_search":
            results = rag_engine.search(
                query=arguments["query"],
                limit=arguments.get("limit", 5),
                where=arguments.get("where"),
            )
            if not results:
                return _ok("📄 No RAG results found.")
            summary = f"🔍 RAG Search: {len(results)} results for '{arguments['query']}':\n"
            for r in results:
                summary += f"\n  [{r['file_type']}] {r['source']} (score: {r['score']:.2f})"
                summary += f"\n  > {r['content'][:150]}"
            return _ok(summary, results)

        elif name == "amnis_index_file":
            result = rag_engine.index_file(arguments["file_path"])
            if "error" in result:
                return _ok(f"⚠️ {result['error']}")
            return _ok(f"📥 Indexed {result['indexed']} chunks from {arguments['file_path']}", result)

        elif name in ("amnis_index_notes", "amnis_index_vault"):
            result = rag_engine.index_notes()
            return _ok(f"📚 Notes indexed: {result['files_indexed']} files, {result['chunks_indexed']} chunks", result)

        elif name == "amnis_rag_stats":
            stats = rag_engine.stats()
            return _ok(f"📊 RAG Stats: {stats.get('total_chunks', 0)} chunks from {stats.get('unique_sources', 0)} sources. Keyword index: {stats.get('keyword_index', {}).get('total_entries', 0)} entries", stats)

        elif name == "amnis_hybrid_search":
            results = rag_engine.hybrid_search(
                query=arguments["query"],
                limit=arguments.get("limit", 5),
                semantic_weight=arguments.get("semantic_weight"),
                where=arguments.get("where"),
            )
            if not results:
                return _ok("📄 No hybrid search results found.")
            summary = f"🔀 Hybrid Search: {len(results)} results for '{arguments['query']}'\\n"
            for r in results:
                label = f"[{r.get('search_type', '?')}]"
                heading = f" › {r['heading']}" if r.get('heading') else ""
                summary += f"\\n  {label} {r['source']}{heading} (score: {r.get('hybrid_score', r['score']):.2f})"
                summary += f"\\n  > {r['content'][:150]}"
            return _ok(summary, results)

        elif name == "amnis_compile_wiki":
            result = wiki_compiler.compile(topics=arguments.get("topics"))
            return _ok(f"📝 Wiki compiled: {result['compiled']} pages", result)

        elif name == "amnis_wiki_query":
            result = wiki_compiler.query(arguments["question"])
            summary = f"❓ Wiki Query: '{arguments['question']}'\n"
            if result.get("wiki_pages"):
                summary += f"  Wiki matches: {len(result['wiki_pages'])}\n"
                for w in result["wiki_pages"]:
                    summary += f"  - {w['title']}\n"
            if result.get("rag_results"):
                summary += f"  RAG results: {len(result['rag_results'])}"
            return _ok(summary, result)

        elif name == "amnis_wiki_lint":
            result = wiki_compiler.lint()
            return _ok(f"🔍 Wiki lint: {result['pages_checked']} pages, {result['issues_found']} issues", result)

        elif name == "amnis_wiki_stats":
            stats = wiki_compiler.stats()
            return _ok(f"📊 Wiki Stats: {stats['total_pages']} pages", stats)

        # ── Episodic Handlers ──
        elif name == "amnis_episodic_log":
            result = memory_episodic.log_episode(
                session_id=arguments["session_id"],
                role=arguments["role"],
                content=arguments["content"],
                summary=arguments.get("summary"),
                topics=arguments.get("topics"),
                outcome=arguments.get("outcome"),
                results=arguments.get("results"),
            )
            return _ok(f"📝 Episode logged (topics: {result.get('topics', [])})", result)

        elif name == "amnis_episodic_recall":
            results = memory_episodic.recall_episodes(
                session_id=arguments.get("session_id"),
                topic=arguments.get("topic"),
                role=arguments.get("role"),
                limit=arguments.get("limit", 20),
            )
            if not results:
                return _ok("📝 No episodes found.")
            summary = f"📝 Found {len(results)} episode(s):\\n"
            for e in results:
                topics = ", ".join(e.get("topics", [])[:3])
                summary += f"\\n  [{e['role']}] {e['summary'][:100]}"
                summary += f"\\n  Topics: {topics} | {e['timestamp'][:10]}"
            return _ok(summary, results)

        elif name == "amnis_episodic_stats":
            stats = memory_episodic.stats()
            return _ok(f"📊 Episodic Stats: {stats['total_episodes']} episodes across {stats['unique_sessions']} sessions", stats)

        elif name == "amnis_episodic_prune":
            count = memory_episodic.prune_old_episodes(days=arguments.get("days", 30))
            return _ok(f"🧹 Pruned {count} old episodes")

        # ── Pruning Handlers ──
        elif name == "amnis_prune":
            result = memory_pruning.run_pipeline(dry_run=arguments.get("dry_run", False))
            if result.get("dry_run"):
                return _ok(
                    f"🔍 Dry Run: would remove {result['total_removed']} memories "
                    f"({result['expired_removed']} expired, "
                    f"{result['duplicate_candidates']} duplicates, "
                    f"{result['low_importance_candidates']} low-importance stale)",
                    result
                )
            return _ok(
                f"🧹 Pruned {result['total_removed']} memories "
                f"({result['expired_removed']} expired, "
                f"{result['duplicates_merged']} merged duplicates, "
                f"{result['low_importance_pruned']} low-importance stale)",
                result
            )

        # ── Meta ──
        elif name == "amnis_status":
            mem_stats = memory_store.stats()
            rag_stats = rag_engine.stats()
            wiki_stats = wiki_compiler.stats()

            status = "ALL SYSTEMS OPERATIONAL" if mem_stats.get("total_memories", 0) >= 0 else "DEGRADED"
            report = (
                f"🟢 **Amnis Status: {status}**\n\n"
                f"**Memory Store:** {mem_stats.get('total_memories', 0)} memories, "
                f"avg importance {mem_stats.get('avg_importance', '-')}\n"
                f"**RAG Engine:** {rag_stats.get('total_chunks', 0)} chunks, "
                f"{rag_stats.get('unique_sources', 0)} sources\n"
                f"**Wiki:** {wiki_stats.get('total_pages', 0)} pages\n"
                f"**Notes Dir:** {config.notes_dir}\n"
                f"**Embedding Model:** {config.embedding_model}"
            )
            return _ok(report, {
                "status": status,
                "memory": mem_stats,
                "rag": rag_stats,
                "wiki": wiki_stats,
                "config": {
                    "notes_dir": str(config.notes_dir),
                    "data_dir": str(config.data_dir),
                    "embedding_model": config.embedding_model,
                },
            })

        else:
            return _ok(f"Unknown tool: {name}")

    except Exception as e:
        return CallToolResult(
            isError=True,
            content=[TextContent(type="text", text=f"❌ Error: {str(e)}")],
        )


def _ok(text: str, data: dict | None = None) -> CallToolResult:
    """Return a successful tool result with optional JSON data."""
    content = [TextContent(type="text", text=text)]
    if data:
        content.append(TextContent(
            type="text",
            text=f"\n---\n```json\n{json.dumps(data, indent=2, default=str)}\n```",
        ))
    return CallToolResult(content=content)


async def main():
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
