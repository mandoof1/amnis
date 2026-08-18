"""Amnis command line interface.

The 0.1 CLI parsed ``sys.argv`` by hand and joined every remaining token into
the payload, so::

    amnis remember "I use fish" --importance 9

stored the literal text ``I use fish --importance 9`` at the default
importance. Flags are real flags now, every subcommand documents itself, and
``--json`` is available on all of them for scripting.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any


def _print(data: Any, as_json: bool = True) -> None:
    if as_json:
        print(json.dumps(data, indent=2, default=str))
    else:
        print(data)


def _csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


# ─── Commands ──────────────────────────────────────────────────────────


def cmd_server(args: argparse.Namespace) -> int:
    from .server.mcp import main as run_mcp

    run_mcp()
    return 0


def cmd_web(args: argparse.Namespace) -> int:
    from .config import config

    if args.host:
        config.host = args.host
    if args.port:
        config.port = args.port
    from .server.web import main as run_web

    run_web()
    return 0


def cmd_remember(args: argparse.Namespace) -> int:
    from .memory import store

    result = store.store(
        fact=args.fact,
        category=args.category,
        importance=args.importance,
        source=args.source,
        tags=args.tags,
        context=args.context,
        expiry=args.expiry,
    )
    _print(result, args.json)
    return 0


def cmd_recall(args: argparse.Namespace) -> int:
    from .memory import store

    results = store.recall(
        query=args.query or "",
        category=args.category,
        limit=args.limit,
        min_importance=args.min_importance,
        tags=args.tags,
        semantic=not args.no_semantic,
    )
    if args.json:
        _print(results)
    elif not results:
        print("No memories matched.")
    else:
        for m in results:
            print(f"[{m['category']:<10}] ({m['importance']:>2}/10) {m['fact']}")
            print(f"             {m['id']}  {(m['timestamp'] or '')[:19]}")
    return 0


def cmd_forget(args: argparse.Namespace) -> int:
    from .memory import store

    deleted = store.forget(args.memory_id)
    _print({"deleted": deleted, "id": args.memory_id}, args.json)
    return 0 if deleted else 1


def cmd_search(args: argparse.Namespace) -> int:
    from .rag.engine import RagError, get_engine

    engine = get_engine()
    try:
        if args.mode == "keyword":
            results = engine.keyword_search(args.query, limit=args.limit)
        elif args.mode == "semantic":
            results = engine.search(args.query, limit=args.limit)
        else:
            results = engine.hybrid_search(args.query, limit=args.limit)
    except RagError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if args.json:
        _print(results)
    elif not results:
        print("No results.")
    else:
        for r in results:
            score = r.get("hybrid_score", r.get("score", r.get("rank", "")))
            print(f"{r.get('search_type', '?'):<9} {score}  {r.get('source', '')}")
            print(f"  {r.get('content', '')[:200]}")
    return 0


def cmd_index(args: argparse.Namespace) -> int:
    from .rag.engine import RagError, get_engine

    try:
        result = get_engine().index_file(args.path, origin=args.origin)
    except RagError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    _print(result, args.json)
    return 0


def cmd_index_notes(args: argparse.Namespace) -> int:
    from .rag.engine import get_engine

    _print(get_engine().index_notes(), args.json)
    return 0


def cmd_index_wiki(args: argparse.Namespace) -> int:
    from .rag.engine import get_engine

    _print(get_engine().index_wiki(), args.json)
    return 0


def cmd_reindex(args: argparse.Namespace) -> int:
    from .memory import store

    _print(store.reindex_keywords(), args.json)
    return 0


def cmd_compile_wiki(args: argparse.Namespace) -> int:
    from .wiki.compiler import get_compiler

    _print(get_compiler().compile(topics=args.topics or None), args.json)
    return 0


def cmd_wiki_query(args: argparse.Namespace) -> int:
    from .wiki.compiler import get_compiler

    _print(get_compiler().query(args.question), args.json)
    return 0


def cmd_wiki_lint(args: argparse.Namespace) -> int:
    from .wiki.compiler import get_compiler

    result = get_compiler().lint()
    _print(result, args.json)
    return 1 if result["issues_found"] else 0


def cmd_consolidate(args: argparse.Namespace) -> int:
    from .memory import consolidation

    _print(consolidation.run_pipeline(), args.json)
    return 0


def cmd_prune(args: argparse.Namespace) -> int:
    from .memory import pruning

    _print(pruning.run_pipeline(dry_run=args.dry_run), args.json)
    return 0


def cmd_episodic_log(args: argparse.Namespace) -> int:
    from .memory import episodic

    _print(
        episodic.log_episode(
            session_id=args.session_id,
            role=args.role,
            content=args.content,
            summary=args.summary,
            topics=args.topics,
            outcome=args.outcome,
        ),
        args.json,
    )
    return 0


def cmd_episodic_recall(args: argparse.Namespace) -> int:
    from .memory import episodic

    _print(
        episodic.recall_episodes(
            session_id=args.session_id,
            topic=args.topic,
            role=args.role,
            limit=args.limit,
        ),
        args.json,
    )
    return 0


def cmd_episodic_prune(args: argparse.Namespace) -> int:
    from .memory import episodic

    _print({"pruned": episodic.prune_old_episodes(days=args.days), "retention_days": args.days}, args.json)
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    from .config import config, unknown_env_vars
    from .memory import episodic
    from .memory import store as memory_store

    report: dict[str, Any] = {"errors": {}}
    for name, fn in (("memory", memory_store.stats), ("episodic", episodic.stats)):
        try:
            report[name] = fn()
        except Exception as exc:  # noqa: BLE001 - degraded reporting
            report[name] = {}
            report["errors"][name] = str(exc)
    try:
        from .rag.engine import get_engine

        report["rag"] = get_engine().stats()
    except Exception as exc:  # noqa: BLE001
        report["rag"] = {}
        report["errors"]["rag"] = str(exc)
    try:
        from .wiki.compiler import get_compiler

        report["wiki"] = get_compiler().stats()
    except Exception as exc:  # noqa: BLE001
        report["wiki"] = {}
        report["errors"]["wiki"] = str(exc)

    report["status"] = "degraded" if report["errors"] else "ok"
    report["config"] = {
        "data_dir": str(config.data_dir),
        "notes_dir": str(config.notes_dir),
        "wiki_dir": str(config.wiki_dir),
        "memory_db": str(config.memory_db),
        "embedding_model": config.embedding_model,
    }
    unknown = unknown_env_vars()
    if unknown:
        report["config"]["ignored_env_vars"] = unknown

    if args.json:
        _print(report)
    else:
        print(f"Amnis: {report['status']}")
        print(f"  Memories  {report['memory'].get('total_memories', 0)}")
        print(
            f"  Chunks    {report['rag'].get('total_chunks', 0)} "
            f"from {report['rag'].get('unique_sources', 0)} sources"
        )
        print(f"  Wiki      {report['wiki'].get('total_pages', 0)} pages")
        print(f"  Episodes  {report['episodic'].get('total_episodes', 0)}")
        print(f"  Data dir  {config.data_dir}")
        for layer, message in report["errors"].items():
            print(f"  ! {layer}: {message}", file=sys.stderr)
        for var in unknown:
            print(f"  ! ignoring unknown env var {var}", file=sys.stderr)
    return 1 if report["errors"] else 0


def cmd_init(args: argparse.Namespace) -> int:
    from .config import config
    from .rag.engine import get_engine
    from .wiki.compiler import get_compiler

    for directory in (
        config.data_dir,
        config.notes_dir,
        config.wiki_dir,
        config.wiki_facts_dir,
        config.chroma_dir,
    ):
        directory.mkdir(parents=True, exist_ok=True)
    print("Directories ready:")
    print(f"  data  {config.data_dir}")
    print(f"  notes {config.notes_dir}")
    print(f"  wiki  {config.wiki_dir}")

    print("\nIndexing notes...")
    result = get_engine().index_notes()
    print(f"  {result.get('files_indexed', 0)} files, {result.get('chunks_indexed', 0)} chunks")

    if not args.skip_wiki:
        print("\nCompiling wiki...")
        wiki_result = get_compiler().compile()
        print(f"  {wiki_result.get('compiled', 0)} pages")

    print("\nAmnis is ready. Try: amnis status")
    return 0


# ─── Parser ────────────────────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="amnis",
        description="Persistent memory, RAG, and wiki compilation for AI agents.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Run `amnis <command> --help` for per-command options.",
    )
    from . import __version__

    parser.add_argument("--version", action="version", version=f"amnis {__version__}")
    parser.add_argument("--json", action="store_true", help="always emit JSON")
    sub = parser.add_subparsers(dest="command", metavar="<command>")

    def add(name: str, func, help_text: str, **kwargs):
        p = sub.add_parser(name, help=help_text, description=help_text, **kwargs)
        p.set_defaults(func=func)
        return p

    add("server", cmd_server, "Run the MCP server over stdio (default).")

    p = add("web", cmd_web, "Run the web dashboard.")
    p.add_argument("--host", help="bind address (default: from config)")
    p.add_argument("--port", type=int, help="port (default: from config)")

    p = add("remember", cmd_remember, "Store a fact.")
    p.add_argument("fact")
    p.add_argument(
        "-c",
        "--category",
        default="general",
        choices=["preference", "fact", "event", "procedure", "concept", "theme", "meta", "general"],
    )
    p.add_argument("-i", "--importance", type=int, default=5, choices=range(1, 11), metavar="1-10")
    p.add_argument("-t", "--tags", type=_csv, default=None, help="comma-separated")
    p.add_argument("--context")
    p.add_argument("--source", default="cli")
    p.add_argument("--expiry", help="ISO-8601 timestamp after which the fact expires")

    p = add("recall", cmd_recall, "Recall memories by meaning and keyword.")
    p.add_argument("query", nargs="?", default="")
    p.add_argument("-c", "--category")
    p.add_argument("-n", "--limit", type=int, default=10)
    p.add_argument("-m", "--min-importance", type=int, default=0)
    p.add_argument("-t", "--tags", type=_csv, default=None)
    p.add_argument("--no-semantic", action="store_true", help="keyword search only")

    p = add("forget", cmd_forget, "Delete a memory by ID.")
    p.add_argument("memory_id")

    p = add("search", cmd_search, "Search indexed documents.")
    p.add_argument("query")
    p.add_argument("-n", "--limit", type=int, default=5)
    p.add_argument("--mode", choices=["hybrid", "semantic", "keyword"], default="hybrid")

    p = add("index", cmd_index, "Index a single file.")
    p.add_argument("path")
    p.add_argument("--origin", default="note", choices=["note", "wiki", "memory", "compiled"])

    add("index-notes", cmd_index_notes, "Index every note in the notes directory.")
    add("index-wiki", cmd_index_wiki, "Index the wiki directory.")
    add("reindex", cmd_reindex, "Rebuild the memory keyword index (run once after upgrading from 0.1).")

    p = add("compile-wiki", cmd_compile_wiki, "Compile wiki pages.")
    p.add_argument("topics", nargs="*", help="specific topics (default: all)")

    p = add("wiki-query", cmd_wiki_query, "Ask the wiki a question.")
    p.add_argument("question")

    add("wiki-lint", cmd_wiki_lint, "Report stale, sourceless, or duplicate wiki pages.")
    add("consolidate", cmd_consolidate, "Extract facts from recent logs, then reflect.")

    p = add("prune", cmd_prune, "Decay, deduplicate, and prune the memory store.")
    p.add_argument("--dry-run", action="store_true", help="report without deleting")

    p = add("episodic-log", cmd_episodic_log, "Log a conversation turn.")
    p.add_argument("session_id")
    p.add_argument("role", choices=["user", "assistant"])
    p.add_argument("content")
    p.add_argument("--summary")
    p.add_argument("--topics", type=_csv, default=None)
    p.add_argument("--outcome", choices=["success", "failure", "neutral"])

    p = add("episodic-recall", cmd_episodic_recall, "Recall logged turns.")
    p.add_argument("--session-id")
    p.add_argument("--topic")
    p.add_argument("--role", choices=["user", "assistant"])
    p.add_argument("-n", "--limit", type=int, default=20)

    p = add("episodic-prune", cmd_episodic_prune, "Delete episodes past the retention window.")
    p.add_argument("--days", type=int, default=30)

    add("status", cmd_status, "Show the health and size of every layer.")

    p = add("init", cmd_init, "Create directories, index notes, compile the wiki.")
    p.add_argument("--skip-wiki", action="store_true")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if not getattr(args, "command", None):
        # No subcommand: run the MCP server, matching how agent hosts spawn it.
        return cmd_server(args)

    try:
        return args.func(args) or 0
    except KeyboardInterrupt:
        return 130
    except (ValueError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
