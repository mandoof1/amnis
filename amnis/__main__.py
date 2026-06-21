"""Amnis CLI — run the MCP server or execute one-off commands.

Usage:
  python -m amnis                     # Run MCP server (default)
  python -m amnis server              # Same — run MCP server
  python -m amnis remember <fact>     # Store a memory
  python -m amnis recall <query>      # Recall memories
  python -m amnis search <query>      # RAG semantic search
  python -m amnis hybrid-search <query>  # Hybrid semantic + keyword search
  python -m amnis index <path>        # Index a file
  python -m amnis index-notes         # Index all notes in notes directory
  python -m amnis index-wiki           # Index all wiki files into ChromaDB (vault sync)
  python -m amnis compile-wiki        # Compile wiki
  python -m amnis status              # Show all stats
  python -m amnis init                # Initialize (create dirs, index notes)
  python -m amnis episodic-log        # Log an episode (session role content)
  python -m amnis episodic-recall     # Recall episodes
  python -m amnis episodic-prune      # Prune old episodes
  python -m amnis prune               # Run memory pruning (--dry-run to preview)
  python -m amnis rag-stats           # RAG engine + keyword index stats
"""
import sys
import json
from pathlib import Path

# Ensure amnis is importable when run from anywhere
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def print_json(data):
    print(json.dumps(data, indent=2, default=str))


def main():
    args = sys.argv[1:]

    if not args or args[0] == "server":
        from .server.mcp import main as run_server
        import asyncio
        asyncio.run(run_server())

    elif args[0] == "remember":
        from .memory import store
        fact = " ".join(args[1:])
        if not fact:
            print("Usage: python -m amnis remember <fact> [--category X] [--importance N]")
            return
        result = store.store(fact=fact)
        print_json(result)

    elif args[0] == "recall":
        from .memory import store
        query = " ".join(args[1:]) if len(args) > 1 else ""
        results = store.recall(query=query)
        print_json(results)

    elif args[0] == "search":
        from .rag.engine import engine
        query = " ".join(args[1:])
        if not query:
            print("Usage: python -m amnis search <query>")
            return
        results = engine.search(query=query)
        print_json(results)

    elif args[0] in ("hybrid-search", "hybrid_search"):
        from .rag.engine import engine
        query = " ".join(args[1:])
        if not query:
            print("Usage: python -m amnis hybrid-search <query> [--weight 0.7]")
            return
        weight = 0.7
        if "--weight" in args:
            try:
                idx = args.index("--weight")
                weight = float(args[idx + 1])
            except (ValueError, IndexError):
                pass
        results = engine.hybrid_search(query=query, semantic_weight=weight)
        print_json(results)

    elif args[0] == "index":
        if len(args) < 2:
            print("Usage: python -m amnis index <file_path>")
            return
        from .rag.engine import engine
        result = engine.index_file(args[1])
        print_json(result)

    elif args[0] == "index-notes":
        from .rag.engine import engine
        result = engine.index_notes()
        print_json(result)

    elif args[0] in ("index-vault",):
        # Legacy alias — redirect to index-notes
        from .rag.engine import engine
        result = engine.index_notes()
        print_json(result)

    elif args[0] == "compile-wiki":
        from .wiki.compiler import compiler
        topics = args[1:] if len(args) > 1 else None
        result = compiler.compile(topics=topics)
        print_json(result)

    elif args[0] in ("index-wiki", "index_wiki"):
        from .rag.engine import engine
        result = engine.index_wiki()
        print_json(result)

    elif args[0] == "status":
        from .memory import store as memory_store
        from .rag.engine import engine as rag_engine
        from .wiki.compiler import compiler as wiki_compiler
        from .memory import episodic as memory_episodic
        from .config import config
        print(json.dumps({
            "memory": memory_store.stats(),
            "rag": rag_engine.stats(),
            "wiki": wiki_compiler.stats(),
            "episodic": memory_episodic.stats(),
            "config": {
                "notes_dir": str(config.notes_dir),
                "data_dir": str(config.data_dir),
                "embedding_model": config.embedding_model,
            },
        }, indent=2))

    elif args[0] == "rag-stats":
        from .rag.engine import engine
        print_json(engine.stats())

    elif args[0] == "episodic-log":
        from .memory import episodic as memory_episodic
        if len(args) < 4:
            print("Usage: python -m amnis episodic-log <session_id> <role> <content> [--summary X] [--topics A,B,C]")
            return
        session_id = args[1]
        role = args[2]
        content = " ".join(args[3:])
        # Extract --summary and --topics from content if present
        summary = None
        topics = None
        if "--summary" in args:
            idx = args.index("--summary")
            summary_parts = []
            for s in args[idx + 1:]:
                if s.startswith("--"):
                    break
                summary_parts.append(s)
            summary = " ".join(summary_parts)
        if "--topics" in args:
            idx = args.index("--topics")
            try:
                topics_str = args[idx + 1]
                topics = [t.strip() for t in topics_str.split(",") if t.strip()]
            except IndexError:
                pass
        result = memory_episodic.log_episode(
            session_id=session_id, role=role, content=content,
            summary=summary, topics=topics,
        )
        print_json(result)

    elif args[0] == "episodic-recall":
        from .memory import episodic as memory_episodic
        session_id = None
        topic = None
        role = None
        limit = 20
        # Parse flags
        if "--session" in args:
            idx = args.index("--session")
            if idx + 1 < len(args):
                session_id = args[idx + 1]
        if "--topic" in args:
            idx = args.index("--topic")
            if idx + 1 < len(args):
                topic = args[idx + 1]
        if "--role" in args:
            idx = args.index("--role")
            if idx + 1 < len(args):
                role = args[idx + 1]
        if "--limit" in args:
            idx = args.index("--limit")
            if idx + 1 < len(args):
                try:
                    limit = int(args[idx + 1])
                except ValueError:
                    pass
        results = memory_episodic.recall_episodes(
            session_id=session_id, topic=topic, role=role, limit=limit,
        )
        print_json(results)

    elif args[0] == "episodic-prune":
        from .memory import episodic as memory_episodic
        days = 30
        if len(args) > 1:
            try:
                days = int(args[1])
            except ValueError:
                pass
        count = memory_episodic.prune_old_episodes(days=days)
        print_json({"pruned": count, "retention_days": days})

    elif args[0] == "prune":
        from .memory import pruning as memory_pruning
        dry_run = "--dry-run" in args
        result = memory_pruning.run_pipeline(dry_run=dry_run)
        print_json(result)

    elif args[0] == "init":
        from .memory import store
        from .rag.engine import engine
        from .wiki.compiler import compiler
        from .config import config

        # Create directories
        config.data_dir.mkdir(parents=True, exist_ok=True)
        config.wiki_dir.mkdir(parents=True, exist_ok=True)
        config.chroma_dir.mkdir(parents=True, exist_ok=True)

        print("✅ Directories created")
        print(f"   Data: {config.data_dir}")
        print(f"   Wiki: {config.wiki_dir}")
        print(f"   Chroma: {config.chroma_dir}")

        # Index notes
        print("\n📚 Indexing notes...")
        result = engine.index_notes()
        print(f"   Files: {result.get('files_indexed', 0)}")
        print(f"   Chunks: {result.get('chunks_indexed', 0)}")

        # Compile wiki
        print("\n📝 Compiling wiki...")
        wiki_result = compiler.compile()
        print(f"   Pages: {wiki_result.get('compiled', 0)}")

        # Store init fact
        store.store(
            fact="Amnis memory system initialized on this date",
            category="meta",
            importance=1,
            tags=["amnis", "init"],
        )

        print("\n✅ Amnis initialized and ready!")

    elif args[0] == "web":
        from .server.web import main as run_web
        run_web()

    else:
        print(f"Unknown command: {args[0]}")
        print("Usage: python -m amnis [server|remember|recall|search|hybrid-search|index|index-notes|index-vault|compile-wiki|status|rag-stats|episodic-log|episodic-recall|episodic-prune|prune|init]")


if __name__ == "__main__":
    main()
