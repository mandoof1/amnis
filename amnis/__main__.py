"""Amnis CLI — run the MCP server or execute one-off commands.

Usage:
  python -m amnis                     # Run MCP server (default)
  python -m amnis server              # Same — run MCP server
  python -m amnis remember <fact>     # Store a memory
  python -m amnis recall <query>      # Recall memories
  python -m amnis search <query>      # RAG search
  python -m amnis index <path>        # Index a file
  python -m amnis index-vault         # Index entire vault
  python -m amnis compile-wiki        # Compile wiki
  python -m amnis status              # Show stats
  python -m amnis init                # Initialize (create dirs, index vault)
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
        # Default: run MCP server
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

    elif args[0] == "index":
        if len(args) < 2:
            print("Usage: python -m amnis index <file_path>")
            return
        from .rag.engine import engine
        result = engine.index_file(args[1])
        print_json(result)

    elif args[0] == "index-vault":
        from .rag.engine import engine
        result = engine.index_vault()
        print_json(result)

    elif args[0] == "compile-wiki":
        from .wiki.compiler import compiler
        topics = args[1:] if len(args) > 1 else None
        result = compiler.compile(topics=topics)
        print_json(result)

    elif args[0] == "status":
        from .memory import store as memory_store
        from .rag.engine import engine as rag_engine
        from .wiki.compiler import compiler as wiki_compiler
        from .config import config
        print(json.dumps({
            "memory": memory_store.stats(),
            "rag": rag_engine.stats(),
            "wiki": wiki_compiler.stats(),
            "config": {
                "vault": str(config.vault_path),
                "data_dir": str(config.data_dir),
                "embedding_model": config.embedding_model,
            },
        }, indent=2))

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

        # Index vault
        print("\n📚 Indexing vault...")
        result = engine.index_vault()
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
        # Run the Web UI
        from .server.web import main as run_web
        run_web()

    else:
        print(f"Unknown command: {args[0]}")
        print("Usage: python -m amnis [server|remember|recall|search|index|index-vault|compile-wiki|status|init]")


if __name__ == "__main__":
    main()
