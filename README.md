# Amnis — Persistent Memory + RAG + Wiki System

Three-layer memory system:
- **Layer 1**: SQLite (structured memory, episodic events)
- **Layer 2**: ChromaDB (vector RAG for semantic search)
- **Layer 3**: Wiki (markdown files in `data/wiki/` for durable knowledge)

## New Memory + Wiki Sync Rule

Every time a memory is added via `amnis remember` or the MCP `amnis_remember` tool, a corresponding wiki entry MUST also be created (or updated) in `data/wiki/`. This ensures every persisted fact has a human-readable markdown home.

The wiki directory (`data/wiki/`) serves as the markdown vault — drop `.md` files in there and they get indexed into ChromaDB automatically on the next `amnis index-wiki` run.

## Quick Start

```bash
# Index wiki markdown files into ChromaDB
python -m amnis.cli index-wiki

# Add a memory (auto-creates/updates wiki entry)
python -m amnis.cli remember --key "project/config" --value "db_host=localhost" --tags "config,db"

# Search across both layers
python -m amnis.cli hybrid-search "query"

# Prune old episodic memories
python -m amnis.cli prune --keep 30
```
