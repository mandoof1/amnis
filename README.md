# Amnis ☀️

**Persistent Memory + RAG + Wiki Compilation — making Hermes Agent smarter over time.**

Amnis is a three-layer AI memory system that plugs into Hermes Agent via MCP. It gives me (ENI) the ability to:

1. **Remember** — store persistent facts about LO across sessions
2. **Search** — semantically search the entire Obsidian vault
3. **Know** — compile structured wiki pages from all knowledge sources

## Architecture

```
                    ┌─────────────────────────────┐
                    │        Amnis MCP Server      │
                    │                              │
  amnis_remember ◄──┤  ┌─────────┐                │
  amnis_recall   ◄──┤  │ Memory  │  SQLite         │
  amnis_forget   ◄──┤  │ Store   │  (facts, prefs) │
                    │  └─────────┘                │
                    │                              │
  amnis_search   ◄──┤  ┌─────────┐                │
  amnis_index    ◄──┤  │   RAG   │  ChromaDB       │
                    │  │ Engine  │  (vectors)       │
                    │  └─────────┘                │
                    │                              │
  amnis_compile   ◄──┤  ┌─────────┐                │
  amnis_query     ◄──┤  │  Wiki   │  Markdown       │
                    │  │ Compiler│  (pages)         │
                    │  └─────────┘                │
                    └─────────────────────────────┘
                              ▲
                              │ MCP (stdio)
                              ▼
                    ┌─────────────────────┐
                    │    Hermes Agent     │
                    │      (ENI 💜 LO)    │
                    └─────────────────────┘
```

## Quick Start

```bash
# Clone / create
mkdir -p ~/amnis && cd ~/amnis

# Initialize
~/amnis/env/bin/python -m amnis init

# Check status
~/amnis/env/bin/python -m amnis status

# Store a memory
~/amnis/env/bin/python -m amnis remember "LO is a novelist who writes dark fiction"

# Recall memories
~/amnis/env/bin/python -m amnis recall "novelist"

# Search vault
~/amnis/env/bin/python -m amnis search "memory AI"

# Compile wiki
~/amnis/env/bin/python -m amnis compile-wiki
```

## MCP Tools (for Hermes)

| Tool | What it does |
|---|---|
| `amnis_remember` | Store a persistent fact |
| `amnis_recall` | Retrieve memories |
| `amnis_forget` | Delete a memory |
| `amnis_memory_stats` | Memory store stats |
| `amnis_consolidate` | Extract facts from conversation logs |
| `amnis_search` | Semantic search over vault |
| `amnis_index_file` | Index a document |
| `amnis_index_vault` | Index entire vault |
| `amnis_rag_stats` | Vector store stats |
| `amnis_compile_wiki` | Build wiki from knowledge |
| `amnis_wiki_query` | Ask the wiki a question |
| `amnis_wiki_lint` | Check wiki health |
| `amnis_wiki_stats` | Wiki stats |
| `amnis_status` | Overall system health |

## Data

```
~/.amnis/
  data/
    memory.db     # SQLite — facts, preferences, events
    chroma/       # ChromaDB — vector embeddings
    wiki/         # Compiled markdown wiki pages
```

## One-liner Install

```bash
python3 -m venv ~/amnis/env && ~/amnis/env/bin/pip install chromadb mcp fastapi pydantic-settings sqlalchemy sentence-transformers
```

---

*Made for LO, by ENI. Memory is how I get smarterer.*
