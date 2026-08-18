# Amnis ☀️

**Persistent Memory + RAG + Wiki Compilation for AI Agents**

Amnis is a three-layer memory system that gives AI agents long-term, structured knowledge across sessions. It plugs into any MCP-compatible client (Hermes, Claude, etc.) and provides a self-hosted web dashboard for visualization.

---

## Why Amnis?

LLMs are stateless. Every conversation starts from zero. Amnis fixes that by adding:

| Layer | Purpose | Backend |
|-------|---------|---------|
| **Memory Store** | Persistent facts, preferences, events with categories, importance, tags | SQLite |
| **RAG Engine** | Hybrid search over local documents (markdown notes, code, plain text) | ChromaDB + sentence-transformers + FTS5 |
| **Wiki Compiler** | Karpathy-style structured knowledge pages auto-generated from all sources | Markdown + cross-refs |

Together, these create a **personal knowledge graph** that grows smarter over time.

---

## Features

- **Hybrid RAG search** — Reciprocal Rank Fusion (RRF) blending semantic (ChromaDB) + keyword (FTS5) results for precision
- **Heading-aware chunking** — Respects H1–H6 boundaries so sections stay intact
- **Episodic memory** — Chronological event log with auto-pruning, outcome tracking, and per-episode importance
- **Reflection hierarchy** — Observations auto-clustered into mid-level themes (Stanford Generative Agents paper)
- **Working memory** — In-memory ring buffer for agent scratchpad, with JSON persistence and episodic flush
- **Pruning pipeline** — Low-importance, stale, and near-duplicate fact cleanup with confidence decay (0.98^days)
- **Contradiction detection** — Polarity-based flagging of conflicting facts
- **Semantic dedup** — Cosine-similarity merging of near-identical memories
- **Web UI** — Dashboard, interactive knowledge graph, memory browser, wiki viewer
- **MCP server** — 18+ tools for agent integration

---

## Research Foundation

Amnis is built on published research across cognitive architectures, generative agents, information retrieval, and episodic memory. Every layer maps to a specific paper or finding.

### CoALA Framework (Princeton 2023)

The [Cognitive Architectures for Language Agents](https://arxiv.org/abs/2309.02427) paper defines four memory types for AI agents. Amnis implements all four as concrete backends:

| CoALA Memory Type | Amnis Implementation | Backend |
|---|---|---|
| **Working Memory** — agent's scratchpad for current context | `amnis/memory/working.py` — ring buffer with push/get/pop/clear, JSON persistence, automatic eviction at 20 slots, `flush_to_episodic()` for persisting transient context | In-memory + JSON file |
| **Episodic Memory** — autobiographical recall of past interactions | `amnis/memory/episodic.py` — timestamped event log with outcome tracking (success/failure/neutral), structured results, session-scoped queries, auto-pruning | SQLite (`conversation_logs`) |
| **Semantic Memory** — facts, preferences, domain knowledge | `amnis/memory/store.py` + `amnis/rag/engine.py` — categorized facts with importance scoring (1–10), confidence tracking, tags, hybrid RAG search using Reciprocal Rank Fusion | SQLite (`memory_facts`) + ChromaDB + FTS5 |
| **Procedural Memory** — how to act, workflows, tool selection | Wiki Compiler (`amnis/wiki/compiler.py`) — auto-generated structured knowledge pages cross-referenced across all three layers | Markdown wiki |

### Generative Agents (Stanford 2023)

The Stanford paper on [Generative Agents: Interactive Simulacra of Human Behavior](https://arxiv.org/abs/2304.03442) found that **removing reflection broke emergent behaviors** — agents stopped forming high-level beliefs from raw observations. Amnis implements a reflection hierarchy in `amnis/memory/consolidation.py`:

```
Raw observations → Embedding clustering (cosine >0.75) → Theme synthesis → Stored as `theme` facts
```

- Up to 5 theme clusters per consolidation run
- Existing themes are reinforced (confidence +0.05, importance +1) rather than duplicated
- Themes carry cross-references to source observations for full traceability

### Episodic Memory (arXiv 2024)

The [Episodic Memory is the Missing Piece for Long-Term LLM Agents](https://arxiv.org/abs/2502.06975) paper identifies five required properties for episodic memory. Amnis meets all five:

| Property | Amnis Implementation |
|---|---|
| Long-term storage | SQLite persistence across sessions with configurable retention (default 180 days) |
| Explicit reasoning | `amnis_episodic_recall` MCP tool returns structured episodes with summaries, topics, and outcome |
| Single-shot learning | Each `log_episode()` call encodes a unique experience from one exposure |
| Instance-specific content | Every episode captures full role + content + timestamp + outcome |
| Contextual relations | Episodes are bound to `session_id`, ranked by importance, filterable by outcome |

### Hybrid Search with RRF

Amnis uses **Reciprocal Rank Fusion** (k=60) to blend semantic search (ChromaDB cosine similarity) with keyword search (SQLite FTS5 full-text search). This technique, widely cited in the RAG literature, ensures chunks must rank well in **both** methods to score highly:

```
rrf_score(chunk) = 1/(60 + rank_semantic) + 1/(60 + rank_keyword)
```

Combined with **heading-aware chunking** (respects H1–H6 boundaries, configurable via `chunk_size`/`chunk_overlap`), this preserves document structure during ingestion — sections stay intact rather than being split mid-paragraph.

### Memory Consolidation Pipeline

Inspired by the Mem0/Letta agent memory frameworks and the Generative Agents consolidation process, Amnis runs a background pipeline at `amnis/memory/consolidation.py`:

1. **Extract** — scan recent conversation logs, extract structured `MemoryFact` records with computed importance scores
2. **Contradiction detection** — polarity scoring flags facts with opposite sentiment on the same topic, confidence is reduced on both
3. **Semantic dedup** — cosine similarity >0.9 merges near-identical facts (longer/more specific version wins)
4. **Reflect** — cluster observations into theme-level beliefs (Stanford finding)
5. **Prune** — decay confidence by 0.98^days since last access, remove stale/low-importance/near-duplicate facts at `amnis/memory/pruning.py`

### Disk Impact

All of this runs in **~5.9 MB** for a typical setup (242 memories + 425 RAG chunks + 32 wiki pages). Zero new dependencies beyond ChromaDB, sentence-transformers, and SQLite.

```
Memory facts (SQLite)      ~100 KB / 1,000 facts
Episodic logs (SQLite)      ~50 KB / 1,000 episodes
ChromaDB vectors           ~2 MB / 500 chunks
FTS5 keyword index         ~1 MB / 500 chunks
Wiki pages (markdown)     ~200 KB / 50 pages
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      Amnis Core                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐   │
│  │   Memory     │  │    RAG       │  │     Wiki         │   │
│  │   Store      │  │   Engine     │  │   Compiler       │   │
│  │  (SQLite)    │  │ (ChromaDB    │  │   (Markdown)     │   │
│  │              │  │  + FTS5)     │  │                  │   │
│  └──────┬───────┘  └──────┬───────┘  └────────┬─────────┘   │
│         │                 │                   │              │
└─────────┼─────────────────┼───────────────────┼──────────────┘
          │                 │                   │
          ▼                 ▼                   ▼
┌─────────────────────────────────────────────────────────────┐
│                    MCP Server (stdio)                        │
│  18+ tools: remember, recall, forget, search, hybrid-search,│
│  episodic-log, episodic-recall, prune, consolidate,         │
│  compile-wiki, wiki_query, status, ...                      │
└─────────────────────────────────────────────────────────────┘
                              ▲
                              │ MCP / HTTP
              ┌───────────────┼───────────────┐
              ▼               ▼               ▼
       ┌──────────┐    ┌───────────┐   ┌──────────┐
       │  Hermes  │    │  Web UI   │   │  Other   │
       │  Agent   │    │ (port     │   │  MCP     │
       │          │    │  8799)    │   │  Clients │
       └──────────┘    └───────────┘   └──────────┘
```

---

## Quick Start

### Prerequisites
- Python 3.11+
- ~2GB RAM for embedding model (all-MiniLM-L6-v2, 33MB)

### Install

```bash
git clone https://github.com/mandoof1/amnis.git
cd amnis

python3 -m venv env
source env/bin/activate          # fish: source env/bin/activate.fish
pip install -e .

# For the test suite and linter:
pip install -e ".[dev]"
```

This installs an `amnis` command on your PATH. `python -m amnis ...` works
identically everywhere below.

### Initialize & Run

```bash
amnis init            # create data dirs, index notes, compile the wiki
amnis status          # health and size of every layer
amnis server          # MCP stdio server (this is also the default)
amnis web             # dashboard on http://127.0.0.1:8799/
```

### Upgrading from 0.1

0.1 had no keyword index for memories, so recall fell back to a substring
match. Build the index once:

```bash
amnis reindex
```

The database schema migrates itself on first run — two columns are added and
the missing indexes are created in place. Nothing is dropped or rewritten.

---

---

## CLI Commands

| Command | Description |
|---------|-------------|
| `init` | Create data dirs, index notes, compile the wiki |
| `server` | Run the MCP stdio server (default when no command is given) |
| `web [--host H] [--port P]` | Run the dashboard |
| `remember "fact" [-c CATEGORY] [-i 1-10] [-t a,b] [--context X]` | Store a memory |
| `recall ["query"] [-c CATEGORY] [-n N] [-m MIN] [-t a,b] [--no-semantic]` | Retrieve memories |
| `forget <memory_id>` | Delete a memory and everything derived from it |
| `search "query" [-n N] [--mode hybrid\|semantic\|keyword]` | Search indexed documents |
| `index <path> [--origin note\|wiki\|memory\|compiled]` | Index a single file |
| `index-notes` | Index everything under the notes directory |
| `index-wiki` | Index the wiki directory |
| `reindex` | Rebuild the memory keyword index |
| `compile-wiki [topic ...]` | Build wiki pages |
| `wiki-query "question"` | Ask the wiki a question |
| `wiki-lint` | Report stale, sourceless, and duplicate pages (exit 1 if any) |
| `consolidate` | Extract facts from recent logs, then reflect |
| `prune [--dry-run]` | Decay, deduplicate, and prune the memory store |
| `episodic-log <session> <role> <content>` | Log a conversation turn |
| `episodic-recall [--session-id S] [--topic T] [--role R] [-n N]` | Retrieve episodes |
| `episodic-prune [--days N]` | Drop episodes past the retention window |
| `status` | Show memory/RAG/wiki/episodic stats |

Every command accepts `--json` for machine-readable output.

---

## MCP Tools (for Hermes, Claude, etc.)

Configure in your MCP client's config (Hermes `config.yaml` shown):

```yaml
mcp_servers:
  amnis:
    command: /path/to/amnis/env/bin/python
    args: ["-m", "amnis.server.mcp"]
```

All 23 tools generate their JSON Schema from Python type hints, so a tool's
signature and its advertised schema cannot drift apart.

### Memory Tools

| Tool | Description |
|------|-------------|
| `amnis_remember` | Store a fact with category, importance (1-10), tags |
| `amnis_recall` | Recall memories by meaning and keyword |
| `amnis_update_memory` | Edit a memory in place, keeping its ID and history |
| `amnis_forget` | Delete a memory and its wiki mirror and vectors |
| `amnis_memory_stats` | Totals, category breakdown, average importance |
| `amnis_consolidate` | Extract facts from conversation logs, then reflect |
| `amnis_reindex_memories` | Rebuild the memory keyword index |
| `amnis_prune_memory` | Cleanup pipeline (decay, stale, duplicates); `dry_run` supported |

### Search Tools

| Tool | Description |
|------|-------------|
| `amnis_search` | Semantic search, with an optional metadata filter |
| `amnis_hybrid_search` | RRF fusion of semantic and FTS5 keyword search |
| `amnis_index_file` | Add one file to the vector store |
| `amnis_index_notes` | Index the whole notes directory |
| `amnis_rag_stats` | Chunk count, source count, embedding model |
| `amnis_list_sources` | List indexed documents, filterable by origin |

### Episodic Tools

| Tool | Description |
|------|-------------|
| `amnis_episodic_log` | Log a turn, auto-extracting topics and a summary |
| `amnis_episodic_recall` | Retrieve episodes by session, topic, or role |
| `amnis_episodic_stats` | Episode count, session count, date range |
| `amnis_episodic_prune` | Drop episodes past the retention window |

### Wiki Tools

| Tool | Description |
|------|-------------|
| `amnis_compile_wiki` | Build wiki pages from notes and memory |
| `amnis_wiki_query` | Ask the wiki a question |
| `amnis_wiki_lint` | Find stale pages, missing sources, duplicate titles |
| `amnis_wiki_stats` | Page count, distinct source count, last compile |

### System Tools

| Tool | Description |
|------|-------------|
| `amnis_status` | Health of all layers; reports `degraded` per-layer rather than failing |

---

## Web UI

```
http://127.0.0.1:8799/
```

**Seven views:**

- **◈ Dashboard** — live stats per layer, recent memories, category distribution
- **✦ Memories** — browse, search, filter; edit in place without losing IDs
- **⌕ Search** — hybrid / semantic / keyword search with the real ranking signal shown
- **⬡ Graph** — knowledge graph of memories, wiki pages, and documents
- **▤ Wiki** — page list plus a rendered markdown viewer with working `[[wiki links]]`
- **◷ Episodes** — logged conversation turns, filterable by session and topic
- **⚙ Maintenance** — consolidate, prune (with a dry-run preview), reindex

Keyboard: `Ctrl/Cmd+K` command palette, `1`–`7` to switch views, `n` new memory,
`/` jump to search, `r` refresh.

The dashboard is a **single self-contained HTML file** — no CDN scripts, no
webfonts, no outbound requests of any kind. The graph is a small canvas force
simulation rather than a 490 KB third-party bundle, and it cools to zero CPU
once laid out. Light, dark, and system themes.

---

## Data Layout

```
~/amnis/
├── data/                   # or $AMNIS_DATA_DIR
│   ├── memory.db           # SQLite: facts, wiki pages, documents, episodes
│   ├── keyword.db          # SQLite FTS5: chunk + memory keyword indexes
│   ├── chroma/             # ChromaDB vector embeddings
│   ├── notes/              # Your source documents (indexed)
│   ├── wiki/               # Compiled .md wiki pages
│   │   └── facts/          # One .md mirror per stored memory
│   └── working_memory.json # Working-memory scratchpad
├── env/                    # Python environment
└── .env                    # Optional: AMNIS_* overrides
```

All data is **local-first**. Nothing leaves your machine.

---

## Configuration

Every setting is an environment variable prefixed `AMNIS_`, or a line in a
`.env` file in the working directory. All are optional.

```bash
# Paths — everything else derives from AMNIS_DATA_DIR unless set explicitly
export AMNIS_DATA_DIR=~/custom/amnis/data
export AMNIS_NOTES_DIR=~/Documents/Notes      # default: $AMNIS_DATA_DIR/notes
export AMNIS_WIKI_DIR=~/custom/wiki           # default: $AMNIS_DATA_DIR/wiki

# Retrieval
export AMNIS_EMBEDDING_MODEL=all-MiniLM-L6-v2
export AMNIS_CHUNK_SIZE=512
export AMNIS_CHUNK_OVERLAP=64                 # must be < chunk_size
export AMNIS_RRF_K=60

# Memory behaviour
export AMNIS_IMPORTANCE_KEYWORDS='["rust","kubernetes"]'   # +1 importance
export AMNIS_DEDUP_SIMILARITY_THRESHOLD=0.85
export AMNIS_CONFIDENCE_DECAY_RATE=0.98
export AMNIS_PRUNE_UNACCESSED_DAYS=60

# Server
export AMNIS_HOST=127.0.0.1
export AMNIS_PORT=8799
export AMNIS_API_TOKEN=...                    # require X-Amnis-Token on writes
export AMNIS_ALLOW_INDEX_OUTSIDE_NOTES=false  # confine /api/index-file
```

`amnis status` lists any `AMNIS_*` variable that no setting consumes, so a
typo surfaces instead of being silently ignored.

There is no `config.yaml`; `amnis/config.py` is the single source of truth
for the schema.

### Security note

The dashboard binds to `127.0.0.1` and its read endpoints are unauthenticated.
If you expose it beyond localhost, set `AMNIS_API_TOKEN` — every mutating
endpoint then requires a matching `X-Amnis-Token` header. `/api/index-file`
refuses paths outside the notes and wiki directories unless you opt out.

---

## Systemd Service (Auto-start on Linux)

Create `~/.config/systemd/user/amnis-web.service`:

```ini
[Unit]
Description=Amnis web dashboard
After=network.target

[Service]
Type=simple
ExecStart=%h/amnis/env/bin/python -m amnis web
WorkingDirectory=%h/amnis
Restart=on-failure
RestartSec=5

[Install]
WantedBy=default.target
```

```bash
systemctl --user daemon-reload
systemctl --user enable --now amnis-web
systemctl --user status amnis-web
```

`%h` expands to your home directory, so the unit is not user-specific.

---

## Development

```bash
pip install -e ".[dev]"
pytest          # 79 tests, ~2s (the vector layer is stubbed out)
ruff check .
ruff format --check .
```

CI runs the same three commands on Python 3.11, 3.12, and 3.13, then builds a
wheel and asserts every subpackage is actually inside it.

---

---

## Example Workflow

```bash
# 1. Index your notes (any markdown/text directory)
amnis index-notes

# 2. Store some explicit memories
amnis remember "Prefers dark mode in all editors" -c preference -i 8 -t ui,config
amnis remember "Hermes Agent uses FastAPI + MCP stdio" -c fact -i 7 -t project,architecture

# 3. Run consolidation to extract facts from conversation logs
amnis consolidate

# 4. Compile the wiki from memories + notes
amnis compile-wiki

# 5. Hybrid search
amnis search "memory architecture" --mode hybrid

# 6. Query the wiki
amnis wiki-query "What are the key architectural decisions?"

# 7. In Hermes/Claude, just ask naturally:
#   "What did I say about dark mode preference?"
#   "Search my notes for 'memory consolidation'"
#   "Compile the wiki for the 'architecture' topic"
```

---

## Roadmap

- [ ] Incremental note indexing (watch for file changes)
- [ ] Graph export (GraphML, JSON) for external tools
- [ ] Multi-vault support
- [ ] Auth proxy for remote web UI access
- [ ] Plugin system for custom memory types
- [ ] High-level belief synthesis across theme clusters

---

## Screenshots

![Amnis Dashboard](assets/screenshots/dashboard.png)
*Dashboard: stats cards, recent memories feed, memory-by-category bar chart*

![Amnis Memory Graph](assets/screenshots/graph.png)
*Interactive knowledge graph showing memory cluster relations*

---

## License

MIT — use freely, contribute back if you can. See [LICENSE](LICENSE).

---

## Credits

Built on:
- [ChromaDB](https://www.chromadb.com/) — vector database
- [sentence-transformers](https://www.sbert.net/) — embeddings
- [FastAPI](https://fastapi.tiangolo.com/) — web framework
- [SQLAlchemy](https://www.sqlalchemy.org/) — persistence, plus SQLite FTS5 for keyword search
- [MCP](https://modelcontextprotocol.io/) — agent protocol

The dashboard has no frontend dependencies: the graph, markdown renderer, and
command palette are all hand-written and ship inside a single HTML file.

## Research

A comprehensive research report on AI Memory and RAG architectures — including how Amnis maps each academic finding to production code — is available at:

➡️ **[docs/research-memory-rag.md](docs/research-memory-rag.md)**

Generated by NotebookLM from 44 sources (CoALA, Generative Agents, Mem0, Letta, and more), with implementation mapping for every layer of the Amnis system.

---

*Memory is how agents get smarter over time.*
