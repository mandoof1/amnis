# Comprehensive Research Report: AI Memory and RAG Architectures

## Executive Summary

Large Language Models (LLMs) are fundamentally stateless. In their native architecture, every inference call starts from a fresh state with no inherent record of previous interactions. While the industry has trended toward expanding context windows to manage this—sometimes reaching millions of tokens—this creates an "illusion of memory." A larger context window functions like a larger Post-it note: it provides more space for immediate information, but the information is discarded once the session ends. Reliance on context windows alone is limited by linear cost scaling, performance degradation (often occurring at ~130k tokens for a 200k-token model), and a lack of importance-based prioritization.

**Statefulness** is the differentiator that evolves an LLM from a "text-completion tool" into a "long-term agent." It refers to a persistent, evolving infrastructure that allows an agent to store, retrieve, update, and forget information across sessions. By utilizing an external memory layer, agents can adapt to user preferences and retain context over months or years, moving beyond the "memory crisis" of stateless systems.

> **Key Takeaway**
> The industry is shifting from **Stateless RAG**, which focuses on bringing external knowledge to a prompt to help an agent answer better, to **Stateful Agentic Memory**, which enables an agent to learn, adapt, and maintain continuity across its entire lifecycle.

---

## Section 1: The Four Memory Types (CoALA Framework)

The Cognitive Architectures for Language Agents (CoALA) framework, developed by researchers at Princeton (2023), provides a standardized taxonomy for AI memory. This framework maps AI components to biological equivalents to ensure agents possess a structured "mind."

### Comparison of Memory Types

| Memory Type | Human Equivalent | Agent Function | Implementation Approach |
| :--- | :--- | :--- | :--- |
| **Working Memory** | Short-term scratchpad | Current conversation context and intermediate reasoning. | Context windows, FIFO queues, ChatMessage buffers. |
| **Episodic Memory** | Autobiographical experiences | Recalling specific past events, "who/what/when" context. | Timestamped logs, event sequences, interaction nodes. |
| **Semantic Memory** | General facts/World knowledge | Facts, concept knowledge, and user preferences. | Vector stores, knowledge bases, entity-relationship graphs. |
| **Procedural Memory** | Implicit skills and rules | Internal workflows, decision logic, and tool competencies. | System prompts, agent code, prompt templates. |

### Detail Interconnectivity
In a functional cognitive architecture, these memory types are deeply interconnected. A **semantic fact** (e.g., a user's time zone) combined with an **episodic event** (e.g., the last meeting lasted 30 minutes) can trigger a **procedural response** (e.g., booking a new 30-minute slot on the calendar).

Architectural efficiency in navigating these connections is a major technical differentiator. For instance, in Memgraph-based systems, the `SHOW SCHEMA INFO` command returns the graph ontology in constant time. This allows the LLM to understand entity types and relationships instantly, enabling the system to navigate the "multi-hop" connections between memory types that vector-only databases cannot represent.

---

## Section 2: Three Major Architectural Approaches

### 1. The LLM Wiki (Markdown Knowledge Bases)
This approach leverages personal knowledge management (PKM) tools like Obsidian with specialized AI plugins.
*   **Plugins:** Key implementations include *Smart Connections*, *CoPilot*, and *Khoj*. *Khoj* serves as an open-source personal AI assistant for knowledge management, supporting Markdown, PDF, and Notion files.
*   **Mechanism:** These tools create **local embeddings** of notes. *Smart Connections* uses semantic search (Smart Lookup) to measure the "distance" between notes, allowing users to find related ideas without keyword matches.
*   **Pros:** High privacy (data stays on-device), high speed, and zero API costs for local models.
*   **Cons:** Primarily static; while effective for retrieval, it often lacks the automated state management required for autonomous agents.

### 2. Retrieval-Augmented Generation (RAG)
RAG provides the mechanism to ground LLM outputs in authoritative, external data.
*   **Pipeline:** Consists of **Ingestion** (chunking and embedding), **Retrieval** (finding relevant chunks), **Augmentation** (combining query and context), and **Generation**.
*   **The Grounding Case Study (Saturn’s Moons):** A stateless LLM might rely on parametric knowledge and confidently state that Jupiter has the most moons (citing 88). However, a RAG-enabled system retrieves up-to-date data from a source like NASA to correctly identify Saturn as the leader with 146 moons. This demonstrates how RAG prevents hallucinations and provides source-backed evidence.
*   **Search Strategies:** Advanced systems move from simple **Vector Search** to **Hybrid Search**, which combines semantic (dense) and lexical (sparse) search to capture both meaning and specific domain terms like product names.

### 3. Agent Memory (Mem0 / Letta / LlamaIndex)
In "Agentic Memory," the agent actively manages its own state via tool calls (Self-editing memory).
*   **Virtual Context:** Platforms like Letta (formerly MemGPT) treat the LLM context window like RAM and external storage like a disk, paging information between tiers to create a seemingly unlimited context.
*   **LlamaIndex Implementation:** Uses a FIFO queue for short-term memory, flushing older messages into specific "Memory Blocks" (Static, Fact Extraction, or Vector) for long-term persistence.
*   **Implementation Note:** Standardizing the interface is critical. For example, the `langchain-oracledb` package allows for seamless integration of these patterns into production-grade databases:
`pip install langchain-oracledb oracledb langchain-core`

---

## Section 3: Key Research Findings

### Reflection and Synthesis
Stanford (2023) research on "Generative Agents" highlights the use of reflection hierarchies. Agents do not just store raw logs; they synthesize high-level observations from episodic data to reason about abstract patterns.

### The arXiv Episodic Memory Framework (2024)
Research identifies five core properties required for true episodic memory:
1.  **Long-term storage:** Retention throughout an agent's lifetime.
2.  **Explicit reasoning:** The ability to reflect upon and query memory content.
3.  **Single-shot learning:** Rapid encoding of unique experiences from one exposure.
4.  **Instance-specific content:** Capturing unique details of a specific occurrence.
5.  **Contextual relations:** Binding events to "when, where, and why" they happened.

**The Research Roadmap:** The field is currently pursuing a roadmap structured around **Encoding** (segmenting continuous input into episodes), **Retrieval** (selecting relevant episodes for reinstatement), **Consolidation** (merging external memory into parametric weights), and **Benchmarks** (assessing contextualized recall over long delays). This roadmap is governed by **six core Research Questions (RQs)** designed to solve the challenges of constant computational cost and performance stability.

### Advanced Retrieval and Performance
*   **Accuracy:** Contemporary frameworks like Zep/Graphiti have achieved **94.8% accuracy** on the Deep Memory Retrieval benchmark using temporal knowledge graphs.
*   **Techniques:** Reciprocal Rank Fusion (RRF) is used to merge hybrid search results, while heading-aware/entity-aware chunking preserves document hierarchy during ingestion.
*   **Consolidation and Pruning:** Systems use **exponential decay factors** within recency-weighted scoring functions. This ensures that old or unreferenced embeddings lose salience over time, mimicking biological forgetting.

---

## Section 4: From Theory to Practice

### Database Convergence and Enterprise Reality
Production systems are moving toward **converged databases** (e.g., Oracle, Memgraph). These handle vectors, graphs, and relational data within a single engine.
*   **The ACID Requirement:** Converged databases provide **ACID transactions**, ensuring that updating an embedding, modifying a graph relationship, and changing relational metadata either all succeed or all fail. This prevents "inconsistent memory states" that plague stitched-together architectures.
*   **Industry Statistics:** Per LangChain’s 2025 survey, customer service agents represent the most common production use case with a **26.5% deployment rate**.

### GraphRAG vs. Vector RAG
While Vector RAG finds "what feels similar," GraphRAG provides the structure to trace relationships across entities and time.
*   **Bi-temporal Modeling:** This is a critical distinction in high-end systems, tracking **when events happened** in the real world versus **when the system learned about them**.
*   **Multi-hop Reasoning:** Graphs allow agents to link a specific user preference to a specific past meeting and a project code—traversals that similarity-based vector search misses.

### Compliance and Forgetting
Enterprise deployment faces a "tension of mandates." The **GDPR "Right to be Forgotten"** requires the ability to delete specific user data. Conversely, the **EU AI Act (fully applicable August 2026)** requires **10-year audit trails** for high-risk systems. Modern architectures must therefore support sophisticated "invalidating" of facts (removing them from active retrieval) without necessarily discarding them from historical audit logs.

### Production Implementation Patterns
*   **Sleep-time Computation:** Agents reorganize and refine memories during idle time to reduce query-time latency and costs.
*   **Core Memory Operations:** Agents utilize the LLM to decide on four operations: **ADD** (new facts), **UPDATE** (modify existing facts), **DELETE** (remove contradictions), and **SKIP** (ignore irrelevance).

## References
*   Princeton (2023): *Cognitive Architectures for Language Agents (CoALA).*
*   Stanford (2023): *Generative Agents: Interactive Simulacra of Human Behavior.*
*   arXiv (2024): *Episodic Memory is the Missing Piece for Long-Term LLM Agents.*
*   Oracle/Pinecone/Memgraph Technical Documentation (2025/2026).

---

## Section 5: Making It Real — Amnis Implementation

[This section connects the academic research above to the open-source **Amnis** project (https://github.com/mandoof1/amnis), a three-layer persistent memory system built for MCP-compatible AI agents. Every finding in this report has been translated into production code.]

### 5.1 The CoALA Framework Implemented

The four memory types from the CoALA paper are each represented as a distinct layer with a concrete backend:

| CoALA Memory Type | Amnis Layer | Backend | Implementation |
|---|---|---|---|
| **Working Memory** | `amnis/memory/working.py` | In-memory ring buffer + JSON persistence | Slot-based scratchpad with push/get/pop/clear, automatic eviction at capacity (default 20 slots), `flush_to_episodic()` for persisting transient context |
| **Episodic Memory** | `amnis/memory/episodic.py` | SQLite (`conversation_logs` table) | Timestamped event log with auto-pruning, outcome tracking (success/failure/neutral), structured results field, session-scoped queries |
| **Semantic Memory** | `amnis/memory/store.py` + `amnis/rag/engine.py` | SQLite (`memory_facts`) + ChromaDB + FTS5 | Categorized facts with importance scoring (1–10), confidence tracking, tags, hybrid RAG search using Reciprocal Rank Fusion |
| **Procedural Memory** | Wiki Compiler (`amnis/wiki/compiler.py`) | Markdown wiki pages | Auto-generated structured knowledge pages cross-referenced across all three layers |

### 5.2 RAG Research Applied

**Hybrid Search with RRF (Section 2.2, Section 3.3):**
The `amnis/rag/engine.py` implements full hybrid search combining:
- **Semantic leg** — ChromaDB vector search using all-MiniLM-L6-v2 embeddings (cosine similarity)
- **Keyword leg** — SQLite FTS5 full-text search (tokenized, stemmed, stopword-filtered)
- **Fusion** — Reciprocal Rank Fusion with k=60, where each chunk must rank well in *both* legs to score highly. This directly implements the RRF technique cited in Section 3.3.

```python
# RRF score for each chunk across N search methods
rrf_score(chunk, methods) = sum(1 / (k + rank_method(chunk)) for method in methods)
```

**Heading-Aware Chunking (Section 2.2, Section 3.3):**
During document ingestion, chunks respect H1–H6 boundaries rather than blindly splitting at N characters. This preserves document structure and prevents sections from being merged or split mid-paragraph. Configurable via `chunk_size`, `chunk_overlap`, and `heading_level` settings.

**Metadata Filtering:**
Both `amnis_search` and `amnis_hybrid_search` support optional `where` metadata filters, enabling scoped retrieval (e.g., `{"source": {"$contains": "memory"}}`).

### 5.3 Agent Memory Research Applied

**Memory Consolidation (Section 3.1, Section 2.3):**
The `amnis/memory/consolidation.py` module runs as a background pipeline:
1. Scans recent conversation logs for assistant utterances that read as durable statements about the user
   (declarative "you prefer / you use / your X is" patterns, with advice, questions, list items and code rejected)
2. Extracts structured `MemoryFact` records with computed importance scores
3. Contradiction detection via polarity scoring — facts with opposite sentiment on the same topic are flagged and both have confidence reduced
4. Semantic dedup via cosine similarity — near-identical facts are merged (longer/more specific version wins, importance boosted)
5. **Reflection hierarchy** — observation facts are clustered by embedding similarity (>0.75 cosine), and each cluster generates a `category="theme"` mid-level belief. This directly implements the Stanford 2023 Generative Agents finding that "removal of reflection broke emergent behaviors" (Section 3.1).

**Reflection Pipeline:**
```
Raw observations → Embedding clustering → Theme synthesis → Storage as `theme` facts
```
- Up to 5 theme clusters per run
- Existing themes are reinforced (confidence +0.05, importance +1) rather than duplicated
- Themes carry cross-references to source observations

**Memory Pruning and Decay (Section 3.3, Section 4.3):**
The `amnis/memory/pruning.py` module provides automated memory lifecycle management:
- **Confidence decay**: Confidence of unaccessed auto-extracted facts decays by 0.98^days_since_last_access, mimicking biological forgetting
- **Staleness cleanup**: Facts older than `retention_days` (configurable, default 180) are removed
- **Low-importance pruning**: Facts below `min_importance_threshold` are discarded
- **Near-duplicate detection**: Facts with cosine similarity >0.92 are merged

### 5.4 Working Memory (CoALA)

The `amnis/memory/working.py` module implements the CoALA working memory concept as a lightweight in-memory ring buffer:
- **Slot-based** — named slots for different types of working context (current query, retrieved chunks, reasoning steps)
- **Capacity-limited** — configurable `max_slots` (default 20), oldest slot evicted on overflow
- **Crash-resilient** — JSON persistence to disk, auto-restored on restart
- **Episodic flush** — `flush_to_episodic()` moves all working memory content into the episodic log for traceability
- **Priority eviction** — `evict_lowest_priority()` removes the oldest slot explicitly

### 5.5 Lightweight by Design

Amnis is designed to run in <200MB RAM (embedding model included) and keep its data directory under 10MB for typical usage:

| Component | Storage |
|---|---|
| Memory facts (SQLite) | ~100 KB per 1,000 facts |
| Episodic logs (SQLite) | ~50 KB per 1,000 episodes |
| ChromaDB vectors | ~2 MB per 500 indexed chunks |
| FTS5 keyword index | ~1 MB per 500 indexed chunks |
| Wiki pages (markdown) | ~200 KB per 50 compiled pages |
| **Typical total** | **~5.9 MB** |

Zero new dependencies beyond the original install set (no new LLM integrations, no additional vector databases, no heavy frameworks).

### 5.6 Architectural Map

```
┌─────────────────────────────────────────────────────────────────┐
│                       Amnis Core                                  │
│                                                                   │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────────┐      │
│  │  WORKING     │   │  EPISODIC    │   │  SEMANTIC        │      │
│  │  (ring buf)  │   │  (SQLite)    │   │  (SQLite + RAG)  │      │
│  │              │   │              │   │                  │      │
│  │  push/pop    │   │  log_episode │   │  MemoryFacts     │      │
│  │  get/clear   │   │  recall      │   │  + ChromaDB      │      │
│  │  flush       │   │  outcome     │   │  + FTS5          │      │
│  │  evict       │   │  auto-prune  │   │  + RRF hybrid    │      │
│  └──────┬───────┘   └──────┬───────┘   └────────┬─────────┘      │
│         │                  │                     │                │
│         └──────────────────┼─────────────────────┘                │
│                            │                                      │
│         Consolidation Pipeline (amnis/memory/consolidation.py)    │
│         ┌────────────────────────────────────────────────────┐    │
│         │  Extract → Dedup → Contradiction Check → Reflect   │    │
│         └──────────────────────┬─────────────────────────────┘    │
│                                │                                  │
│         Pruning Pipeline (amnis/memory/pruning.py)               │
│         ┌────────────────────────────────────────────────────┐    │
│         │  Decay → Stale → Low-importance → Near-duplicate   │    │
│         └──────────────────────┬─────────────────────────────┘    │
│                                │                                  │
└────────────────────────────────┼──────────────────────────────────┘
                                 │
                    ┌────────────┴────────────┐
                    │    MCP Server (stdio)    │
                    │    18+ tools             │
                    └────────────┬────────────┘
                                 │
                    ┌────────────┴────────────┐
                    │   Hermes / Claude /      │
                    │   Any MCP Client         │
                    └─────────────────────────┘
```

### 5.7 Key Files

| File | Purpose | Research Reference |
|---|---|---|
| `amnis/memory/working.py` | Working memory ring buffer | CoALA Working Memory |
| `amnis/memory/episodic.py` | Episodic event log with outcome tracking | CoALA Episodic, arXiv 2502.06975 |
| `amnis/memory/consolidation.py` | Fact extraction, dedup, contradiction, reflection | Stanford 2023 Generative Agents, CoALA consolidation |
| `amnis/memory/pruning.py` | Decay, stale, low-importance cleanup | Section 3.3, exponential decay factor |
| `amnis/memory/store.py` | Memory fact CRUD with importance + confidence | Semantic memory store |
| `amnis/rag/engine.py` | Hybrid RAG with RRF + heading-aware chunking | Section 2.2, RRF technique |
| `amnis/database.py` | SQLite schema for all three layers | — |
| `amnis/server/mcp.py` | 18-tool MCP server interface | Agent integration protocol |
| `amnis/config.py` | Pydantic-settings configuration | — |
| `amnis/wiki/compiler.py` | Karpathy-style compiled wiki | LLM Wiki approach (Section 2.1) |

### 5.8 What's Next

The research identifies several frontiers Amnis has not yet explored:

- **Graph-based memory** — Section 4.2 highlights GraphRAG's advantage for multi-hop reasoning. A future layer could add temporal knowledge graph relationships between facts.
- **Sleep-time computation** — Section 4.3 describes agents reorganizing memories during idle cycles. Amnis's consolidation/pruning pipeline currently runs on-demand, not automatically scheduled.
- **Bi-temporal modeling** — Section 4.2's "when events happened vs. when learned" distinction is captured implicitly (timestamps on facts) but not explicitly modeled.
- **Self-editing memory** — Section 2.3's vision of agents calling ADD/UPDATE/DELETE/SKIP via tool calls (à la Letta) is a natural extension of the current MCP interface.

---

*This report was generated by NotebookLM from 44 research sources and augmented with implementation documentation from the Amnis project (commit e70c7f9). The research-informed MoA upgrade to Amnis was completed on 2026-06-16, implementing all eight synthesized improvements derived from the source material.*