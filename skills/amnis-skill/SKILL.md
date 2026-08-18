# Amnis — Memory, RAG & Wiki for Hermes Agent

Use this skill when you need to remember something across sessions, search the user's knowledge base, or compile structured knowledge.

## Tools Available (via MCP server)

### Memory Layer (persistent facts)
- `amnis_remember(fact, category, importance, tags)` — Store a fact
- `amnis_recall(query, category, limit, min_importance, tags)` — Retrieve memories
- `amnis_forget(memory_id)` — Delete a memory
- `amnis_memory_stats()` — Memory store statistics
- `amnis_consolidate()` — Extract facts from conversation logs

### RAG Layer (vector search over indexed documents)
- `amnis_search(query, limit)` — Semantic search over indexed documents
- `amnis_index_file(file_path)` — Index a file
- `amnis_index_notes()` — Index all notes in Amnis's notes directory
- `amnis_rag_stats()` — Vector store statistics

### Wiki Layer (compiled knowledge)
- `amnis_compile_wiki(topics)` — Compile wiki pages from indexed notes + memory
- `amnis_wiki_query(question)` — Query the compiled wiki
- `amnis_wiki_lint()` — Check for stale/missing content
- `amnis_wiki_stats()` — Wiki statistics

### Meta
- `amnis_status()` — Overall health check

## When to Use What

| Situation | Tool |
|---|---|
| The user tells you something about themselves | `amnis_remember` — category=preference/fact, importance=7-10 |
| You need to recall past context | `amnis_recall` — search by keyword |
| The user asks about something in their notes | `amnis_search` — semantic search |
| You need compiled knowledge on a topic | `amnis_wiki_query` or `amnis_compile_wiki` |
| End of a session | `amnis_consolidate` to extract facts from recent chat |

## Category System
- `preference` — the user's likes, dislikes, habits
- `fact` — things learned about the user, their setup, their work
- `event` — things that happened
- `procedure` — how to do things
- `concept` — domain knowledge
- `meta` — about Amnis itself

## Best Practices

1. **Store immediately** — when the user tells you something, call `amnis_remember` right away
2. **Be specific** — include context in the fact, not just bare info
3. **Use tags** — tag with project names, topics, people
4. **Consolidate periodically** — run `amnis_consolidate` at end of sessions
5. **Compile wiki** — run `amnis_compile_wiki` after significant new content is added
6. **Check first** — before asking the user something you should know, run `amnis_recall` to check your memory
