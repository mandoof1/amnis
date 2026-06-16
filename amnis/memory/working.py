"""Working Memory — structured scratchpad for the CoALA working memory layer.

Working memory is the agent's "scratchpad" — current conversation context,
reasoning steps, and retrieved data that need to be tracked across
sub-turns but don't belong in long-term semantic or episodic storage.

This module provides an in-memory ring buffer with JSON persistence for
crash resilience. Contents should be small, transient, and regularly
flushed to episodic memory.

Based on the CoALA framework research:
  - Working Memory = "current conversation, reasoning steps, retrieved data"
  - Capacity-limited — explicit eviction when over budget
  - Can be flushed to episodic for longer-term traceability
"""

import json
import time
from pathlib import Path
from typing import Any, Optional
from collections import OrderedDict

from ..config import config


class WorkingMemory:
    """Ring-buffer working memory with JSON persistence.

    Each slot stores a named piece of working context. When the buffer
    exceeds max_slots, the oldest slot is evicted.

    Attributes:
        max_slots: Maximum number of slots before eviction kicks in.
        persist_path: Optional path to JSON file for crash resilience.
    """

    def __init__(
        self,
        max_slots: int = 20,
        persist_path: Optional[Path] = None,
    ):
        self.max_slots = max_slots
        self.persist_path = persist_path or (
            config.data_dir / "working_memory.json"
        )
        self._slots: OrderedDict[str, dict] = OrderedDict()
        self._load()

    # ─── Slot Operations ────────────────────────────────────────────

    def push(self, slot: str, content: Any, metadata: Optional[dict] = None) -> dict:
        """Write to a working memory slot. Creates or overwrites.

        Args:
            slot: Slot name (e.g. 'current_query', 'retrieved_chunks')
            content: The data to store (str, dict, list, etc.)
            metadata: Optional {key: value} metadata tags

        Returns:
            The slot entry created
        """
        entry = {
            "slot": slot,
            "content": content,
            "metadata": metadata or {},
            "timestamp": time.time(),
        }
        self._slots[slot] = entry
        self._evict_if_needed()
        self._save()
        return entry

    def pop(self, slot: str) -> Optional[dict]:
        """Read and remove a slot.

        Args:
            slot: Slot name to read and remove

        Returns:
            The slot entry, or None if not found
        """
        entry = self._slots.pop(slot, None)
        if entry:
            self._save()
        return entry

    def get(self, slot: str) -> Optional[dict]:
        """Read a slot without removing it. Updates last_accessed.

        Args:
            slot: Slot name to read

        Returns:
            The slot entry, or None if not found
        """
        entry = self._slots.get(slot)
        if entry:
            entry["last_accessed"] = time.time()
        return entry

    def clear(self) -> int:
        """Clear all working memory slots.

        Returns:
            Number of slots cleared
        """
        count = len(self._slots)
        self._slots.clear()
        self._save()
        return count

    # ─── Bulk Operations ────────────────────────────────────────────

    def flush_to_episodic(self, session_id: str) -> list[dict]:
        """Flush all slots to episodic memory, then clear working memory.

        Args:
            session_id: Session ID for the episodic log entries

        Returns:
            List of episodic log results
        """
        from . import episodic

        results = []
        for slot, entry in self._slots.items():
            content = entry.get("content", "")
            if isinstance(content, (dict, list)):
                content = json.dumps(content, default=str)
            else:
                content = str(content)

            result = episodic.log_episode(
                session_id=session_id,
                role="assistant",
                content=content,
                summary=f"[WM] {slot}",
                topics=["working_memory", slot],
            )
            results.append(result)

        self._slots.clear()
        self._save()
        return results

    def snapshot(self) -> dict:
        """Get a snapshot of all current slots (for context injection)."""
        now = time.time()
        return {
            "active_slots": len(self._slots),
            "max_slots": self.max_slots,
            "utilization": f"{len(self._slots)}/{self.max_slots}",
            "slots": {
                k: {
                    "preview": str(v.get("content", ""))[:200],
                    "age_seconds": int(now - v.get("timestamp", now)),
                    "metadata": v.get("metadata", {}),
                }
                for k, v in self._slots.items()
            },
        }

    def would_fit(self, text: str, limit_chars: int = 4000) -> bool:
        """Check if text fits in a slot under a character budget."""
        return len(text) <= limit_chars

    def evict_lowest_priority(self) -> Optional[str]:
        """Evict the slot with the oldest timestamp.

        Returns:
            Slot name that was evicted, or None if buffer is empty
        """
        if not self._slots:
            return None
        oldest_slot = min(
            self._slots,
            key=lambda s: self._slots[s].get("timestamp", 0),
        )
        entry = self._slots.pop(oldest_slot)
        self._save()
        return oldest_slot

    # ─── Internal ───────────────────────────────────────────────────

    def _evict_if_needed(self):
        """Evict oldest slots when over capacity."""
        while len(self._slots) > self.max_slots:
            self._slots.popitem(last=False)  # FIFO eviction

    def _save(self):
        """Persist to JSON file."""
        if not self.persist_path:
            return
        try:
            self.persist_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.persist_path, "w") as f:
                json.dump(
                    {
                        "max_slots": self.max_slots,
                        "slots": list(self._slots.values()),
                    },
                    f,
                    default=str,
                )
        except Exception:
            pass  # Non-critical — working memory recovers from scratch

    def _load(self):
        """Restore from JSON file on init."""
        if not self.persist_path or not self.persist_path.exists():
            return
        try:
            with open(self.persist_path) as f:
                data = json.load(f)
            self.max_slots = data.get("max_slots", self.max_slots)
            for entry in data.get("slots", []):
                slot = entry.get("slot")
                if slot:
                    self._slots[slot] = entry
        except Exception:
            self._slots.clear()


# Singleton — shared across the app
working_memory = WorkingMemory()
