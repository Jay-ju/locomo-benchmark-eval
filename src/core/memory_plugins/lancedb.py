
from typing import Dict, Any
from .base import MemoryPlugin
from ...core.prompts.extraction import SESSION_EXTRACTION_PROMPT

class LanceDBPlugin(MemoryPlugin):
    """Plugin for OpenClaw's memory-lancedb-pro backend."""
    
    @property
    def name(self) -> str:
        return "lancedb"

    def get_extraction_prompt(self) -> str:
        return SESSION_EXTRACTION_PROMPT

    def normalize_entry(self, raw_entry: Dict[str, Any]) -> Dict[str, Any]:
        # The prompt already asks for the correct format, but we can enforce defaults here.
        return {
            "text": raw_entry.get("text", ""),
            "category": raw_entry.get("category", "other"),
            "importance": raw_entry.get("importance", 0.5),
            "metadata": raw_entry.get("metadata", {}),
            "scope": raw_entry.get("scope", "global"), # Usually injected by runner, but prompt might produce it
            # id, timestamp, vector are handled by the Store Adapter
        }
