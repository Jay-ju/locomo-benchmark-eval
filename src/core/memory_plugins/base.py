
from typing import Dict, Any, Protocol, List

class MemoryPlugin(Protocol):
    """Protocol for Memory Backend Plugins.
    
    Each backend (e.g. LanceDB, OpenClaw Native, Chroma) might require:
    1. A specific Prompt to extract memories in the right format.
    2. A specific logic to adapt/normalize the extracted JSON into storage-ready format.
    """
    
    @property
    def name(self) -> str:
        ...

    def get_extraction_prompt(self) -> str:
        """Return the system prompt for extracting memories from text/conversation."""
        ...

    def normalize_entry(self, raw_entry: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize a raw extracted item into a standardized dictionary.
        
        The standard dictionary should ideally have keys like:
        - text
        - category
        - importance
        - metadata
        - scope
        
        This allows downstream components (like LanceDBVectorStore) to ingest it.
        """
        ...
