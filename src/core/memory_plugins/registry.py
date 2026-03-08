
from typing import Dict, Type
from .base import MemoryPlugin
from .lancedb import LanceDBPlugin

_PLUGINS: Dict[str, Type[MemoryPlugin]] = {
    "lancedb": LanceDBPlugin,
    # Future: "native": OpenClawNativePlugin
}

def get_plugin(name: str) -> MemoryPlugin:
    plugin_cls = _PLUGINS.get(name.lower())
    if not plugin_cls:
        raise ValueError(f"Unknown memory plugin: {name}. Available: {list(_PLUGINS.keys())}")
    return plugin_cls()
