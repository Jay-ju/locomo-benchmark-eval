import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import daft
from daft import DataType

logger = logging.getLogger(__name__)

@daft.udf(return_dtype=DataType.string())
def read_file_udf(path_col):
    """Read text content from files."""
    results = []
    for path in path_col:
        if not path:
            results.append("")
            continue
        try:
            # Handle potential None or non-string
            p = Path(str(path))
            if not p.exists() or not p.is_file():
                results.append("")
                continue
            text = p.read_text(encoding="utf-8").strip()
            results.append(text)
        except Exception as exc:
            logger.warning(f"Failed to read file {path}: {exc}")
            results.append("")
    return results


@daft.udf(return_dtype=DataType.python())
class ExtractionUDF:
    def __init__(self, api_key: str, base_url: str, model: str, prompt: str, temperature: float, max_tokens: int, dry_run: bool):
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.prompt = prompt
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.dry_run = dry_run
        
        if not self.dry_run:
            # We don't need to init client for litellm, but we can set env vars if needed
            # litellm uses environment variables or passed args
            pass

    def __call__(self, text_col, path_col):
        results = []
        for text, path in zip(text_col, path_col):
            if not text:
                results.append([])
                continue
            
            if self.dry_run:
                results.append(self._fake_distill(path, text))
            else:
                results.append(self._distill(path, text))
        return results

    def _fake_distill(self, path: str, text: str) -> List[Dict[str, Any]]:
        filename = Path(path).name if path else "unknown"
        return [{
            "text": f"Dry-run memory from {filename}",
            "category": "fact",
            "scope": "global",
            "importance": 0.5,
            "metadata": {"source": "dry-run", "path": path}
        }]

    def _distill(self, path: str, text: str) -> List[Dict[str, Any]]:
        try:
            from litellm import completion
            print(f"[DEBUG] Calling LiteLLM for {path} with model {self.model}...")
            
            # LiteLLM handles various providers. 
            # If using OpenAI compatible endpoints (like Doubao/Volcengine), 
            # we might need to adjust model name or base_url.
            # For standard OpenAI compatible:
            response = completion(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.prompt},
                    {"role": "user", "content": text},
                ],
                api_key=self.api_key,
                base_url=self.base_url,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                custom_llm_provider="openai" # Force OpenAI protocol
            )
            content = response.choices[0].message.content or ""
            print(f"[DEBUG] LLM Response for {path}: {content[:200]}...") # Print first 200 chars
            return self._parse_json(content, path)
        except Exception as exc:
            logger.error(f"Distill failed for {path}: {exc}")
            return []

    def _parse_json(self, content: str, path: str) -> List[Dict[str, Any]]:
        content = content.strip()
        if content.startswith("```"):
            lines = content.splitlines()
            if len(lines) >= 3:
                content = "\n".join(lines[1:-1]).strip()
        
        try:
            parsed = json.loads(content)
            if isinstance(parsed, list):
                # Normalize
                normalized = []
                for item in parsed:
                    if isinstance(item, dict):
                        # Ensure basic fields
                        if "metadata" not in item:
                            item["metadata"] = {}
                        item["metadata"]["source_path"] = path
                        normalized.append(item)
                return normalized
            return []
        except Exception:
            return []


@daft.udf(return_dtype=DataType.python())
class SearchUDF:
    def __init__(
        self,
        embed_config_json: str,
        store_config_json: str,
        top_k: int,
        min_score: float,
        mode: str = "vector"
    ):
        self.top_k = top_k
        self.min_score = min_score
        self.mode = mode
        
        import json
        
        # Deserialize configs
        self.embed_config_data = json.loads(embed_config_json)
        self.store_config_data = json.loads(store_config_json)
        
        self.embedder = None
        self.store = None
        
        try:
            from ..adapters.embedder import OpenAICompatibleConfig, OpenAICompatibleEmbedder, RandomEmbedder
            from ..adapters.vector_store import create_vector_store
            
            # Init Embedder
            if self.embed_config_data.get("dry_run"):
                self.embedder = RandomEmbedder(dimensions=self.embed_config_data.get("dimensions", 1024))
            else:
                embed_cfg = OpenAICompatibleConfig(**self.embed_config_data)
                self.embedder = OpenAICompatibleEmbedder(embed_cfg)
            
            print(f"[DEBUG] SearchUDF initialized embedder type: {type(self.embedder)}")
            
            # Init Store
            store_type = self.store_config_data.get("store_type", "lancedb")
            db_path = self.store_config_data.get("db_path")
            table_name = self.store_config_data.get("table_name")
            vector_dim = self.store_config_data.get("vector_dim")
            
            self.store = create_vector_store(
                store_type=store_type,
                db_path=db_path,
                table_name=table_name,
                vector_dim=vector_dim
            )
            
        except Exception as e:
            logger.error(f"Failed to initialize SearchUDF: {e}")

    def __call__(self, payload_col):
        results = []
        for payload in payload_col:
            if not isinstance(payload, dict):
                results.append([])
                continue
            
            query = payload.get("query", "").strip()
            if not query:
                results.append([])
                continue
                
            if not self.embedder or not self.store:
                results.append([])
                continue
                
            try:
                # 1. Embed
                # Note: embedder.embed_query returns List[float] or numpy array
                vector = self.embedder.embed_query(query)
                
                # 2. Search
                # vector_search should return List[StoredMemoryEntry]
                raw_results = self.store.vector_search(
                    vector,
                    top_k=self.top_k,
                    min_score=self.min_score
                )
                
                # 3. Normalize
                normalized = []
                for r in raw_results:
                    # StoredMemoryEntry has attributes, not dict keys.
                    # Verify if r is a dict or object. 
                    # If vector_search returns dicts (e.g. from lancedb directly), handle it.
                    if isinstance(r, dict):
                        item = {
                            "id": r.get("id"),
                            "text": r.get("text"),
                            "category": r.get("category"),
                            "scope": r.get("scope"),
                            "importance": r.get("importance"),
                            "timestamp": r.get("timestamp"), # Might need isoformat if it's datetime
                            "metadata": r.get("metadata"),
                            "score": r.get("score"),
                        }
                    else:
                        item = {
                            "id": r.id,
                            "text": r.text,
                            "category": r.category,
                            "scope": r.scope,
                            "importance": r.importance,
                            "timestamp": r.timestamp.isoformat() if r.timestamp else None,
                            "metadata": r.metadata,
                            "score": r.score,
                        }
                    normalized.append(item)
                
                results.append(normalized)
            except Exception as e:
                logger.error(f"Search failed for query '{query}': {e}")
                results.append([])
        return results


@daft.udf(return_dtype=DataType.string())
class QAUDF:
    def __init__(self, api_key: str, base_url: str, model: str, prompt: str, temperature: float, max_tokens: int, dry_run: bool):
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.prompt = prompt
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.dry_run = dry_run
        
        if not self.dry_run:
            pass

    def __call__(self, question_col, context_col):
        results = []
        for q, ctx in zip(question_col, context_col):
            if not q:
                results.append("")
                continue
            
            if self.dry_run:
                results.append(f"Dry-run answer to: {q}")
                continue
                
            try:
                from litellm import completion
                user_content = f"Question: {q}\n\nRelated Memories:\n{ctx}"
                
                # For custom OpenAI-compatible providers like Doubao, we need to specify custom_llm_provider="openai"
                # or prefix the model name with "openai/" if the base_url is set.
                # Since we have base_url, we can treat it as generic OpenAI compatible.
                
                resp = completion(
                    model=self.model, # litellm might expect "openai/doubao..." if we want to force openai client usage
                    messages=[
                        {"role": "system", "content": self.prompt},
                        {"role": "user", "content": user_content}
                    ],
                    api_key=self.api_key,
                    base_url=self.base_url,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    custom_llm_provider="openai" # Force OpenAI protocol
                )
                results.append(resp.choices[0].message.content or "")
            except Exception as e:
                logger.error(f"QA failed: {e}")
                results.append("Error generating answer")
        return results


@daft.udf(return_dtype=DataType.python())
class JudgeUDF:
    def __init__(self, api_key: str, base_url: str, model: str, system_prompt: str, user_template: str, dry_run: bool):
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.system_prompt = system_prompt
        self.user_template = user_template
        self.dry_run = dry_run
        
        if not self.dry_run:
            pass

    def __call__(self, question_col, answer_col, ground_truth_col):
        results = []
        for q, ans, gt in zip(question_col, answer_col, ground_truth_col):
            if self.dry_run:
                results.append({"score": 5, "reasoning": "Dry-run pass", "label": "CORRECT"})
                continue
            
            try:
                from litellm import completion
                user_content = self.user_template.format(
                    question=q,
                    answer=ans,
                    ground_truth=gt
                )
                resp = completion(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": user_content}
                    ],
                    api_key=self.api_key,
                    base_url=self.base_url,
                    temperature=0.0, # Judge should be deterministic
                    max_tokens=1024,
                    custom_llm_provider="openai" # Force OpenAI protocol
                )
                content = resp.choices[0].message.content or ""
                
                # Simple parsing for now - assumes JSON or clear structure
                # In EverMemOS judge prompt, it outputs JSON.
                try:
                    parsed = self._parse_json(content)
                    results.append(parsed)
                except:
                    results.append({"score": 0, "reasoning": "Failed to parse judge output", "raw": content})
                    
            except Exception as e:
                logger.error(f"Judge failed: {e}")
                results.append({"score": 0, "reasoning": str(e)})
        return results

    def _parse_json(self, content: str) -> Dict[str, Any]:
        import json
        content = content.strip()
        if content.startswith("```"):
            lines = content.splitlines()
            if len(lines) >= 3:
                content = "\n".join(lines[1:-1]).strip()
        return json.loads(content)
