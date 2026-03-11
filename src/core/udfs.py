import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import daft
from daft import DataType

logger = logging.getLogger(__name__)


def _build_litellm_completion_kwargs(model: str, api_key: Optional[str], base_url: Optional[str]) -> Dict[str, Any]:
    """Build provider-aware kwargs for litellm.completion."""
    provider = "volcengine" if isinstance(model, str) and model.startswith("volcengine/") else "openai"

    effective_api_key = api_key or ""
    if not effective_api_key:
        if provider == "volcengine":
            effective_api_key = (
                os.getenv("VOLCENGINE_API_KEY")
                or os.getenv("ARK_API_KEY")
                or ""
            )
            if not effective_api_key:
                effective_api_key = os.getenv("OPENAI_API_KEY", "")
        else:
            effective_api_key = (
                os.getenv("OPENAI_API_KEY")
                or os.getenv("VOLCENGINE_API_KEY")
                or os.getenv("ARK_API_KEY")
                or ""
            )

    kwargs: Dict[str, Any] = {"model": model, "custom_llm_provider": provider}
    if effective_api_key:
        kwargs["api_key"] = effective_api_key

    if provider == "openai" and base_url:
        kwargs["base_url"] = base_url

    return kwargs

@daft.udf(return_dtype=DataType.string())
def read_file_udf(path_col):
    """Read text content from files."""
    results = []
    for path in path_col:
        if not path:
            results.append("")
            continue
        try:
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

# -----------------------------------------------------------------------------
# Extraction UDF (Async)
# -----------------------------------------------------------------------------

def _fake_distill(path: str, text: str, user_id: Optional[str] = None) -> List[Dict[str, Any]]:
    filename = Path(path).name if path else "unknown"
    scope = f"user:{user_id}" if user_id else "global"
    return [{
        "text": f"Dry-run memory from {filename}",
        "category": "fact",
        "scope": scope,
        "importance": 0.5,
        "metadata": {"source": "dry-run", "path": path}
    }]

def _parse_json(content: str, path: str) -> List[Dict[str, Any]]:
    content = content.strip()
    if content.startswith("```"):
        lines = content.splitlines()
        if len(lines) >= 3:
            content = "\n".join(lines[1:-1]).strip()
    
    try:
        parsed = json.loads(content)
        if isinstance(parsed, list):
            normalized = []
            for item in parsed:
                if isinstance(item, dict):
                    if "metadata" not in item:
                        item["metadata"] = {}
                    item["metadata"]["source_path"] = path
                    normalized.append(item)
            return normalized
        return []
    except Exception:
        return []

async def _distill_async(text: str, path: str, user_id: Optional[str], api_key: str, base_url: str, model: str, prompt: str, temperature: float, max_tokens: int) -> List[Dict[str, Any]]:
    try:
        from litellm import acompletion
        # print(f"[DEBUG] Calling LiteLLM (Async) for {path} with model {model}...")

        kwargs = _build_litellm_completion_kwargs(model, api_key, base_url)
        response = await acompletion(
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": text},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )
        content = response.choices[0].message.content or ""
        # print(f"[DEBUG] LLM Response for {path}: {content[:200]}...")
        parsed = _parse_json(content, path)
        
        if user_id:
            scope_str = f"user:{user_id}"
            for item in parsed:
                if isinstance(item, dict):
                    item["scope"] = scope_str
        return parsed
    except Exception as exc:
        logger.error(f"Distill failed for {path}: {exc}")
        return []

def make_extraction_udf(api_key: str, base_url: str, model: str, prompt: str, temperature: float, max_tokens: int, dry_run: bool):
    """Create a stateless async UDF for extraction."""
    
    @daft.func(return_dtype=DataType.python())
    async def extract_memories(text: str, path: str, user_id: str):
        if dry_run:
            return _fake_distill(path, text, user_id)
        return await _distill_async(text, path, user_id, api_key, base_url, model, prompt, temperature, max_tokens)
        
    return extract_memories

# -----------------------------------------------------------------------------
# Search UDF (Sync Class - holds heavy resources)
# -----------------------------------------------------------------------------

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
        self.embed_config_data = json.loads(embed_config_json)
        self.store_config_data = json.loads(store_config_json)
        
        self.embedder = None
        self.store = None
        
        try:
            from ..adapters.embedder import OpenAICompatibleConfig, OpenAICompatibleEmbedder, RandomEmbedder, LocalHuggingFaceConfig, LocalHuggingFaceEmbedder
            from ..adapters.vector_store import create_vector_store
            
            # Init Embedder
            if self.embed_config_data.get("dry_run"):
                self.embedder = RandomEmbedder(dimensions=self.embed_config_data.get("dimensions", 1024))
            elif "model_name_or_path" in self.embed_config_data:
                 cfg = LocalHuggingFaceConfig(**self.embed_config_data)
                 self.embedder = LocalHuggingFaceEmbedder(cfg)
            else:
                embed_cfg = OpenAICompatibleConfig(**self.embed_config_data)
                self.embedder = OpenAICompatibleEmbedder(embed_cfg)
            
            # Init Store
            store_type = self.store_config_data.get("store_type", "lancedb")
            db_path = self.store_config_data.get("db_path")
            table_name = self.store_config_data.get("table_name")
            vector_dim = self.store_config_data.get("vector_dim")
            schema_mode = self.store_config_data.get("schema_mode", "pro")
            
            self.store = create_vector_store(
                store_type=store_type,
                db_path=db_path,
                table_name=table_name,
                vector_dim=vector_dim,
                schema_mode=schema_mode
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
            user_id = payload.get("user_id")
            if not query:
                results.append([])
                continue
                
            if not self.embedder or not self.store:
                results.append([])
                continue
                
            try:
                vector = self.embedder.embed_query(query)
                raw_results = self.store.vector_search(
                    vector,
                    top_k=self.top_k,
                    min_score=self.min_score,
                    user_id=user_id
                )
                
                normalized = []
                for r in raw_results:
                    if isinstance(r, dict):
                        item = r.copy()
                    else:
                        item = {
                            "id": r.id,
                            "text": r.text,
                            "category": r.category,
                            "scope": r.scope,
                            "importance": r.importance,
                            "timestamp": r.timestamp,
                            "metadata": r.metadata,
                            "score": getattr(r, "score", 0.0),
                        }
                    normalized.append(item)
                
                results.append(normalized)
            except Exception as e:
                logger.error(f"Search failed for query '{query}': {e}")
                results.append([])
        return results

# -----------------------------------------------------------------------------
# QA UDF (Async)
# -----------------------------------------------------------------------------

async def _qa_async(q: str, ctx: str, api_key: str, base_url: str, model: str, prompt: str, temperature: float, max_tokens: int) -> str:
    try:
        from litellm import acompletion
        user_content = f"Question: {q}\n\nRelated Memories:\n{ctx}"
        kwargs = _build_litellm_completion_kwargs(model, api_key, base_url)
        # print(f"[DEBUG] QA Call (Async) - Model: {model}, Question: {q[:50]}...")
        resp = await acompletion(
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": user_content},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )
        content = resp.choices[0].message.content or ""
        return content
    except Exception as e:
        logger.error(f"QA failed: {e}")
        return "Error generating answer"

def make_qa_udf(api_key: str, base_url: str, model: str, prompt: str, temperature: float, max_tokens: int, dry_run: bool):
    """Create a stateless async UDF for QA."""
    
    @daft.func(return_dtype=DataType.string())
    async def qa(q: str, ctx: str):
        if dry_run:
            return f"Dry-run answer to: {q}"
        return await _qa_async(q, ctx, api_key, base_url, model, prompt, temperature, max_tokens)
    return qa

# -----------------------------------------------------------------------------
# Judge UDF (Async)
# -----------------------------------------------------------------------------

async def _judge_async(q: str, ans: str, gt: str, api_key: str, base_url: str, model: str, system_prompt: str, user_template: str) -> Dict[str, Any]:
    try:
        from litellm import acompletion
        user_content = user_template.format(question=q, answer=ans, ground_truth=gt)
        kwargs = _build_litellm_completion_kwargs(model, api_key, base_url)
        # print(f"[DEBUG] Judge Call (Async) - Model: {model}...")
        resp = await acompletion(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            temperature=0.0,
            max_tokens=1024,
            **kwargs,
        )
        content = resp.choices[0].message.content or ""
        
        try:
            # Clean markdown code blocks
            clean_content = content.strip()
            if clean_content.startswith("```"):
                lines = clean_content.splitlines()
                if len(lines) >= 3:
                    clean_content = "\n".join(lines[1:-1]).strip()
            
            parsed = json.loads(clean_content)
            if not isinstance(parsed, dict):
                parsed = {"score": 0.0, "reasoning": f"Invalid JSON: {content}", "label": "WRONG"}
            
            label = str(parsed.get("label", "")).upper()
            if label == "CORRECT":
                parsed["score"] = 1.0
            elif label == "WRONG":
                parsed["score"] = 0.0
            
            return parsed
        except Exception:
            return {"score": 0.0, "reasoning": f"JSON Decode Error: {content}", "label": "ERROR"}
            
    except Exception as e:
        logger.error(f"Judge failed: {e}")
        return {"score": 0.0, "reasoning": f"Error: {e}", "label": "ERROR"}

def make_judge_udf(api_key: str, base_url: str, model: str, system_prompt: str, user_template: str, dry_run: bool):
    """Create a stateless async UDF for Judge."""
    
    @daft.func(return_dtype=DataType.python())
    async def judge(q: str, ans: str, gt: str):
        if dry_run:
            return {"score": 1.0, "reasoning": "Dry-run pass", "label": "CORRECT"}
        return await _judge_async(q, ans, gt, api_key, base_url, model, system_prompt, user_template)
    return judge

# -----------------------------------------------------------------------------
# OpenClaw QA UDF (Async)
# -----------------------------------------------------------------------------

def _extract_openclaw_text(response_json: Dict[str, Any]) -> str:
    """Extract text from OpenClaw /v1/responses format."""
    try:
        for item in response_json.get("output", []):
            if item.get("type") == "message":
                for content in item.get("content", []):
                    if content.get("type") == "output_text":
                        return content.get("text", "")
        # Fallback
        for item in response_json.get("output", []):
            if "text" in item:
                return item["text"]
            for content in item.get("content", []):
                if isinstance(content, dict) and "text" in content:
                    return content["text"]
    except Exception:
        pass
    return str(response_json)  # Return raw JSON if extraction fails

async def _openclaw_qa_async(q: str, user_id: Optional[str], base_url: str, api_key: str, model: str, temperature: float, max_tokens: int) -> str:
    import httpx
    import asyncio
    
    # Ensure base_url does not end with slash
    base = base_url.rstrip("/")
    url = f"{base}/v1/responses"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    payload = {
        "model": model or "openclaw",
        "input": q,
        "stream": False,
    }
    if user_id:
        payload["user"] = user_id
    
    retries = 5
    delay = 2.0
    last_error = None
    
    for attempt in range(retries):
        try:
            async with httpx.AsyncClient(timeout=300.0) as client:
                resp = await client.post(url, json=payload, headers=headers)
                resp.raise_for_status()
                return _extract_openclaw_text(resp.json())
        except httpx.HTTPStatusError as e:
            last_error = e
            # Retry on server errors (5xx)
            if e.response.status_code >= 500:
                logger.warning(f"OpenClaw QA attempt {attempt+1}/{retries} failed with {e.response.status_code}. Retrying in {delay}s...")
                await asyncio.sleep(delay)
                delay *= 2.0
            elif e.response.status_code == 429: # Rate limit
                logger.warning(f"OpenClaw QA rate limited. Retrying in {delay}s...")
                await asyncio.sleep(delay)
                delay *= 2.0
            else:
                logger.error(f"OpenClaw QA failed with {e.response.status_code}: {e}")
                return f"Error: {e}"
        except (httpx.ConnectError, httpx.TimeoutException, httpx.ReadTimeout, httpx.RemoteProtocolError) as e:
            last_error = e
            logger.warning(f"OpenClaw QA attempt {attempt+1}/{retries} failed with connection error: {e}. Retrying in {delay}s...")
            await asyncio.sleep(delay)
            delay *= 2.0
        except Exception as e:
            logger.error(f"OpenClaw QA failed: {e}")
            return f"Error: {e}"
            
    logger.error(f"OpenClaw QA failed after {retries} attempts. Last error: {last_error}")
    return f"Error: {last_error}"

def make_openclaw_qa_udf(base_url: str, api_key: str, model: str, temperature: float, max_tokens: int, dry_run: bool):
    """Create a stateless async UDF for OpenClaw QA."""
    
    @daft.func(return_dtype=DataType.string())
    async def openclaw_qa(q: str, user_id: str):
        if dry_run:
            return f"Dry-run OpenClaw answer to: {q}"
        if not q:
            return ""
        uid = user_id if user_id else None
        return await _openclaw_qa_async(q, uid, base_url, api_key, model, temperature, max_tokens)
    return openclaw_qa

# -----------------------------------------------------------------------------
# OpenClaw Ingest UDF (Async)
# -----------------------------------------------------------------------------

async def _openclaw_ingest_async(text: str, path: str, user_id: Optional[str], base_url: str, api_key: str, model: str) -> str:
    try:
        from litellm import acompletion
        import hashlib
        
        uid = user_id
        if not uid:
            uid = f"user_{hashlib.md5(path.encode()).hexdigest()[:8]}"
            
        prompt = f"[remember what's said, keep existing memory]\n\n{text}"
        
        kwargs = {
            "model": model, 
            "api_key": api_key, 
            "base_url": base_url,
            "custom_llm_provider": "openai"
        }
        if uid:
            kwargs["user"] = uid
            
        resp = await acompletion(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            **kwargs
        )
        content = resp.choices[0].message.content or ""
        return f"Ingested via chat. Response: {content[:50]}..."
    except Exception as e:
        logger.error(f"OpenClaw ingest failed for {path}: {e}")
        return f"Error: {e}"

def make_openclaw_ingest_udf(base_url: str, api_key: str, model: str, dry_run: bool):
    """Create a stateless async UDF for OpenClaw Ingest."""
    
    @daft.func(return_dtype=DataType.string())
    async def openclaw_ingest(text: str, path: str, user_id: str):
        if dry_run:
            return f"Dry-run ingest: {path}"
        if not text:
            return "Skipped: empty text"
        uid = user_id if user_id else None
        return await _openclaw_ingest_async(text, path, uid, base_url, api_key, model)
    return openclaw_ingest
