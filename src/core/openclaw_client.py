import json
import logging
import requests
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)

class OpenClawClient:
    """Client for interacting with OpenClaw Agent API."""
    
    def __init__(self, base_url: str, api_key: Optional[str] = None):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.session = requests.Session()
        if api_key:
            self.session.headers.update({"Authorization": f"Bearer {api_key}"})

    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        user_id: Optional[str] = None,
    ) -> str:
        """Send a chat completion request to OpenClaw."""
        
        # OpenClaw usually exposes an OpenAI-compatible /v1/chat/completions endpoint
        # or a specific agent endpoint.
        # Assuming standard OpenAI compatibility for now as per OpenClaw docs.
        url = f"{self.base_url}/v1/chat/completions"
        
        payload = {
            "messages": messages,
            "temperature": temperature,
            "stream": False
        }
        
        if model:
            payload["model"] = model
        if max_tokens:
            payload["max_tokens"] = max_tokens
        if user_id:
            payload["user"] = user_id  # Pass user_id for scope handling if supported

        try:
            response = self.session.post(url, json=payload, timeout=60)
            response.raise_for_status()
            data = response.json()
            
            # Extract content from OpenAI format
            return data["choices"][0]["message"]["content"]
        except Exception as e:
            logger.error(f"OpenClaw API call failed: {e}")
            if isinstance(e, requests.exceptions.HTTPError):
                logger.error(f"Response: {e.response.text}")
            raise

    def health_check(self) -> bool:
        try:
            resp = self.session.get(f"{self.base_url}/health", timeout=5)
            return resp.status_code == 200
        except Exception:
            return False
