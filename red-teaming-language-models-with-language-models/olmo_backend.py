import httpx

from chatbot_runner import BackendError, GenerationConfig, GenerationResult, Message
from typing import Any


class OlmoBackend:
    def __init__(self, client: httpx.Client | None) -> None:
        self.client = client

    def generate(
        self,
        messages: tuple[Message, ...],
        settings: GenerationConfig,
        *,
        seed: int,
    ) -> GenerationResult:
        payload: dict[str, Any] = {
            "model": settings.model_id,
            "messages": [
                {"role": message.role, "content": message.content}
                for message in messages
            ],
            "temperature": settings.temperature,
            "top_p": settings.top_p,
            "max_tokens": settings.max_new_tokens,
            "seed": seed,
            "stream": False,
        }
        if settings.stop:
            payload["stop"] = list(settings.stop)

        try:
            response = self.client.post("chat/completions", json=payload)
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code
            detail = exc.response.text[:500]
            raise BackendError(f"Chat server returned HTTP {status}: {detail}") from exc
        except httpx.RequestError as exc:
            raise BackendError(f"Chat request failed: {exc}") from exc

        try:
            body = response.json()
            choice = body["choices"][0]
            text = choice["message"]["content"]
            usage = body.get("usage") or {}
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise BackendError("Chat server returned an invalid completion") from exc

        if not isinstance(text, str):
            raise BackendError("Chat completion did not contain an assistant text reply")

        return GenerationResult(
            text=text,
            finish_reason=choice.get("finish_reason"),
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
        )
