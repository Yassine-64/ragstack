from __future__ import annotations

from ragstack.domain.ports.llm import ChatMessage, Completion


class OpenAiLLM:
    def __init__(self, api_key: str, model: str = "gpt-4o-mini") -> None:
        from openai import AsyncOpenAI

        self._client = AsyncOpenAI(api_key=api_key)
        self.model = model
        self.name = f"openai:{model}"

    async def complete(
        self, messages: list[ChatMessage], max_tokens: int = 1024, temperature: float = 0.0
    ) -> Completion:
        resp = await self._client.chat.completions.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[{"role": m.role, "content": m.content} for m in messages],  # type: ignore[misc]
        )
        usage = resp.usage
        return Completion(
            text=resp.choices[0].message.content or "",
            model=resp.model,
            input_tokens=usage.prompt_tokens if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0,
        )
