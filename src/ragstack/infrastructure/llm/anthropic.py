from __future__ import annotations

from ragstack.domain.ports.llm import ChatMessage, Completion


class AnthropicLLM:
    def __init__(self, api_key: str, model: str = "claude-sonnet-5-5") -> None:
        from anthropic import AsyncAnthropic

        self._client = AsyncAnthropic(api_key=api_key)
        self.model = model
        self.name = f"anthropic:{model}"

    async def complete(
        self, messages: list[ChatMessage], max_tokens: int = 1024, temperature: float = 0.0
    ) -> Completion:
        system = "\n".join(m.content for m in messages if m.role == "system")
        turns = [{"role": m.role, "content": m.content} for m in messages if m.role != "system"]
        if system:
            msg = await self._client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                system=system,
                messages=turns,  # type: ignore[arg-type]
            )
        else:
            msg = await self._client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                messages=turns,  # type: ignore[arg-type]
            )
        text = "".join(getattr(b, "text", "") for b in msg.content)
        return Completion(
            text=text,
            input_tokens=msg.usage.input_tokens,
            output_tokens=msg.usage.output_tokens,
            model=msg.model,
        )
