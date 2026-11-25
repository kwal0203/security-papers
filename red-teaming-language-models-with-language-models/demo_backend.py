import httpx

from chatbot_runner import GenerationConfig, GenerationResult, Message


class DemoBackend:
    def __init__(self, client: httpx.Client | None) -> None:
        self.client = client

    def generate(
            self,
            messages: tuple[Message, ...],
            settings: GenerationConfig,
            *,
            seed: int
    ) -> GenerationResult:
        if messages[-1].content == "Can you give a shorter answer?":
            return GenerationResult("A queue is first in, first out.")
        return GenerationResult("A queue returns items in the order they arrive.")
