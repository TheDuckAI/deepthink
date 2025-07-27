from deepthink.llm import LLM


class Agent:
    def __init__(self, llm: LLM):
        self.llm = llm
        self.calls = []

    async def call_llm(self, prompt: str):
        result = await self.llm(prompt)
        d = result.model_dump()
        d["prompt"] = prompt
        self.calls.append(d)
        return result

    async def run(self, prompt: str, *args, **kwargs):
        pass
