from deepthink.agent import Agent
from deepthink.registry import register


@register("zeroshot")
class ZeroShotAgent(Agent):
    async def run(self, prompt, *args, **kwargs):
        return await self.call_llm(prompt)
