from .agent import Agent
from .registry import register


@register("imo_deepthink")
class IMODeepthink(Agent):
    """
    Based on https://arxiv.org/abs/2507.15855
    """

    async def run(self, prompt, *args, **kwargs):
        raise NotImplementedError()
