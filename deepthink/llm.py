import logging

from openai import AsyncOpenAI
from openai.types.chat import ChatCompletion
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


class LLM:
    def __init__(self, model: str, base_url, api_key, extra_body=None):
        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
        )
        self.model = model
        self.extra_body = extra_body

    @retry(
        stop=stop_after_attempt(7),
        wait=wait_random_exponential(multiplier=1, min=1, max=120),
        retry=retry_if_exception_type(Exception),
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )
    async def __call__(self, prompt: str) -> ChatCompletion:
        result = await self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            extra_body=self.extra_body,
        )
        return result
