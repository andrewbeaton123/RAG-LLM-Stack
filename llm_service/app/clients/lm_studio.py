

from langchain_core.language_models.lms import LLM
from openai import OpenAI


class LMStudioLLM(LLM):
    """
    Custom LLM for interacting with an LM Studio instance.
    """
    model_name: str = "local-model"
    base_url: str

    def __init__(self, **data):
        super().__init__(**data)
        # We initialize the OpenAI client here, using the provided base_url
        self.client = OpenAI(base_url=self.base_url, api_key="not-needed")

    @property
    def _llm_type(self) -> str:
        return "lm_studio"

    def _call(self, prompt: str, stop=None, **kwargs) -> str:
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            **kwargs,
        )
        return response.choices[0].message.content