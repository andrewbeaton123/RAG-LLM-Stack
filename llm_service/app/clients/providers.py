from enum import Enum

class LLMProvider(Enum):
    LM_STUDIO = "lm_studio"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GEMINI  = "gemini"