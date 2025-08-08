import enum


class LLMProvider(enum):
    LM_STUDIO = "lm_studio"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GEMINI  = "gemini"