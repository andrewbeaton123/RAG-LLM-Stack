from typing import  Dict, List
from abc import ABC, abstractmethod  

from .providers import LLMProvider

class BaseLLMInterface(ABC):
    def __init__(self,  provider:LLMProvider,  **kwargs):
        self.provider = provider
        self.config = kwargs
    
    
    @abstractmethod
    def generate(self, prompt: str , **kwargs)  -> str:
        """Generate a response  from a  text prompt"""

        pass

    @abstractmethod
    def chat(self, messages: List[Dict[str,str]], **kwargs) ->  str:
        """Chat interface with message history"""

        pass