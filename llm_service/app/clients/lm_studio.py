import requests
import os 

from loguru  import logger
from  langchain.llms.base import LLM
from langchain.callbacks.manager import CallbackManagerForLLMRun
from typing import Optional,Dict,List, Any
from pydantic import Field

from .providers import LLMProvider
from .lm_interface_ABC import BaseLLMInterface





class LMStudioLLM(LLM,  BaseLLMInterface): 

    # langchain expects  a provider variable as part of its pydantic model
    provider: LLMProvider = Field(default=LLMProvider.LM_STUDIO)
    config: Dict[str, Any] = Field(default_factory=dict)
    base_url: str = Field(default="localhost")
    model: str = Field(default="deepseek-r1-distill-qwen-14b")
    temperature: float = Field(default=0.5)
    max_tokens: int = Field(default=1000)
    prompt: str = Field(default='')
    
    _retriever: Any = None  # set via set_retriever()
    _retrieval_k: int = 4


    def __init__(
        
            self,
            base_url:str = "http://localhost:1234/v4",
            model : str = "local-model",
            temperature : float = 0.5,
            max_tokens : int = 1000,
            **kwargs):
        #the pydantic model of the super class,  LLM needs to be started
        # Initialize LLM (Pydantic model) first
        super().__init__(**kwargs)

        # The base interface must  be initialized
        BaseLLMInterface.__init__(self, LLMProvider.LM_STUDIO, **kwargs)

        self.base_url = base_url
        self.model = model 
        self.temperature  = temperature
        self.max_tokens = max_tokens

        self.provider = LLMProvider.LM_STUDIO
        
        if os.environ.get("TESTING") != "TRUE":  # Skip during testing
            self.verify_connection()
        

    def _llm_type(self) -> str:
        return "lm_studio"
    

    def verify_connection(self):
        try :
            response = requests.get(f"{self.base_url}/models", timeout = 5)
            response.raise_for_status()
            logger.info(f"Established connection  for LM Studio at  {self.base_url}/models")
        
        except requests.RequestException as e: 
            logger.error(f"Failed to establish LM Studio connection : {e}")
        
    
    def _call (self,
               prompt: str,
               stop : Optional [CallbackManagerForLLMRun] = None,
               **kwargs : Any ) -> str: 


        payload = {
            "model": self.model,
            "prompt": prompt,
            "temperature": kwargs.get("temperature", self.temperature),
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
            "stop" : stop or []
        }

        try: 
            response = requests.post(
                f"{self.base_url}/v1/completions",
                json = payload,
                headers= {"Content-Type": "application/json"},
                timeout = 30 
            )

            response.raise_for_status()
            
            result = response.json()

            return result["choices"][0]["text"].strip()
        
        except requests.exceptions.RequestException as e : 
            logger.error(f"LM studio api error {e}")
            raise Exception(f" LM studio api error {e}")
        
    def generate(self,  prompt: str, **kwargs) -> str:
        """generate text from a prompt"""

        return self._call(prompt, **kwargs)

    def chat(self, messages: List[Dict[str, str]],  **kwargs) -> str: 

        prompt_parts  = []

        for msg in messages:
            role = msg.get("role",  "user")
            content  =  msg.get("content", "")

            if  role == "system": 
                prompt_parts.append(f"System: {content}")
            elif role == "user":
                prompt_parts.append(f"Human: {content}")
            elif role == "assistant":
                prompt_parts.append(f"Assistant: {content}")
        
        prompt_parts.append("Assistant:")
        self.prompt = "\n\n".join(prompt_parts)

        return self.generate(self.prompt, **kwargs)
    

    