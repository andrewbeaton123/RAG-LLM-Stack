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
    

    ## RAG 

    def set_retriever(self, 
                      retriever: Any,
                      default_k : int = 4 )  -> None:
        
        logger.debug("Setting retriever in lm studio interface")

        self._retriever = retriever
        self._retrieval_k = default_k
        
        logger.debug("Finished Setting retriever in lm studio interface")

    
    def _unpack_doc_text(self,
                         doc:Any):
        
        # handle strings and dict of strings 

        if doc is None: 
            return ""
        
        if isinstance(doc, str):
            return doc
        
        if isinstance(doc, dict):
            return doc.get("page_content") or doc.get("text") or doc.get("content") or str(doc)

        return str(doc)
    

    def retrieve_contect(self,
                         query : str , 
                         k : Optional[int] = None) -> str: 
        
        if self._retriever is None: 
            return ""
        
        # either use the  number of documents from 
        # what  was passed in or from the class variable
        k = k or self._retrieval_k

        if hasattr( self._retriever, "get_relevant_documents"):
            docs = self._retriever.get_relevant_documents(query, k=k)
        
        elif hasattr(self._retriever, "retrieve"):
            try:
                docs = self._retriever.retriever(query, k=k)
            except TypeError:
                # If there is an issue then try  and not 
                # use the number of docs as a variable 
        
        elif callable(self._retriever):
            docs = self._retriever(query, k )
        
        else:
            # if all else fails  just return an empty  string 
            return ""

        if docs is None:
            return ""

        # ensure that the docs are in a sequence and limit to k 
        docs = list(docs)[:k]

        pieces = [self._unpack_doc_text(t) for t in docs if self._unpack_doc_text(t)]

        return "\n\n---\n\n".join(pieces)

    #TODO write tests for the  retrieve context
    #TODO  interim search of extra contect 
    #TODO use content  and prompt together 