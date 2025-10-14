from typing import  Dict, List, Optional
from abc import ABC, abstractmethod  
from loguru import logger 

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
    

    def retrieve_content(self,
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
                logger.error("Base LLM interface has encountered a\
                             type error when trying to use the \
                             retrieve method in retrieve content")
                
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