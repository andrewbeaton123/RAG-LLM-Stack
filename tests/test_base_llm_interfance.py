import sys
import os 

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


import pytest 

from typing import Any
from unittest.mock import MagicMock
from llm_service.app.clients.lm_interface_ABC import BaseLLMInterface
from llm_service.app.clients.providers import LLMProvider



class TestLLM(BaseLLMInterface):
    def __init__(self):
        super().__init__(provider=LLMProvider.LM_STUDIO)

    def generate (self, prompt : str , **kwargs) -> str: 
        return "test"
    
    def chat(self, messages: list, **kwargs) -> str: 
        return "test"
    
@pytest.fixture
def base_llm():
    return TestLLM()



def test_set_retriever(base_llm):
    base_llm.set_retriever("test", 5)
    assert base_llm._retriever == "test"
    assert base_llm._retrieval_k == 5


def test__unpack_doc_text_None_input(base_llm):
    assert base_llm._unpack_doc_text(None) == ""


def test__unpack_doc_text_dict_input(base_llm):

   
    
    assert base_llm._unpack_doc_text({"text": "Test string"}) == "Test string"
    assert base_llm._unpack_doc_text({"page_content": "Test string"}) == "Test string"
    assert base_llm._unpack_doc_text({"content": "Test string"}) == "Test string"
    assert base_llm._unpack_doc_text({"Test string"}) == "{'Test string'}"
    

def test__retriever_content(base_llm):
    base_llm.set_retriever("test",5)
    assert base_llm._retriever == "test"
    assert base_llm._retrieval_k == 5

def test__unpack_doc_text_text_input(base_llm):
    assert  base_llm._unpack_doc_text("Test String") == "Test String"


def test__unpack_doc_text_none_input(base_llm):
    assert  base_llm._unpack_doc_text(None) == ""


def test__unpack_doc_text_dict_input(base_llm):
    assert base_llm._unpack_doc_text({"text":"Test String"}) == "Test String"
    assert base_llm._unpack_doc_text({"page_content":"Test String"}) == "Test String"
    assert base_llm._unpack_doc_text({"content": "Test String"}) == "Test String"
    assert base_llm._unpack_doc_text({"Test String"}) == "{'Test String'}"



def test__retrive_context_with_get_relevant_documents(base_llm):

    mock_retriever = MagicMock()
    mock_retriever.get_relevant_documents.return_value = ["doc1", "doc2"]

    base_llm.set_retriever(mock_retriever, 5)

    results =  base_llm.retrieve_context("test query")
    assert results == "doc1\n\n---\n\ndoc2"
    mock_retriever.get_relevant_documents.assert_called_once_with("test query", k=5)