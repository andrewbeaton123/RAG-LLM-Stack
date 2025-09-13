
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


import pytest

from langchain.schema import LLMResult
from unittest.mock import patch, MagicMock
from llm_service.app.clients import  LMStudioLLM

@pytest.fixture(autouse=True)
def set_test_environment():
    # this is used to make sure that the connection is 
    #not verified for testing as its being mocked
    os.environ["TESTING"] = "TRUE"
    yield
    os.environ["TESTING"] = "FALSE"


@pytest.fixture
def mock_requests_post():
    with patch("requests.post") as mock_post:
        mock_response = MagicMock()  # Create a mock response object
        mock_response.status_code = 200  # Set the status code
        mock_response.json.return_value = {"choices": [{"text": "Mocked response"}]}
        mock_post.return_value = mock_response  # Set the return value of requests.post
        yield mock_post



@pytest.fixture
def mock_llm_result():
    mock_llm_result = MagicMock(spec=LLMResult)
    mock_llm_result.generations = [[("This is a mocked response.",)]]
    return mock_llm_result



def test_lm_studio_generates_correct_response(mock_llm_result):
    with patch("llm_service.app.clients.lm_studio.LMStudioLLM.generate", return_value=mock_llm_result) as mock_generate:
        llm = LMStudioLLM(base_url="http://mocked_url:1234/v1")
        prompt = "Hello, world!"
        response = llm.generate(prompt)
        assert response.generations[0][0][0] == "This is a mocked response."



def test_lm_studio_call_temperature(mock_requests_post):
    llm = LMStudioLLM(base_url="http://mocked_url:1234/v1")
    llm._call("test prompt", temperature = 0.7)
    args, kwargs  = mock_requests_post.call_args

    assert kwargs["json"]["temperature"] == 0.7

def test_lm_studio_call_max_tokens(mock_requests_post):
    llm = LMStudioLLM(base_url="http://mocked_url:1234/v1")
    llm._call("test prompt", max_tokens = 200)
    args, kwargs  = mock_requests_post.call_args

    assert kwargs["json"]["max_tokens"] == 200


def  test_lm_studio_call_api_error(mock_requests_post):
    mock_requests_post.return_value.status_code = 500
    llm = LMStudioLLM(base_url = "http://mocked_url:1234/v1")
    
    import requests as _requests
    mock_requests_post.return_value.raise_for_status.side_effect = _requests.exceptions.HTTPError("500")

    with pytest.raises(Exception):
        llm._call("test prompt")
        

