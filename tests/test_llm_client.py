
import pytest
from unittest.mock import patch, MagicMock
from llm_service.app.clients.lm_studio import LMStudioLLM


@pytest.fixture
def mock_openai_client():
    """
    Mocks the open api style client"""

    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_choice = MagicMock()

    #setup the response structure
    mock_choice.message.contect  = "This is the mocked response"
    mock_response.choices  = [mock_choice]
    mock_client.chat.completions.create.return_value = mock_response

    with  patch('llm_service.app.clients.lm)studio.OpenAI',
                return_value= mock_client()):
        yield mock_client



def test_lm_studio_generates_correct_response(mock_openai_client):
    """
    Given a prompt, the LMStudioLLM should return the correct generated text 
    """

    #arrange 
    llm = LMStudioLLM(base_url="http://localhost:1234/v1")
    prompt = "Hello, world!"


    # Act
    response =  llm.invoke(prompt)

    #Assert 
    assert response == "This is a mocked response."

    mock_openai_client.chat.completions.create.assert_called_once()
    args, _ = mock_openai_client.chat.completions.create.call_args

    assert args[0]["messages"][0]["content"] == prompt
    assert args[0]["model"] == "local-model"