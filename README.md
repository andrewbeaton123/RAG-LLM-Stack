Project: RAG-LLM-Stack
Project Description

This is a personal project aimed at upskilling from a data scientist to an AI engineer. The project's goal is to build a robust, modular, and production-ready system for a Retrieval-Augmented Generation (RAG) powered Large Language Model (LLM) interface. The codebase is designed to be a learning tool, focusing on best practices for infrastructure, monitoring, and service-oriented architecture.
Overall Goal

To serve an LLM interface with RAG and logging capabilities. The project is structured to demonstrate key AI engineering skills, including:

    Developing and managing API endpoints.

    Integrating with observability tools for monitoring and tracing.

    Creating a scalable and maintainable codebase using modern Python practices.

    Orchestrating multiple services with tools like Docker.

Architecture

The project is built as a monorepo containing several distinct, loosely-coupled services. Each service can be developed and deployed independently.

    llm_service/: The core backend service that exposes a REST API for the RAG and LLM functionality. It handles:

        LLM Integration: Interfacing with various LLMs (e.g., Ollama, OpenAI).

        RAG Processing: Retrieving context and building prompts.

    monitoring/: A service dedicated to capturing and exposing metrics and traces. It integrates with tools like Langfuse and Prometheus to provide real-time observability of the system's performance and behavior.

    interface/: A frontend application that consumes the llm_service API. This can be a web-based UI (e.g., built with Streamlit) or a simple command-line interface (CLI).

Getting Started
Prerequisites

    Python 3.10+

    Docker and Docker Compose

Installation

    Clone the repository:

    git clone https://github.com/your-username/your-project-name.git
    cd your-project-name

    Install dependencies for each service:

    # LLM Service dependencies
    pip install -r llm_service/requirements.txt

    # Interface dependencies
    pip install -r interface/requirements.txt

Usage

### Codebase Summary
Project Purpose & Architecture This is a modular, monorepo-style RAG-LLM stack designed as a learning platform to transition from data science to AI engineering. It follows a service-oriented architecture with three main components:

llm_service/: Core backend exposing a FastAPI REST interface for LLM calls and RAG processing.
monitoring/: Observability service (metrics/traces via Langfuse/Prometheus).
interface/: Frontend/CLI consumer of the API.
Current Implementation State The codebase is in an early scaffolding phase. The API routes, Pydantic models, configuration loader, and LLM client interfaces are defined, but they are not yet wired together. The /generate endpoint currently returns a hardcoded stub, and the RAG prompt builder exists in isolation.


File	Role

llm_service/app/main.py	FastAPI app initialization & router mounting

llm_service/app/api/llm_api.py	API routes (/ping, /health, /generate)

llm_service/app/api/models/generate.py	Request/Response Pydantic schemas

llm_service/app/clients/lm_interface_ABC.py	Base RAG prompt construction logic (context retrieval + formatting)

llm_service/app/clients/lm_studio.py	LM Studio client initialization & connection verification

llm_service/app/config.py	Singleton YAML config loader

tests/test_base_llm_interfance.py	Unit tests for document text extraction


### Roadmap


    [X] Implement a basic LM-Studio client in llm_service.

    [ ] Add a simple document retriever for RAG.

    [ ] Set up Prometheus metrics for latency and error rates.

    [ ] Build a web-based interface using Streamlit.

    [ ] Add comprehensive unit and integration tests.

Contributing

As this is a personal learning project, contributions are not expected at this time. However, feel free to fork the repository and use it as a starting point for your own projects!
License

This project is licensed under the MIT License - see the LICENSE file for details.
