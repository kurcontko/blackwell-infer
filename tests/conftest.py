"""Pytest configuration and shared fixtures"""

import tempfile
from pathlib import Path

import pytest


@pytest.fixture
def temp_dir():
    """Provide a temporary directory for tests"""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_tasks():
    """Provide sample tasks for testing"""
    return [
        {
            "id": "test-1",
            "prompt": "What is the capital of France?",
            "max_tokens": 50,
            "temperature": 0.0,
        },
        {
            "id": "test-2",
            "prompt": "Explain quantum computing",
            "max_tokens": 100,
            "temperature": 0.7,
        },
    ]


@pytest.fixture
def mock_api_url():
    """Provide a mock API URL for testing"""
    return "http://localhost:8000/v1/chat/completions"
