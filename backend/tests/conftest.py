"""
tests/conftest.py
==================
Shared pytest fixtures and test configuration.
"""
import asyncio
import pytest


# Use asyncio event loop for async tests
@pytest.fixture(scope="session")
def event_loop():
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


# Minimal test settings
@pytest.fixture(autouse=True)
def mock_settings(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/researchgraph_test")
    monkeypatch.setenv("DEBUG", "true")
    monkeypatch.setenv("DEFAULT_EMBEDDING_MODEL", "miniml")
    monkeypatch.setenv("TORCH_DEVICE", "cpu")
