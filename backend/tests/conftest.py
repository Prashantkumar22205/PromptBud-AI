"""
tests/conftest.py

Shared pytest fixtures and configuration.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture(scope="session")
def client() -> TestClient:
    """Return a FastAPI test client (no live server needed)."""
    return TestClient(app)
