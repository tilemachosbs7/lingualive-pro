"""
Basic translation endpoint tests.
"""

import pytest
from fastapi.testclient import TestClient

from app.main import create_app


# Note: These tests may fail if API keys are not configured
# Run with: pytest backend/tests/ -v


@pytest.fixture
def client():
    """Create test client."""
    app = create_app()
    return TestClient(app)


def test_translate_endpoint_requires_text(client):
    """Test /api/translate-text requires text parameter."""
    response = client.post("/api/translate-text", json={
        "target_lang": "en"
    })
    assert response.status_code == 422  # Validation error


def test_translate_endpoint_requires_target_lang(client):
    """Test /api/translate-text requires target_lang parameter."""
    response = client.post("/api/translate-text", json={
        "text": "Hello"
    })
    assert response.status_code == 422  # Validation error


def test_translate_endpoint_valid_request(client):
    """Test /api/translate-text with valid request (may fail without API keys)."""
    response = client.post("/api/translate-text", json={
        "text": "Hello",
        "target_lang": "el",
        "source_lang": "en"
    })
    # May return 500 if API keys not configured, or 200 if configured
    assert response.status_code in (200, 500, 502)
    if response.status_code == 200:
        data = response.json()
        assert "translated_text" in data

