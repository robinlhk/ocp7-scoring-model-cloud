import pytest
import requests

@pytest.fixture
def base_url():
    return "http://127.0.0.1:5000"

def test_api_reachable(base_url):
    response = requests.get(base_url + "/health")  # Assuming there's a health endpoint
    assert response.status_code == 200
