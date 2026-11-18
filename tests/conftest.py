import pytest
from fastapi.testclient import TestClient

from app.dependencies import SERVICES
from app.main import app


@pytest.fixture(scope="session")
def client(tmp_path_factory):
    models_dir = tmp_path_factory.mktemp("models")
    for service in SERVICES.values():
        service.models_dir = models_dir
        service.model_path = models_dir / service.model_filename
    with TestClient(app) as test_client:
        yield test_client
