import pytest
from fastapi.testclient import TestClient

from app.dependencies import SERVICES
from app.main import app
from app.services.base import DEFAULT_DATA_PATH
from app.services.features import read_energy_csv


@pytest.fixture(scope="session")
def models_dir(tmp_path_factory):
    """Directory holding the models trained once (with calibration) per session."""
    return tmp_path_factory.mktemp("models")


@pytest.fixture(scope="session")
def client(models_dir):
    for service in SERVICES.values():
        service.models_dir = models_dir
        service.model_path = models_dir / service.model_filename
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture(scope="session")
def trained_services(client):
    """The API's service instances, trained and calibrated at startup."""
    del client
    return SERVICES


@pytest.fixture(scope="session")
def raw():
    return read_energy_csv(DEFAULT_DATA_PATH)
