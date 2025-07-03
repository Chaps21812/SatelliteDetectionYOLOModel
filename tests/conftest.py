import pydantic
import pytest
from Model import entities
from Model.YOLO import YOLO_Satellite_Detection
from importlib import resources
from .data import payloads, responses
import json
from typing import cast


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture
def model() -> YOLO_Satellite_Detection:
    yolo_model = YOLO_Satellite_Detection()
    yolo_model.load_local()
    return yolo_model


@pytest.fixture(params=[])
def collect_id(request: pytest.FixtureRequest) -> str:
    return cast(str, request.param)


@pytest.fixture
def payload(collect_id: str) -> list[dict]:
    return cast(
        list[dict], json.loads(resources.read_text(payloads, f"{collect_id}.json"))
    )


@pytest.fixture
def response(collect_id: str) -> list[dict]:
    return cast(
        list[dict], json.loads(resources.read_text(responses, f"{collect_id}.json"))
    )


@pytest.fixture
def payload_entity(payload: list[dict]) -> list[entities.FitsFile]:
    return pydantic.TypeAdapter(list[entities.FitsFile]).validate_python(payload)


@pytest.fixture
def response_entity(
    response: list[dict],
) -> list[entities.ObjectDetections]:
    return pydantic.TypeAdapter(list[entities.ObjectDetections]).validate_python(
        response
    )
