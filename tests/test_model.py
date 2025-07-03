from Model.YOLO import YOLO_Satellite_Detection
from Model import entities
import pytest


@pytest.mark.parametrize(
    "collect_id",
    ["detect-with-metadata-fc46157d-13b9-461b-ac71-7a528f419e43"],
    indirect=True,
)
@pytest.mark.anyio
async def test_inference(
    model: YOLO_Satellite_Detection, payload_entity: list[entities.FitsFile]
) -> None:
    detections = await model.inference(payload_entity)
    assert len(detections) == len(payload_entity)
