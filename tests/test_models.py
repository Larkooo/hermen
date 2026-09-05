from pathlib import Path
from types import SimpleNamespace

from hermen.config import QueryModelCapabilities, QueryModelConfig
from hermen.models import OpenAICompatibleQueryModel


def test_remote_vision_sends_image_bytes_not_local_path(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("HERMEN_TEST_KEY", "test-placeholder")
    model = OpenAICompatibleQueryModel(
        QueryModelConfig(base_url="https://example.invalid/v1", api_key_env="HERMEN_TEST_KEY"),
        QueryModelCapabilities(vision=True),
    )
    calls = []
    def post(url, **kwargs):
        calls.append(kwargs["json"])
        return SimpleNamespace(
            raise_for_status=lambda: None,
            json=lambda: {"choices": [{"message": {"content": "a diagram"}}]},
        )
    monkeypatch.setattr(model._httpx, "post", post)
    path = tmp_path / "diagram.png"
    path.write_bytes(b"synthetic image bytes")
    assert model.describe_image(str(path)) == "a diagram"
    url = calls[0]["messages"][0]["content"][1]["image_url"]["url"]
    assert url.startswith("data:image/png;base64,")
    assert str(tmp_path) not in url
