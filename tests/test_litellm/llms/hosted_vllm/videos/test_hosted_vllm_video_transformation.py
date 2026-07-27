import pytest

from litellm.llms.hosted_vllm.videos.transformation import HostedVLLMVideoConfig
from litellm.types.router import GenericLiteLLMParams
from litellm.utils import ProviderConfigManager
from litellm.types.utils import LlmProviders


def test_get_complete_url_uses_api_base():
    config = HostedVLLMVideoConfig()
    url = config.get_complete_url(
        model="vllm-omni",
        api_base="http://localhost:8000/v1",
        litellm_params={},
    )
    assert url == "http://localhost:8000/v1/videos"


def test_get_complete_url_strips_trailing_slash():
    config = HostedVLLMVideoConfig()
    url = config.get_complete_url(
        model="vllm-omni",
        api_base="http://localhost:8000/v1/",
        litellm_params={},
    )
    assert url == "http://localhost:8000/v1/videos"


def test_get_complete_url_raises_when_no_api_base(monkeypatch):
    monkeypatch.delenv("HOSTED_VLLM_API_BASE", raising=False)
    config = HostedVLLMVideoConfig()
    with pytest.raises(ValueError, match="api_base is required"):
        config.get_complete_url(model="vllm-omni", api_base=None, litellm_params={})


def test_validate_environment_no_api_key_sets_fake_key():
    config = HostedVLLMVideoConfig()
    headers = config.validate_environment(
        headers={},
        model="vllm-omni",
        api_key=None,
        litellm_params=GenericLiteLLMParams(),
    )
    assert "Authorization" not in headers
    assert headers.get("Content-Type") == "application/json"


def test_validate_environment_with_api_key_sets_auth_header():
    config = HostedVLLMVideoConfig()
    headers = config.validate_environment(
        headers={},
        model="vllm-omni",
        api_key="my-secret-key",
        litellm_params=GenericLiteLLMParams(),
    )
    assert "Authorization" in headers


def test_validate_environment_user_headers_take_priority():
    config = HostedVLLMVideoConfig()
    headers = config.validate_environment(
        headers={"X-Custom": "value", "Content-Type": "text/plain"},
        model="vllm-omni",
        api_key=None,
        litellm_params=GenericLiteLLMParams(),
    )
    assert headers["X-Custom"] == "value"
    assert headers["Content-Type"] == "text/plain"


def test_provider_config_manager_returns_hosted_vllm_video_config():
    config = ProviderConfigManager.get_provider_video_config(
        model="vllm-omni",
        provider=LlmProviders.HOSTED_VLLM,
    )
    assert isinstance(config, HostedVLLMVideoConfig)
