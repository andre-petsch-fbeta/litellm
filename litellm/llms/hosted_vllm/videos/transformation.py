from litellm.secret_managers.main import get_secret_str
from litellm.types.router import GenericLiteLLMParams
from litellm.types.videos.main import VideoCreateOptionalRequestParams

from ...openai.videos.transformation import OpenAIVideoConfig


class HostedVLLMVideoConfig(OpenAIVideoConfig):
    def get_supported_openai_params(self, model: str) -> list:
        return [
            "model",
            "prompt",
            "input_reference",
            "seconds",
            "size",
            "user",
            "extra_headers",
        ]

    def map_openai_params(
        self,
        video_create_optional_params: VideoCreateOptionalRequestParams,
        model: str,
        drop_params: bool,
    ) -> dict:
        return dict(video_create_optional_params)

    def validate_environment(
        self,
        headers: dict,
        model: str,
        api_key: str | None = None,
        litellm_params: GenericLiteLLMParams | None = None,
    ) -> dict:
        if api_key is None and litellm_params is not None:
            api_key = litellm_params.api_key
        if api_key is None:
            api_key = get_secret_str("HOSTED_VLLM_API_KEY") or "fake-api-key"

        default_headers: dict = {"Content-Type": "application/json"}
        if api_key and api_key != "fake-api-key":
            default_headers["Authorization"] = f"Bearer {api_key}"

        return {**default_headers, **headers}

    def get_complete_url(
        self,
        model: str,
        api_base: str | None,
        litellm_params: dict,
    ) -> str:
        if api_base is None:
            api_base = get_secret_str("HOSTED_VLLM_API_BASE")
        if api_base is None:
            raise ValueError("api_base is required for hosted_vllm video generation")

        return f"{api_base.rstrip('/')}/videos"
