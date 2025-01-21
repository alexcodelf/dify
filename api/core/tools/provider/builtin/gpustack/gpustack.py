import requests

from core.tools.errors import ToolProviderCredentialValidationError
from core.tools.provider.builtin_tool_provider import BuiltinToolProviderController


class GPUStackProvider(BuiltinToolProviderController):
    def _validate_credentials(self, credentials: dict) -> None:
        base_url = credentials.get("base_url", "").removesuffix("/")
        api_key = credentials.get("api_key", "")
        model = credentials.get("model", "")
        base_url = base_url.rstrip("/").removesuffix("/v1-openai")

        if not base_url or not api_key or not model:
            raise ToolProviderCredentialValidationError("GPUStack base_url, api_key and model is required")
        headers = {
            "accept": "application/json",
            "authorization": f"Bearer {api_key}",
        }

        response = requests.get(f"{base_url}/v1-openai/models/{model}", headers=headers)
        if response.status_code != 200:
            raise ToolProviderCredentialValidationError(f"Failed to validate GPUStack API key, status code: {response.status_code}-{response.text}")
