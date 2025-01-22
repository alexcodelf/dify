import requests
from typing import Any, Union

from core.tools.entities.tool_entities import ToolInvokeMessage
from core.tools.tool.builtin_tool import BuiltinTool


class StableDiffusionTool(BuiltinTool):
    def _invoke(self, user_id: str, tool_parameters: dict[str, Any]) -> Union[ToolInvokeMessage, list[ToolInvokeMessage]]:
        headers = {
            "Authorization": f"Bearer {self.runtime.credentials['api_key']}",
        }
        payload = {
            "cfg_scale": tool_parameters.get("cfg_scale", 4.5),
            "model": tool_parameters.get("model"),
            "prompt": tool_parameters.get("prompt"),
            "n": tool_parameters.get("n", 1),
            "size": tool_parameters.get("size", "512x512"),
            "sample_steps": tool_parameters.get("sample_steps", 20),
            "sampler": "euler",
            "schedule": tool_parameters.get("schedule", "discrete"),
            "seed": tool_parameters.get("seed"),
        }
        response = requests.post(f"{self.runtime.credentials['base_url']}/v1-openai/images/generations", headers=headers, json=payload)
        if response.status_code != 200:
            return self.create_text_message(f"Got Error Response:{response.text}")
        res = response.json()

        result = [self.create_json_message(res)]
        for image_data in res.get("data", []):
            if image_data.get("b64_json"):
                result.append(
                    self.create_blob_message(
                        blob=image_data["b64_json"],
                        meta={"mime_type": "image/png"},
                        save_as=self.VariableKey.IMAGE.value
                    )
                )

        return result
