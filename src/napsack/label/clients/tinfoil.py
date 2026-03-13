import base64
import json
import os
from pathlib import Path
from typing import Optional, Dict, Any, List

from tinfoil import TinfoilAI
from napsack.label.clients.client import VLMClient, CAPTION_SCHEMA


class TinfoilClient(VLMClient):
    """
    VLMClient backed by TinfoilAI (confidential inference in a TEE).

    Usage:
        client = TinfoilClient(
            model_name="meta-llama/Llama-3.2-11B-Vision-Instruct",
            api_key=os.environ["TINFOIL_API_KEY"],
        )

    Optional:
        enclave  – specific enclave address; if omitted, a router is fetched automatically
        repo     – GitHub repo for attestation (default: "tinfoilsh/confidential-model-router")
    """

    def __init__(
        self,
        model_name: str,
        api_key: Optional[str] = None,
        enclave: str = "",
        repo: str = "tinfoilsh/confidential-model-router",
        max_tokens: int = 8192,
    ):
        self.model_name = model_name
        self.max_tokens = max_tokens

        resolved_key = api_key or os.environ.get("TINFOIL_API_KEY", "tinfoil")
        self._client = TinfoilAI(enclave=enclave, repo=repo, api_key=resolved_key)

        print(f"[TinfoilClient] Model: {self.model_name}")
        print(f"[TinfoilClient] Enclave: {self._client.enclave}")

    # ------------------------------------------------------------------
    # File upload helpers (always inline base64 – no remote Files API)
    # ------------------------------------------------------------------

    def upload_file(self, path: str, session_id: str = None) -> Dict:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"File not found: {path}")
        with open(p, "rb") as f:
            b64_data = base64.b64encode(f.read()).decode("utf-8")
        is_video = p.suffix.lower() in {".mp4", ".mov", ".avi", ".mkv"}
        file_type = "video" if is_video else "image"
        mime_type = "video/mp4" if is_video else "image/jpeg"
        return {
            "path": str(p.absolute()),
            "type": file_type,
            "data_url": f"data:{mime_type};base64,{b64_data}",
        }

    def upload_images(self, paths: List[str], session_id: str = None) -> Dict:
        frames = []
        for i, path in enumerate(paths):
            p = Path(path)
            mime_type = "image/png" if p.suffix.lower() == ".png" else "image/jpeg"
            with open(p, "rb") as f:
                b64_data = base64.b64encode(f.read()).decode("utf-8")
            frames.append({"data_url": f"data:{mime_type};base64,{b64_data}", "label": f"Frame {i + 1}"})
        return {"type": "image_list", "frames": frames}

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def generate(
        self,
        prompt: str,
        file_descriptor: Optional[Any] = None,
        schema: Optional[Dict] = None,
    ) -> Any:
        if schema is None:
            schema = CAPTION_SCHEMA

        messages = self._build_messages(prompt, file_descriptor)

        completion = self._client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            max_tokens=self.max_tokens,
            response_format={"type": "json_object"},
        )
        return json.loads(completion.choices[0].message.content)

    def _build_messages(self, prompt: str, file_desc: Optional[Any]) -> list:
        if not file_desc:
            return [{"role": "user", "content": prompt}]

        # Image-list mode: interleave frame labels with base64 image blocks
        if isinstance(file_desc, dict) and file_desc.get("type") == "image_list":
            content = []
            for frame in file_desc["frames"]:
                content.append({"type": "text", "text": frame["label"]})
                content.append({"type": "image_url", "image_url": {"url": frame["data_url"]}})
            content.append({"type": "text", "text": prompt})
            return [{"role": "user", "content": content}]

        # Single file (video or image)
        file_type = file_desc.get("type", "image")
        data_url = file_desc.get("data_url")
        if not data_url:
            file_path = file_desc.get("path")
            with open(file_path, "rb") as f:
                b64_data = base64.b64encode(f.read()).decode("utf-8")
            mime_type = "video/mp4" if file_type == "video" else "image/jpeg"
            data_url = f"data:{mime_type};base64,{b64_data}"

        if file_type == "video":
            content = [
                {"type": "video_url", "video_url": {"url": data_url}},
                {"type": "text", "text": prompt},
            ]
        else:
            content = [
                {"type": "image_url", "image_url": {"url": data_url}},
                {"type": "text", "text": prompt},
            ]

        return [{"role": "user", "content": content}]
