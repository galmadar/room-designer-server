"""The thin service that holds the API keys.

Deliberately stateless: it takes a conditioning image, calls fal, and returns
URLs. Nothing is stored here — the phone keeps everything, and backups go
straight from the device to object storage without passing through this.
"""

from __future__ import annotations

import base64
import binascii
import os
import tempfile

import pathlib

import fal_client
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


def _load_local_secrets() -> None:
    """Pick up ../secrets.env when running on a Mac.

    On Vercel the key comes from the environment instead, so this is a no-op
    there — nothing gitignored ever ships.
    """
    path = pathlib.Path(__file__).resolve().parent.parent / "secrets.env"
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, _, value = line.partition("=")
        value = value.strip().strip('"').strip("'")
        if value:
            os.environ.setdefault(name.strip(), value)


_load_local_secrets()

app = FastAPI(title="Room Designer")

# Verified against fal's OpenAPI schema on 2026-09-09.
#
# `preprocess_depth: False` is the point of the whole arrangement: the phone
# renders a true depth buffer from the scan, so having fal re-estimate depth
# from that render would throw away the one advantage a scan gives us.
ENDPOINTS = {
    "depth": ("fal-ai/flux-control-lora-depth", {"preprocess_depth": False}),
    "lines": ("fal-ai/flux-control-lora-canny", {"preprocess_canny": False}),
    # No hosted endpoint was verified to take a normal map on its own.
    # `sdxl-controlnet-union` accepts one but its billing is unpublished, so
    # normals go through the depth model until someone confirms that.
    "normal": ("fal-ai/flux-control-lora-depth", {"preprocess_depth": False}),
}

MAX_IMAGE_BYTES = 4_000_000     # Vercel caps request bodies at 4.5 MB


class GenerateRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=2000)
    image: str = Field(description="base64-encoded PNG of the conditioning render")
    strength: float = Field(default=1.0, ge=0.0, le=2.0)
    conditioning: str = Field(default="depth")
    steps: int = Field(default=28, ge=1, le=50)


class GenerateResponse(BaseModel):
    images: list[str]
    endpoint: str


@app.get("/health")
def health() -> dict:
    return {"ok": True, "fal_key_configured": bool(os.environ.get("FAL_KEY"))}


@app.post("/generate", response_model=GenerateResponse)
def generate(request: GenerateRequest) -> GenerateResponse:
    if not os.environ.get("FAL_KEY"):
        raise HTTPException(status_code=503, detail="FAL_KEY is not set on the server")

    endpoint, options = ENDPOINTS.get(request.conditioning, ENDPOINTS["depth"])

    try:
        image_bytes = base64.b64decode(request.image, validate=True)
    except (binascii.Error, ValueError) as error:
        raise HTTPException(status_code=400, detail=f"image is not valid base64: {error}")

    if not image_bytes:
        raise HTTPException(status_code=400, detail="image is empty")
    if len(image_bytes) > MAX_IMAGE_BYTES:
        raise HTTPException(status_code=413, detail="conditioning image is too large")

    # fal takes a URL, so the render has to be uploaded first. /tmp is the only
    # writable path in a serverless function and is fine for something this size.
    with tempfile.NamedTemporaryFile(suffix=".png", delete=True) as handle:
        handle.write(image_bytes)
        handle.flush()
        try:
            image_url = fal_client.upload_file(handle.name)
        except Exception as error:                      # noqa: BLE001
            raise HTTPException(status_code=502, detail=f"upload to fal failed: {error}")

    arguments = {
        "prompt": request.prompt,
        "control_lora_image_url": image_url,
        "control_lora_strength": request.strength,
        "num_inference_steps": request.steps,
        "image_size": "square_hd",
        **options,
    }

    try:
        result = fal_client.subscribe(endpoint, arguments=arguments)
    except Exception as error:                          # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"generation failed: {error}")

    urls = [image["url"] for image in result.get("images", []) if "url" in image]
    if not urls:
        raise HTTPException(status_code=502, detail="fal returned no image")

    return GenerateResponse(images=urls, endpoint=endpoint)
