"""The thin service that holds the API keys.

Deliberately stateless: it takes a conditioning image, calls fal, and returns
URLs. Nothing is stored here — the phone keeps everything, and backups go
straight from the device to object storage without passing through this.
"""

from __future__ import annotations

import base64
import binascii
import json
import os
import tempfile

import pathlib

import fal_client
import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


def _load_local_secrets() -> None:
    """Pick up ../secrets.env when running on a Mac.

    On Vercel the key comes from the environment instead, so this is a no-op
    there — nothing gitignored ever ships.
    """
    here = pathlib.Path(__file__).resolve().parent
    for path in (here / "secrets.env", here.parent / "secrets.env"):
        if path.exists():
            break
    else:
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
#
# `fal-ai/z-image/turbo/controlnet` was measured as a cheaper replacement on
# 2026-09-10 and rejected: it ignores `image_url` outright. Same seed, same
# prompt, a depth map and its mirror gave pixel-identical output, as did a
# photograph and `control_scale: 0`, and all of them matched plain
# `fal-ai/z-image/turbo`. Don't switch until fal wires that controlnet up.
ENDPOINTS = {
    "depth": ("fal-ai/flux-control-lora-depth", {"preprocess_depth": False}),
    "lines": ("fal-ai/flux-control-lora-canny", {"preprocess_canny": False}),
    # No hosted endpoint was verified to take a normal map on its own.
    # `sdxl-controlnet-union` accepts one but its billing is unpublished, so
    # normals go through the depth model until someone confirms that.
    "normal": ("fal-ai/flux-control-lora-depth", {"preprocess_depth": False}),
}

MAX_IMAGE_BYTES = 4_000_000     # Vercel caps request bodies at 4.5 MB

LLM_URL = "https://fal.run/openrouter/router/openai/v1/chat/completions"
LLM_MODEL = "google/gemini-2.5-flash-lite"

# Shown when the model gives us something we can't parse. Generic on purpose —
# an empty suggestion strip is worse than one that ignores the room.
FALLBACK_SUGGESTIONS = [
    "Scandinavian calm, pale oak floor, soft morning light",
    "Warm mid-century palette, walnut furniture, brass accents",
    "Bright minimal white, one bold plant, uncluttered surfaces",
    "Japandi restraint, low wood furniture, paper-diffused light",
    "Industrial loft, exposed brick, black steel, worn leather",
    "Coastal linen and driftwood, chalk walls, open shutters",
    "Deep green walls, layered rugs, soft lamplight",
    "Bohemian layers, rattan, terracotta, hanging greenery",
]


class GenerateRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=2000)
    image: str = Field(description="base64-encoded PNG of the conditioning render")
    strength: float = Field(default=1.0, ge=0.0, le=2.0)
    conditioning: str = Field(default="depth")
    steps: int = Field(default=28, ge=1, le=50)
    count: int = Field(default=1, ge=1, le=4)


class GenerateResponse(BaseModel):
    images: list[str]
    endpoint: str


class RoomFacts(BaseModel):
    kind: str | None = Field(default=None, max_length=60)
    width: float | None = Field(default=None, gt=0, le=100)
    length: float | None = Field(default=None, gt=0, le=100)
    height: float | None = Field(default=None, gt=0, le=20)
    objects: list[str] = Field(default_factory=list, max_length=60)
    windows: int | None = Field(default=None, ge=0, le=50)
    doors: int | None = Field(default=None, ge=0, le=50)


class SuggestRequest(BaseModel):
    room: RoomFacts
    count: int = Field(default=5, ge=1, le=8)


class SuggestResponse(BaseModel):
    suggestions: list[str]


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
        "num_images": request.count,
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


def _room_summary(room: RoomFacts) -> str:
    parts = []
    if room.kind:
        parts.append(room.kind)
    if room.width and room.length:
        parts.append(f"{room.width:.1f}m by {room.length:.1f}m floor")
    if room.height:
        parts.append(f"{room.height:.1f}m ceiling")
    if room.windows is not None:
        parts.append(f"{room.windows} window(s)")
    if room.doors is not None:
        parts.append(f"{room.doors} door(s)")
    if room.objects:
        parts.append("already holds " + ", ".join(room.objects[:20]))
    return ", ".join(parts) or "a room of unrecorded size"


def _parse_suggestions(content: str) -> list[str]:
    text = content.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        parsed = json.loads(text)
    except ValueError:
        parsed = None

    if isinstance(parsed, dict):
        parsed = next((value for value in parsed.values() if isinstance(value, list)), None)
    if not isinstance(parsed, list):
        # Prose, then. A brief always has a comma in it and the apology above it
        # never does, which is enough to tell them apart.
        parsed = [line.lstrip("-*0123456789. ") for line in text.splitlines() if "," in line]

    cleaned = (line.strip(" \"'") for line in parsed if isinstance(line, str))
    return [line for line in cleaned if 3 <= len(line) <= 120]


@app.post("/suggest", response_model=SuggestResponse)
def suggest(request: SuggestRequest) -> SuggestResponse:
    key = os.environ.get("FAL_KEY")
    if not key:
        raise HTTPException(status_code=503, detail="FAL_KEY is not set on the server")

    instruction = (
        f"Write {request.count} one-line interior redesign briefs for this room: "
        f"{_room_summary(request.room)}.\n"
        "Voice, exactly: a style or mood, then two or three concrete details — a material, "
        "a colour, a piece of furniture, a quality of light — separated by commas.\n"
        'Example for a different room: "Scandinavian bedroom, oak floor, morning light"\n'
        "Rules: 8 to 14 words each. No sentences, no numbering, no trailing full stop. "
        "Name real materials and colours, never adjectives alone. Work with the room as "
        "measured and with what it already holds; suggest nothing its floor area will not "
        "take. No two briefs may share a style.\n"
        'Reply with JSON and nothing else: {"suggestions": ["...", "..."]}'
    )

    try:
        response = httpx.post(
            LLM_URL,
            headers={"Authorization": f"Key {key}"},
            json={
                "model": LLM_MODEL,
                "messages": [{"role": "user", "content": instruction}],
                "response_format": {"type": "json_object"},
                "max_tokens": 500,
            },
            timeout=30,
        )
        response.raise_for_status()
        # A model that reasons can spend its whole budget thinking and hand back
        # a null message, so this is not guaranteed to be a string.
        content = response.json()["choices"][0]["message"]["content"] or ""
    except (httpx.HTTPError, KeyError, IndexError, ValueError) as error:
        raise HTTPException(status_code=502, detail=f"suggestion model failed: {error}")

    # A generic strip of ideas beats an error in the one place the user is stuck
    # for words, so unparseable output falls back instead of failing.
    suggestions = _parse_suggestions(content) or FALLBACK_SUGGESTIONS
    return SuggestResponse(suggestions=suggestions[: request.count])
