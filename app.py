"""The thin service that holds the API keys.

Deliberately stateless: it takes a conditioning image, calls fal, and returns
URLs. Nothing is stored here — the phone keeps everything, and backups go
straight from the device to object storage without passing through this.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import html
import ipaddress
import json
import os
import re
import socket
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from html.parser import HTMLParser
from typing import Literal
from urllib.parse import urljoin, urlsplit, urlunsplit

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

# Verified against fal's OpenAPI schemas on 2026-09-11. Each model names and
# spells aspect ratio differently, hence the last two fields.
COMPOSE_MODELS = {
    "nano-banana-2": ("fal-ai/nano-banana-2/edit", {"resolution": "1K"}, "aspect_ratio", {"4:3": "4:3", "3:4": "3:4"}),
    "gpt-image-2": ("openai/gpt-image-2/edit", {"quality": "high"}, "image_size", {"4:3": "landscape_4_3", "3:4": "portrait_4_3"}),
}
ComposeModel = Literal[tuple(COMPOSE_MODELS)]

MARKERS = ("red", "blue", "yellow", "green", "purple", "orange", "cyan", "magenta")
Marker = Literal[MARKERS]

# What `fal_client.upload` returned on 2026-09-11. /compose forwards nothing
# else, so it can't be used to make fal fetch arbitrary URLs on our key.
FAL_STORAGE_HOSTS = {"v3b.fal.media"}

# Leaves room under Vercel's 300s cap to answer with an error instead of dying.
COMPOSE_TIMEOUT = 280

MAX_IMAGE_BYTES = 4_000_000     # Vercel caps request bodies at 4.5 MB

# /objects/import fetches pages the user names from a box that holds FAL_KEY.
MAX_HTML_BYTES = 3_000_000
MAX_FETCHED_IMAGE_BYTES = 8_000_000
MAX_IMPORTED_IMAGES = 6
MAX_REDIRECTS = 3
FETCH_TIMEOUT = httpx.Timeout(10.0)
FETCH_DEADLINE = 25             # a byte every 9s would otherwise never time out
BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 "
        "(KHTML, like Gecko) Version/17.5 Safari/605.1.15"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}
PAGE_ACCEPT = "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"
# No AVIF: neither image model takes it, and CDNs pick the format from this.
IMAGE_ACCEPT = "image/webp,image/png,image/jpeg;q=0.9"

# Shown to the user verbatim.
BAD_LINK = "That link doesn't look like a shop page. Copy the product page's address and try again."
NO_PICTURES = "Couldn't find product pictures on that page."
BLOCKED = "That site blocked the download. Save a screenshot of the product and add it from Photos instead."
SAVE_FAILED = "Couldn't save those pictures just now. Please try again in a minute."

# Only consulted when a page yielded no pictures, to tell a bot wall from a bare page.
BOT_WALL_MARKERS = ("captcha", "robot check", "are you a robot", "access denied", "cf-chl", "enable javascript and cookies")

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


class UploadRequest(BaseModel):
    image: str = Field(description="base64-encoded JPEG or PNG")
    content_type: Literal["image/jpeg", "image/png"]


class UploadResponse(BaseModel):
    url: str


class ImportRequest(BaseModel):
    url: str = Field(min_length=1, max_length=4000)


class ImportResponse(BaseModel):
    title: str | None
    images: list[str]
    source: str


class ComposeObject(BaseModel):
    name: str = Field(min_length=1)
    image_url: str
    marker: Marker | None = None     # paintings, curtains, pendants: nothing to box on the floor


class ComposeRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=4000)
    model: ComposeModel = "nano-banana-2"
    room_photo_url: str | None = None
    scan_url: str
    objects: list[ComposeObject] = Field(default_factory=list, max_length=8)
    aspect_ratio: Literal["4:3", "3:4"] = "4:3"


class ComposeResponse(BaseModel):
    images: list[str]
    model: str


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


def _sniff_image(data: bytes) -> str | None:
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    return None


def _store(data: bytes, content_type: str) -> str:
    # No fallback repository: its URLs land on a host /compose would then refuse.
    extension = {"image/jpeg": ".jpg", "image/png": ".png", "image/webp": ".webp"}[content_type]
    return fal_client.upload(
        data, content_type, file_name=f"upload{extension}", repository="fal_v3", fallback_repository=[],
    )


@app.post("/upload", response_model=UploadResponse)
def upload(request: UploadRequest) -> UploadResponse:
    if not os.environ.get("FAL_KEY"):
        raise HTTPException(status_code=503, detail="FAL_KEY is not set on the server")

    try:
        image_bytes = base64.b64decode(request.image, validate=True)
    except (binascii.Error, ValueError) as error:
        raise HTTPException(status_code=400, detail=f"image is not valid base64: {error}")

    if not image_bytes:
        raise HTTPException(status_code=400, detail="image is empty")
    if len(image_bytes) > MAX_IMAGE_BYTES:
        raise HTTPException(status_code=413, detail="image is too large")
    if _sniff_image(image_bytes) != request.content_type:
        raise HTTPException(status_code=400, detail=f"image is not a {request.content_type}")

    try:
        url = _store(image_bytes, request.content_type)
    except Exception as error:                          # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"upload to fal failed: {error}")

    return UploadResponse(url=url)


class _Refused(Exception):
    """The URL points somewhere this server won't fetch."""


class _Blocked(Exception):
    """The site answered, but not with what we asked for."""


def _check_public_url(url: str) -> None:
    parts = urlsplit(url)
    if parts.scheme not in ("http", "https") or not parts.hostname:
        raise _Refused(url)
    try:
        port = parts.port or (443 if parts.scheme == "https" else 80)
    except ValueError:
        raise _Refused(url)
    if port not in (80, 443):
        raise _Refused(url)

    try:
        addresses = socket.getaddrinfo(parts.hostname, port, proto=socket.IPPROTO_TCP)
    except (socket.gaierror, UnicodeError):
        raise _Refused(url)
    for *_, sockaddr in addresses:
        address = ipaddress.ip_address(sockaddr[0].split("%")[0])
        if isinstance(address, ipaddress.IPv6Address) and address.ipv4_mapped:
            address = address.ipv4_mapped
        if (
            address.is_private or address.is_loopback or address.is_link_local
            or address.is_reserved or address.is_multicast or address.is_unspecified
            or not address.is_global
        ):
            raise _Refused(url)


def _fetch(url: str, *, accept: str, limit: int, want: str, truncate: bool = False) -> tuple[bytes, str | None, str]:
    """GET a public URL, following redirects by hand so every hop is re-checked.

    Returns the body, its charset and the final URL.
    """
    deadline = time.monotonic() + FETCH_DEADLINE
    headers = {**BROWSER_HEADERS, "Accept": accept}
    with httpx.Client(timeout=FETCH_TIMEOUT, follow_redirects=False, trust_env=False, headers=headers) as client:
        for _ in range(MAX_REDIRECTS + 1):
            # Resolve-then-connect leaves a DNS-rebinding gap; acceptable at this scale.
            _check_public_url(url)
            try:
                with client.stream("GET", url) as response:
                    if response.is_redirect:
                        url = urljoin(url, response.headers["location"])
                        continue
                    if response.status_code in (404, 410):
                        raise LookupError(url)
                    if response.status_code != 200:
                        raise _Blocked(f"HTTP {response.status_code}")
                    if not response.headers.get("content-type", "").lower().startswith(want):
                        raise LookupError(url)

                    body = bytearray()
                    for chunk in response.iter_bytes():
                        body += chunk
                        if len(body) > limit:
                            if not truncate:
                                raise LookupError(url)
                            # The head, where og:image and JSON-LD live, is what matters.
                            del body[limit:]
                            break
                        if time.monotonic() > deadline:
                            raise _Blocked("too slow")
                    return bytes(body), response.charset_encoding, url
            except httpx.HTTPError as error:
                raise _Blocked(str(error)) from error
    raise _Blocked("too many redirects")


class _ProductPageParser(HTMLParser):
    """Collects the handful of tags that name a product and its pictures."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.json_ld: list[str] = []
        self.meta: dict[str, list[str]] = {}
        self.images: list[str] = []
        self.title = ""
        self._capture: str | None = None
        self._buffer: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = {name.lower(): value or "" for name, value in attrs}
        if tag == "script" and values.get("type", "").lower().startswith("application/ld+json"):
            self._capture, self._buffer = "json_ld", []
        elif tag == "title" and not self.title:
            self._capture, self._buffer = "title", []
        elif tag == "meta":
            key = (values.get("property") or values.get("name") or "").lower()
            if key and values.get("content"):
                self.meta.setdefault(key, []).append(values["content"])
        elif tag == "img":
            self._consider_img(values)

    def handle_data(self, data: str) -> None:
        if self._capture:
            self._buffer.append(data)

    def handle_endtag(self, tag: str) -> None:
        if self._capture == "json_ld" and tag == "script":
            self.json_ld.append("".join(self._buffer))
        elif self._capture == "title" and tag == "title":
            self.title = " ".join("".join(self._buffer).split())
        else:
            return
        self._capture = None

    def _consider_img(self, values: dict[str, str]) -> None:
        # Last resort only, so err towards skipping: icons and logos rarely declare a big size.
        best_width, best_url = 0.0, None
        srcset = values.get("srcset") or values.get("data-srcset") or ""
        for url, size, unit in re.findall(r"([^\s,]\S*)\s+(\d+(?:\.\d+)?)([wx])", srcset):
            width = float(size) if unit == "w" else 0.0
            if best_url is None or width > best_width:
                best_width, best_url = width, url
        declared = max((int(values[key]) for key in ("width", "height") if values.get(key, "").isdigit()), default=0)
        # Amazon's main picture declares no size but names its full-resolution copy.
        if values.get("data-old-hires"):
            self.images.append(values["data-old-hires"])
            return
        url = best_url or values.get("data-src") or values.get("src")
        if url and (best_width >= 600 or declared >= 300):
            self.images.append(url)


def _json_ld_products(value: object):
    """Yield every Product node without descending into it; its children are related products."""
    if isinstance(value, list):
        for child in value:
            yield from _json_ld_products(child)
    elif isinstance(value, dict):
        kinds = value.get("@type")
        kinds = kinds if isinstance(kinds, list) else [kinds]
        if "Product" in kinds or "ProductGroup" in kinds:
            yield value
            return
        for child in value.values():
            yield from _json_ld_products(child)


def _image_urls(value: object) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [url for item in value for url in _image_urls(item)]
    if isinstance(value, dict):
        return _image_urls(value.get("contentUrl") or value.get("url"))
    return []


def _product_page(page: str, base_url: str) -> tuple[str | None, list[str]]:
    """Title and candidate picture URLs, best first."""
    parser = _ProductPageParser()
    try:
        parser.feed(page)
        parser.close()
    except Exception:                                   # noqa: BLE001
        pass    # a malformed tail shouldn't cost what was already parsed

    title, candidates = None, []
    for block in parser.json_ld:
        try:
            data = json.loads(block, strict=False)
        except ValueError:
            continue
        for product in _json_ld_products(data):
            if not title and isinstance(product.get("name"), str):
                title = html.unescape(product["name"])
            candidates += _image_urls(product.get("image"))

    for key in ("og:image", "og:image:url", "og:image:secure_url", "twitter:image", "twitter:image:src"):
        candidates += parser.meta.get(key, [])
    candidates += parser.images

    if not title:
        title = next((parser.meta[key][0] for key in ("og:title", "twitter:title") if key in parser.meta), None) or parser.title
    title = " ".join(title.split())[:300] if title else None

    seen, ordered = set(), []
    for candidate in candidates:
        absolute = urljoin(base_url, candidate.strip())
        parts = urlsplit(absolute)
        if parts.scheme not in ("http", "https"):
            continue
        absolute = urlunsplit(parts._replace(fragment=""))
        # IKEA and Shopify list one file at several ?size= variants; a filename names the picture.
        is_file = re.search(r"\.(?:jpe?g|png|webp|gif|avif)$", parts.path, re.IGNORECASE)
        key = urlunsplit(parts._replace(query="", fragment="")) if is_file else absolute
        if key not in seen:
            seen.add(key)
            ordered.append(absolute)
    return title, ordered


def _download_image(url: str) -> tuple[bytes, str] | None:
    try:
        data, _, _ = _fetch(url, accept=IMAGE_ACCEPT, limit=MAX_FETCHED_IMAGE_BYTES, want="image/")
    except (_Refused, LookupError):
        return None
    content_type = _sniff_image(data)
    return (data, content_type) if content_type else None


@app.post("/objects/import", response_model=ImportResponse)
def import_object(request: ImportRequest) -> ImportResponse:
    if not os.environ.get("FAL_KEY"):
        raise HTTPException(status_code=503, detail="FAL_KEY is not set on the server")

    url = request.url.strip()
    if "://" not in url:
        url = "https://" + url      # people paste "ikea.com/..." as often as the full thing

    try:
        body, charset, source = _fetch(url, accept=PAGE_ACCEPT, limit=MAX_HTML_BYTES, want="text/html", truncate=True)
    except _Refused:
        raise HTTPException(status_code=400, detail=BAD_LINK)
    except LookupError:
        raise HTTPException(status_code=422, detail=NO_PICTURES)
    except _Blocked:
        raise HTTPException(status_code=502, detail=BLOCKED)

    try:
        page = body.decode(charset or "utf-8", errors="replace")
    except LookupError:
        page = body.decode("utf-8", errors="replace")

    title, candidates = _product_page(page, source)
    if not candidates:
        walled = any(marker in page.lower() for marker in BOT_WALL_MARKERS)
        raise HTTPException(status_code=502 if walled else 422, detail=BLOCKED if walled else NO_PICTURES)

    # Twice the cap, so a few broken or duplicate pictures still leave six.
    candidates = candidates[: MAX_IMPORTED_IMAGES * 2]
    blocked = False
    with ThreadPoolExecutor(max_workers=MAX_IMPORTED_IMAGES) as pool:
        futures = [pool.submit(_download_image, candidate) for candidate in candidates]
        downloads, digests = [], set()
        for future in futures:
            try:
                download = future.result()
            except _Blocked:
                blocked = True
                continue
            if download is None:
                continue
            # The same picture often sits in JSON-LD and og:image under different URLs.
            digest = hashlib.sha256(download[0]).digest()
            if digest not in digests:
                digests.add(digest)
                downloads.append(download)

        downloads = downloads[:MAX_IMPORTED_IMAGES]
        if not downloads:
            raise HTTPException(status_code=502 if blocked else 422, detail=BLOCKED if blocked else NO_PICTURES)

        try:
            # Re-hosted because shops block hotlinking and rotate their CDN URLs.
            images = list(pool.map(lambda download: _store(*download), downloads))
        except Exception:                               # noqa: BLE001
            raise HTTPException(status_code=502, detail=SAVE_FAILED)

    return ImportResponse(title=title, images=images, source=source)


def _require_fal_url(url: str, field: str) -> None:
    parts = urlsplit(url)
    try:
        port = parts.port
    except ValueError:
        port = -1
    if (
        parts.scheme != "https" or parts.hostname not in FAL_STORAGE_HOSTS or port not in (None, 443)
        or parts.username is not None or not url.isascii() or "\\" in url or any(c.isspace() for c in url)
    ):
        raise HTTPException(status_code=400, detail=f"{field} must be a URL returned by /upload or /objects/import")


def _boxed_first(objects: list[ComposeObject]) -> list[ComposeObject]:
    # Both the image order and the instruction's numbering come from here, so they can't drift apart.
    return sorted(objects, key=lambda item: item.marker is None)


def _compose_instruction(prompt: str, has_photo: bool, objects: list[ComposeObject]) -> str:
    scan = 2 if has_photo else 1
    if has_photo:
        lines = [
            "Image 1 is a real photograph of the user's room. The result must stay recognisably this "
            "room: the same walls, windows, doors, floor, ceiling, finishes and light. Keep everything "
            "the request below does not ask to change.",
            "Image 2 is a grey render of a 3D scan of the same room. It sets the exact viewpoint and "
            "layout of the result: match its camera position, angle and framing, and keep every wall, "
            "window and door where it is. Its greys are not colours; take colours and materials from image 1.",
        ]
    else:
        lines = [
            "Image 1 is a grey render of a 3D scan of the user's room. It sets the exact viewpoint and "
            "layout of the result: match its camera position, angle and framing, and keep every wall, "
            "window, door and opening exactly where it is. It carries shape only; choose colours, "
            "materials and light to suit the request below.",
        ]

    if objects:
        if any(item.marker is not None for item in objects):
            lines.append(f"Coloured boxes in image {scan} mark where products go, each drawn at the size that product should be:")
        for number, item in enumerate(_boxed_first(objects), start=scan + 1):
            name = " ".join(item.name.split())[:120]
            if item.marker is not None:
                lines.append(f"- The {item.marker} box marks where the {name} goes. Image {number} shows that exact product.")
            else:
                lines.append(
                    f"Image {number} shows the {name}, a product the user wants in the room. It has no box: place "
                    "it where the request says, or where it naturally belongs, such as a painting on a wall or a "
                    "pendant light from the ceiling."
                )
        lines += [
            "The product images show the exact items the user chose. Reproduce each one faithfully: the "
            "same shape, proportions, colour, material and details. Do not substitute a similar item or "
            "invent a different one. Ignore the product images' backgrounds, and light each product as "
            "the room would. If something already stands inside a box, the product replaces it.",
            "The boxes are placement guides only. They must not appear in the result: no coloured "
            "boxes, outlines, edges or tinted patches.",
        ]

    lines += [
        f"The user's request: {prompt}",
        f"Produce one photorealistic photograph of the room from image {scan}'s viewpoint.",
    ]
    return "\n".join(lines)


@app.post("/compose", response_model=ComposeResponse)
def compose(request: ComposeRequest) -> ComposeResponse:
    if not os.environ.get("FAL_KEY"):
        raise HTTPException(status_code=503, detail="FAL_KEY is not set on the server")

    image_urls = []
    if request.room_photo_url is not None:
        _require_fal_url(request.room_photo_url, "room_photo_url")
        image_urls.append(request.room_photo_url)
    _require_fal_url(request.scan_url, "scan_url")
    image_urls.append(request.scan_url)
    objects = _boxed_first(request.objects)
    for item in objects:
        _require_fal_url(item.image_url, "objects[].image_url")
        image_urls.append(item.image_url)

    markers = [item.marker for item in objects if item.marker is not None]
    if len(set(markers)) != len(markers):
        raise HTTPException(status_code=400, detail="each object needs its own marker colour")

    endpoint, options, ratio_field, ratios = COMPOSE_MODELS[request.model]
    arguments = {
        "prompt": _compose_instruction(request.prompt, request.room_photo_url is not None, objects),
        "image_urls": image_urls,
        "num_images": 1,
        "output_format": "jpeg",
        ratio_field: ratios[request.aspect_ratio],
        **options,
    }

    try:
        result = fal_client.subscribe(endpoint, arguments=arguments, client_timeout=COMPOSE_TIMEOUT)
    except Exception as error:                          # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"generation failed: {error}")

    urls = [image["url"] for image in result.get("images", []) if "url" in image]
    if not urls:
        raise HTTPException(status_code=502, detail="fal returned no image")

    return ComposeResponse(images=urls, model=endpoint)


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
