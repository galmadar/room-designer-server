"""What lifecycle fal is asked for, checked without network, key or money.

The point of these tests is the *shape* of the request, not fal's answer: every
call out is stubbed. The one thing they guard hardest is the absence of an ACL —
see test_no_acl_is_requested.
"""

import json
import struct
import zlib

import fal_client
import pytest
from fastapi.testclient import TestClient

import app as server

client = TestClient(server.app)

FAL_URL = "https://v3b.fal.media/files/b/0aaac476/probe.png"


def png(width: int = 8, height: int = 8) -> bytes:
    """A real PNG: /upload checks the magic bytes against the declared type."""
    raw = b"".join(b"\x00" + b"\x78\x8c\xaa" * width for _ in range(height))

    def chunk(tag: bytes, body: bytes) -> bytes:
        payload = tag + body
        return struct.pack(">I", len(body)) + payload + struct.pack(">I", zlib.crc32(payload))

    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw))
        + chunk(b"IEND", b"")
    )


@pytest.fixture(autouse=True)
def key(monkeypatch):
    monkeypatch.setenv("FAL_KEY", "not-a-real-key")


@pytest.fixture
def calls(monkeypatch):
    """Capture every argument the app hands fal, and let nothing leave the box."""
    seen = {"upload": [], "upload_file": [], "subscribe": []}

    def upload(data, content_type, file_name=None, **kwargs):
        seen["upload"].append(kwargs)
        return FAL_URL

    def upload_file(path, **kwargs):
        seen["upload_file"].append(kwargs)
        return FAL_URL

    def subscribe(endpoint, **kwargs):
        seen["subscribe"].append(kwargs)
        return {"images": [{"url": FAL_URL}]}

    monkeypatch.setattr(fal_client, "upload", upload)
    monkeypatch.setattr(fal_client, "upload_file", upload_file)
    monkeypatch.setattr(fal_client, "subscribe", subscribe)
    return seen


def lifetime(kwargs) -> int:
    assert "lifecycle" in kwargs, "the upload asked fal for no lifetime at all"
    return kwargs["lifecycle"].expires_in


# --- the header itself -------------------------------------------------------


def test_header_is_the_one_fal_documents():
    assert list(server.LIFECYCLE_HEADERS) == ["X-Fal-Object-Lifecycle-Preference"]
    body = json.loads(server.LIFECYCLE_HEADERS["X-Fal-Object-Lifecycle-Preference"])
    assert body == {"expiration_duration_seconds": 3600}


def test_an_hour_is_long_enough_for_the_flow():
    # /compose alone may run 280s, and the app uploads before it and downloads after.
    assert server.UPLOAD_LIFETIME_SECONDS >= server.COMPOSE_TIMEOUT * 2
    assert server.UPLOAD_LIFECYCLE.expires_in == server.UPLOAD_LIFETIME_SECONDS


def test_no_acl_is_requested():
    """A `forbid`/`hide` ACL makes fal's own models fail to read the input.

    Measured 2026-09-17. If this ever becomes an `initial_acl`, /compose breaks.
    """
    body = json.loads(server.LIFECYCLE_HEADERS["X-Fal-Object-Lifecycle-Preference"])
    assert "initial_acl" not in body
    assert server.UPLOAD_LIFECYCLE.initial_acl is None


# --- every path that puts bytes on fal --------------------------------------


def test_upload_endpoint_sets_the_lifetime(calls):
    reply = client.post("/upload", json={"image": _b64(png()), "content_type": "image/png"})
    assert reply.status_code == 200, reply.text
    assert lifetime(calls["upload"][0]) == server.UPLOAD_LIFETIME_SECONDS


def test_store_sets_the_lifetime(calls):
    """The shared path, so /objects/import is covered wherever it re-hosts."""
    server._store(png(), "image/png")
    assert lifetime(calls["upload"][0]) == server.UPLOAD_LIFETIME_SECONDS


def test_generate_sets_it_on_both_the_upload_and_the_result(calls):
    reply = client.post("/generate", json={"prompt": "a room", "image": _b64(png())})
    assert reply.status_code == 200, reply.text
    assert lifetime(calls["upload_file"][0]) == server.UPLOAD_LIFETIME_SECONDS
    assert calls["subscribe"][0]["headers"] == server.LIFECYCLE_HEADERS


def test_compose_sets_it_on_the_result(calls):
    reply = client.post("/compose", json={"prompt": "warm", "scan_url": FAL_URL})
    assert reply.status_code == 200, reply.text
    assert calls["subscribe"][0]["headers"] == server.LIFECYCLE_HEADERS


def _b64(data: bytes) -> str:
    import base64

    return base64.b64encode(data).decode()
