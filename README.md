# The server

One job: hold the fal API key so the phone doesn't have to. `/generate` passes a
conditioning render through to the image model; `/suggest` turns the facts of a
scanned room into a handful of one-line redesign briefs, so the app can offer
chips instead of an empty text box.

It stores nothing. The phone keeps every scan, plan and image; backups are meant
to go from the device straight to object storage, never through here — Vercel
caps request bodies at 4.5 MB and a room scan is bigger than that.

## Running it locally

```bash
python3 -m venv .venv
./.venv/bin/pip install -r requirements.txt fastapi[standard]
export FAL_KEY=...
./.venv/bin/fastapi dev app.py
```

Then point the app at `http://<your-mac>:8000` in its Settings.

## Deploying

```bash
vercel deploy
vercel env add FAL_KEY
```

Python is a first-class Vercel runtime and FastAPI is detected automatically
from `requirements.txt`, so there is no build configuration. Hobby tier allows
300s per request, which comfortably covers a 30-90s generation.

## Not verified

`fal-ai/flux-control-lora-depth` and its `preprocess_depth` flag were confirmed
against fal's published schema, but the canny sibling's `preprocess_canny` field
name was **not** — check it before relying on the `lines` mode.

## z-image turbo: measured and rejected

`fal-ai/z-image/turbo/controlnet` looks like an easy win — $0.0065/megapixel
against Flux's $0.04, and 8 steps against 28. It was tried on 2026-09-10 and it
does not work: the endpoint accepts `image_url` and then ignores it.

Held prompt and seed fixed and varied only the control image. A depth map, that
same map mirrored, a canny render and an unrelated photograph all produced the
**same image**, pixel for pixel. So did `control_scale: 0` against
`control_scale: 1`, `preprocess: none` against `depth` and `canny`, and
`acceleration: none`. That image is also what plain `fal-ai/z-image/turbo`
returns for the same prompt and seed, so the controlnet endpoint is serving the
base text-to-image model.

A cheaper model that discards the scan is worth nothing here — the scan is the
product. Re-test before switching; the price is worth going back for.
