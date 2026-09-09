# The server

One job: hold the fal API key so the phone doesn't have to, and pass a
conditioning render through to the image model.

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

Nothing here has been run. `fal-ai/flux-control-lora-depth` and its
`preprocess_depth` flag were confirmed against fal's published schema, but the
canny sibling's `preprocess_canny` field name was **not** — check it before
relying on the `lines` mode.
