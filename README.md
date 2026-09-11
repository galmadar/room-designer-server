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

## The photo flow

`/upload` takes base64 JPEG or PNG up to 4 MB, checks the magic bytes against
the declared type, and returns a fal storage URL. `/objects/import` fetches a
shop page, takes pictures from JSON-LD `Product.image`, then `og:image`, then
`twitter:image`, then big `<img>` tags, and re-uploads up to six to fal — shops
block hotlinking, and fal's URLs don't rotate. `/compose` sends the room photo
(if any), the scan render and the product pictures, in that order, to
`fal-ai/nano-banana-2/edit` or `openai/gpt-image-2/edit`.

**`/compose` only forwards URLs on `v3b.fal.media`**, the one host `/upload` and
`/objects/import` were seen returning. Otherwise it would make fal fetch any URL
on our key. Uploads are pinned to fal's v3 repository with no fallback, so a
fallback host can't slip in; if fal moves hosts, `/compose` answers 400 until
`FAL_STORAGE_HOSTS` is updated.

**`/objects/import` fetches user-supplied URLs from a box that holds the key.**
It allows http and https on ports 80 and 443 only, resolves the host and refuses
anything that isn't a public unicast address, and follows at most three
redirects by hand, re-checking each hop. Pages stop at 3 MB, pictures at 8 MB,
counted while streaming. Resolve-then-connect leaves a DNS-rebinding window;
that's accepted at this scale.

### The instruction wrapper

The user's prompt is wrapped in text that says what each image is. Measured on
2026-09-11 with nano-banana-2, a real photo, a scan render with a green box on
the sofa and a red box in an empty corner, two IKEA products, and the prompt
"Make it feel warm and cosy for winter evenings":

- **Raw prompt, no wrapper:** the lamp landed in its corner, but the old sofa
  stayed. The model had no way to know the green box meant the IKEA sofa.
- **Wrapped:** both products in place, same room, no boxes in the output.
- **Wrapped, minus "the product replaces whatever stands in the box":** still
  swapped. The line that matters is the one binding each colour to a product
  and an image number.

Without a room photo the scan's geometry holds, but a flat grey table in the
test render came back as a rug. Adding "plain grey shapes are existing
furniture" didn't change that, so it was dropped; the render probably needs to
show furniture height more clearly than the test one did.

One sample per variant, so treat these as leads, not proof.

## Function duration

There is no `vercel.json`, so functions get Vercel's default under Fluid
compute: 300s, which is also the Hobby maximum. gpt-image-2 took 99s for one
1024×768 high-quality edit and nano-banana-2 took 12-15s. `/compose` gives up
at 280s so it can answer with an error before Vercel kills it.

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
