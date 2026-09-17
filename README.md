# The server

One job: hold the fal API key so the phone doesn't have to. `/generate` passes a
conditioning render through to the image model; `/suggest` turns the facts of a
scanned room into a handful of one-line redesign briefs, so the app can offer
chips instead of an empty text box; `/interpret` turns a sentence the user says
about their room into corrections the app can apply.

This server stores nothing itself, but it does not follow that nothing is
stored. Every image it handles — the scan renders, the product pictures and the
photographs of the user's own home — is uploaded to fal's CDN, because the image
models take a URL rather than bytes. fal holds those files and serves them to
anyone who has the link. What this server controls is how long that lasts: see
[What fal keeps](#what-fal-keeps). The phone keeps every scan, plan and image;
backups are meant to go from the device straight to object storage, never
through here — Vercel caps request bodies at 4.5 MB and a room scan is bigger
than that.

## What fal keeps

Every upload and every generated picture is given an hour to live, through
`X-Fal-Object-Lifecycle-Preference` — `lifecycle=` on the uploads, the header
itself on the two `subscribe` calls. fal's default is 60 days, so this is the
change that matters most: a photograph of someone's living room stops being
reachable an hour after it was taken rather than two months later. An hour is
far more than the flow needs — the app uploads, composes and downloads in one
run, and nothing on the phone ever refers to a fal URL again — but it leaves
room for a slow `/compose`, which may take 280s.

**The files are public for that hour, and that could not be avoided.** fal can
mark a file `forbid` (403 to strangers) or `hide` (404), and that works: a file
set either way answered an anonymous fetch with 403 and 404 respectively, where
it had answered 200. But measured on 2026-09-17 against
`fal-ai/nano-banana-2/edit`, the same model `/compose` uses, fal's own image
models **cannot read a file whose ACL is not public**. The same bytes and prompt
succeeded on a public URL and failed on both a `forbid` and a `hide` one, so the
restriction, not the test, is what broke it. A restricted file is unreadable
even to the account that owns it and holds the key — the CDN wants a signed
token instead. So an ACL cannot be set on anything `/compose` has to read, and
`app.py` deliberately sends none; `test_uploads.py` guards that, because adding
one would break composing rather than fail loudly here.

Setting the ACL is also a second step rather than part of the upload: passing
`initial_acl` in the upload's lifecycle header made fal silently discard the
whole preference, the expiry along with it. It only took effect through
`PUT https://rest.fal.ai/storage/files/acl?url=…`. That is recorded in case the
models ever learn to read restricted inputs; nothing here uses it today.

What this leaves is worth stating plainly for a privacy policy: images of the
user's home reach a third party, are readable by anyone holding the link, and
are deleted an hour later.

## Running it locally

```bash
python3 -m venv .venv
./.venv/bin/pip install -r requirements.txt fastapi[standard]
export FAL_KEY=...
./.venv/bin/fastapi dev app.py
```

Then point the app at `http://<your-mac>:8000` in its Settings.

## Tests

```bash
./.venv/bin/pip install -r requirements-dev.txt
./.venv/bin/python -m pytest
```

`test_interpret.py` stubs the model call out, so the suite needs no key, no
network and no money.

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

An object's `marker` is optional. Paintings, mirrors, curtains and pendant
lights can't be boxed on the floor plan, so they go after the boxed objects and
the instruction tells the model to place them where the request says.

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

## Correcting a scan in words

RoomPlan mislabels things — it read a wardrobe as a refrigerator and gave it a
box half its real width — and the app infers the room type from the object
categories, so one wrong object turns a guest room into a kitchen and quietly
poisons every suggestion after it. `/interpret` takes one sentence the user
typed or spoke, plus the room as the scan currently believes it, and returns
`roomKind` (free text, `null` when the sentence names no kind), `objectEdits`
for what the sentence settles, `questions` for what it doesn't, and `unchanged`
sentences for whatever couldn't be acted on.

**A vague size is a question, never an edit.** "Wider", "much bigger" and "too
tall" come back as a question with two options measured off the scan — twice
the current value, half the room's width, floor to ceiling — because the
object's box is what the image model is conditioned on, and a guessed number
produces a confidently wrong picture. Only a number you could measure with
("1.2 metres wide", "about a metre and a half") becomes an edit. When the
geometry can't yield two honest options the question is dropped and the user is
told, because a single option is the same guess wearing a hat.

The model is asked for JSON and then disbelieved: ids it invents are dropped,
sizes outside 0-30 m are dropped, an edit that settles nothing is dropped, a
number the sentence actually gave beats a question about the same field, and a
reply that doesn't parse comes back as 200 with a line in `unchanged` rather
than a 500. An empty sentence never reaches the model at all.

Text the user may be shown — `ask`, `unchanged` — comes back in the language
they wrote in; `roomKind` and `category` come back in English, because that is
what `/suggest` and `/compose` read. **Option labels are always English**: they
are built here from the geometry, not by the model, so a Hebrew question
currently carries English labels. Worth fixing when the app grows a second
language.

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
