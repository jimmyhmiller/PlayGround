# bookshelf-cloud

A serverless, statically-hosted **mock AudiobookShelf server** that BookPlayer
(iOS) talks to. Sits in front of an S3 bucket containing finished `.m4b` files
produced by the sibling [`paper-audiobooks`](../claude-experiments/paper-audiobooks/)
pipeline.

## Goal

```
   drop PDF in ~/audiobooks/inbox/
              │
              ▼   (existing watcher — paper-audiobooks pipeline)
   ~/audiobooks/m4b/<book>.m4b
              │
              ▼   (NEW — this project's publish step)
   s3://bookshelf-cloud-<acct>/books/<book>/audio.m4b
   s3://bookshelf-cloud-<acct>/books/<book>/cover.jpg
   s3://bookshelf-cloud-<acct>/api/items/<book>.json
   s3://bookshelf-cloud-<acct>/api/libraries/main/items.json
              │
              ▼
   BookPlayer iOS → CloudFront → JWT-gated → plays from S3
```

End state: drop a PDF in `~/audiobooks/inbox/`, walk away, the book appears in
BookPlayer.

## Why this exists

BookPlayer supports AudiobookShelf as a remote library backend. Real ABS is a
full server (Node + SQLite + transcoding). We don't need any of that — our
books are pre-rendered single-file `.m4b`s with embedded chapters. We want:

- zero idle cost
- real password auth (no shared secret baked into the client)
- no servers to maintain
- works with stock BookPlayer, no fork

BookPlayer's ABS client (`AudiobookShelfConnectionService.swift`) calls a tiny
subset of the ABS API: list/detail/download. No progress sync. No WebSocket.
No playback sessions. This makes a static shim viable.

## Architecture

```
                       BookPlayer iOS
                            │ HTTPS
                            ▼
                  ┌──────────────────────────────┐
                  │   Lambda Function URL        │
                  │   (single handler)           │
                  │                              │
                  │   - /ping                    │
                  │   - /login   (rate-limited,  │
                  │              bcrypt, mint    │
                  │              HS256 JWT)      │
                  │   - /api/*   (JWT verify,    │
                  │              serve manifest  │
                  │              JSON from S3)   │
                  │   - /api/items/<id>/cover    │
                  │   - /api/items/<id>/download │
                  │              (JWT verify,    │
                  │              302 to presigned│
                  │              S3 URL)         │
                  └──────────────┬───────────────┘
                                 │
                                 ▼
                  ┌──────────────────────────────┐
                  │   S3 (private)               │
                  │                              │
                  │   /api/libraries             │
                  │   /api/libraries/main/items  │
                  │   /api/items/<id>            │
                  │   /books/<id>/audio.m4b      │
                  │   /books/<id>/cover.jpg      │
                  │   /ratelimit/<ip>.json       │
                  └──────────────────────────────┘
```

CloudFront was dropped because (a) our account is gated from creating
distributions via the API/CloudFormation, and (b) we had already disabled
caching, so we were paying CloudFront's overhead without using its only
useful feature. Media is served via S3 presigned URLs (30-min TTL) which
AVPlayer follows on 302; S3 supports Range requests so seeking works.

## Data layout in S3

Single bucket. All paths are stable so manifests can hardcode them.

```
bookshelf-cloud-<acct>/
├── api/
│   ├── ping.json                          # {"success": true}
│   ├── libraries.json                     # list of libraries (we have one)
│   ├── libraries/main/
│   │   ├── items.json                     # list of all books
│   │   ├── filterdata.json                # {} — BookPlayer tolerates empty
│   │   ├── collections.json               # {"results": []}
│   │   └── search.json                    # {"book": []} — search unsupported
│   └── items/
│       └── <book-id>.json                 # full item detail w/ chapters
└── books/
    └── <book-id>/
        ├── audio.m4b                      # the actual audiobook
        └── cover.jpg                      # cover art
```

`<book-id>` is the slugified book filename stem (e.g. `nature-of-belief`).
It's stable across re-publishes of the same book.

### Why JSON-as-static-files instead of a Lambda that generates them

- Cheaper (no per-request compute beyond the CF Function).
- Simpler — `publish` writes files, done. No DynamoDB, no SQLite.
- Easy to inspect: `aws s3 cp s3://.../api/items/foo.json -` and see exactly
  what BookPlayer will see.
- Re-publishing is just `aws s3 sync` of a local manifest tree.

## Endpoints BookPlayer calls

All under the CloudFront distribution domain. Auth is `Authorization: Bearer
<jwt>` on every endpoint except `/login` (no auth) and `/api/items/<id>/download`
(JWT in `?token=` query param).

| Method | Path | Returns | Source |
|---|---|---|---|
| GET | `/ping` | `{"success": true}` | static S3 |
| POST | `/login` | `{"user": {"id": "...", "token": "<jwt>"}}` | Lambda |
| GET | `/api/libraries` | `{"libraries": [...]}` | static S3 |
| GET | `/api/libraries/{libId}/items` | `{"results": [...], "total": N}` | static S3 |
| GET | `/api/libraries/{libId}/filterdata` | `{}` | static S3 |
| GET | `/api/libraries/{libId}/collections` | `{"results": []}` | static S3 |
| GET | `/api/libraries/{libId}/search` | `{"book": []}` | static S3 |
| GET | `/api/items/{id}` | full item w/ chapters, tracks | static S3 |
| GET | `/api/items/{id}/cover` | 302 → presigned S3 URL | Lambda |
| GET | `/api/items/{id}/download?token=...` | 302 → presigned S3 URL (m4b, Range-served) | Lambda |

The Lambda handler does URL rewriting: `/api/items/foo/cover` is signed for
the S3 key `/books/foo/cover.jpg`, and `/api/items/foo/download` for
`/books/foo/audio.m4b`.

### Things BookPlayer does NOT call

(Confirmed by reading `AudiobookShelfConnectionService.swift` — this is the
only file in the BookPlayer codebase that makes ABS API calls.)

- No `/api/me`, `/api/me/progress` — progress is local-only
- No `/api/session/*` or `/api/items/{id}/play` — no playback sessions
- No `/socket.io/` — no WebSocket
- No `/api/authorize` — uses legacy `/login` flow

We do NOT need to implement these.

## JWT design

- Algorithm: **HS256** (HMAC-SHA256). CloudFront Functions support
  `crypto.createHmac` natively. No RSA/ECDSA needed — single-user system.
- Secret: stored in CloudFront Function code (rotatable via redeploy) and in
  the login Lambda's env vars. Generated at `cdk deploy` time, stored in
  Secrets Manager, injected into both.
- Claims: `{sub: "user", iat, exp}`. No roles; single-user.
- Expiry: **30 days**. BookPlayer has no refresh logic — re-login is the only
  refresh path, so a short TTL would be annoying. 30 days balances annoyance
  vs. blast radius if a phone is lost.
- No revocation list. If we ever need it: bump the HMAC secret to invalidate
  all outstanding tokens.

## Authentication flow

1. User opens BookPlayer, enters server URL (`https://<distribution>.cloudfront.net`),
   username, password.
2. BookPlayer `POST /login` with `{username, password}`.
3. Login Lambda:
   - Looks up username in env-var-stored bcrypt hash map (one user for now;
     extensible to Secrets Manager later).
   - Verifies password with bcrypt.
   - Signs JWT with HMAC secret.
   - Returns `{"user": {"id": "user", "token": "<jwt>"}}`.
4. BookPlayer stores token in iOS keychain, sends as `Authorization: Bearer`
   on every subsequent request.
5. CloudFront Function on each request:
   - Extracts JWT from `Authorization` header or `?token=` query param
     (for `/download` URLs only).
   - Verifies HMAC-SHA256 signature and `exp` claim.
   - On success: forwards to S3 origin. On failure: returns 401.

## Publishing a book (the dev loop)

The existing `paper-audiobooks` watcher already runs `paper-audiobooks one
<pdf>` and lands an `.m4b` in `~/audiobooks/m4b/`. We extend it with a
post-finish hook:

1. Existing watcher finishes a book → m4b at `~/audiobooks/m4b/<slug>.m4b`.
2. Hook fires `bookshelf-cloud publish ~/audiobooks/m4b/<slug>.m4b`.
3. `publish` command:
   - `ffprobe` the m4b to extract title, author, duration, chapters.
   - Extract embedded cover art (or first-page PDF render as fallback).
   - Upload to S3:
     - `books/<slug>/audio.m4b`
     - `books/<slug>/cover.jpg`
   - Read existing `api/libraries/main/items.json`, splice in the new book
     (or update if `<slug>` already present), write back.
   - Write `api/items/<slug>.json` with chapter list and a `contentUrl`
     hint (though BookPlayer ignores `contentUrl` and uses `/download`).
4. CloudFront has caching disabled, so next BookPlayer refresh sees it.

The watcher integration is a one-line addition to the existing
`audiobooks-watcher` service: tack `&& bookshelf-cloud publish "$M4B"` onto
the end of the per-book pipeline command.

## What goes in the CDK app

```
bookshelf-cloud/
├── DESIGN.md                       (this file)
├── README.md
├── cdk/
│   ├── bin/bookshelf-cloud.ts      (CDK entrypoint)
│   ├── lib/bookshelf-cloud-stack.ts
│   │   ─ S3 bucket (private, OAC)
│   │   ─ CloudFront distribution (caching off, OAC to S3)
│   │   ─ CloudFront Function (JWT verifier — inline JS)
│   │   ─ Lambda (login handler — Node, bcrypt)
│   │   ─ Lambda Function URL (CORS not needed; BookPlayer is native)
│   │   ─ Secrets Manager secret (HMAC secret + bcrypt user table)
│   ├── functions/
│   │   ├── login/index.mjs         (login Lambda source)
│   │   └── jwt-verify/index.js     (CloudFront Function source)
│   ├── package.json
│   ├── tsconfig.json
│   └── cdk.json
├── publish/
│   ├── pyproject.toml              (Python — reuse paper-audiobooks tooling)
│   └── src/bookshelf_cloud/
│       ├── __init__.py
│       ├── cli.py                  (publish, list, remove commands)
│       ├── manifest.py             (build ABS-shaped JSON from m4b metadata)
│       ├── ffprobe.py              (chapter + duration extraction)
│       └── s3.py                   (uploads + manifest reconciliation)
└── tests/
    ├── manifest_test.py            (golden ABS JSON snapshot tests)
    └── jwt_test.mjs                (CF Function logic, runnable under Node)
```

Two languages: TypeScript for CDK + Lambda + CF Function (it's all JS-shaped
already), Python for the `publish` CLI (reuses our existing ffprobe/m4b
tooling and matches the paper-audiobooks repo it sits next to).

## Settled decisions

- **Bucket name**: `bookshelf-cloud-<account-id>`. The account-ID suffix
  guarantees global uniqueness; we never type it by hand so the length
  doesn't matter. The name is exported as a CDK output and read by the
  `publish` CLI from a local config file so the CLI and stack stay in sync.
- **Cover art**: best-effort. Use the m4b's embedded cover if `ffprobe`
  surfaces one. If not, ship without a cover — BookPlayer renders a default.
  No PDF-rendering fallback.
- **Login cold start**: accepted. Lambda is unprovisioned. A ~400ms first hit
  after idle is fine for a once-a-month re-login.
- **Single user**: hardcoded. One bcrypt-hashed user in Secrets Manager.
  No user-management surface; rotating credentials = redeploy.
- **No custom domain**: BookPlayer points directly at the raw
  `*.cloudfront.net` URL. No ACM, no Route53.

## Non-goals

- Listening-progress sync across devices (BookPlayer doesn't sync to ABS
  anyway).
- Search (BookPlayer searches the locally-fetched library).
- Transcoding (every file is already an `.m4b`).
- A web UI (the CLI publishes; BookPlayer is the only consumer).
- Supporting any other client than BookPlayer.
