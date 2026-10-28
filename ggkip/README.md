# ggkip

Crawls official government websites, extracts their text, normalizes it with an
LLM (vLLM or llama.cpp) into a fixed category/subcategory taxonomy and stores the
records in PostgreSQL. Go standard library plus `golang.org/x/net`,
`golang.org/x/time` and the `lib/pq` driver; nothing else.

```text
scheduler → due site → crawl run → robots.txt → sitemaps → links
  → fetch (text/html, text/plain, JSON only) → extract clean text → filter
  → SHA-256 → unchanged? stop : LLM → validate taxonomy → store record
```

See [docs/GUIDE.md](docs/GUIDE.md) for every `make` command, the full API and
server sizing (GPU, CPU, RAM, disk).

## Quick start

```sh
docker run -d --name ggkip-pg -e POSTGRES_USER=ggkip -e POSTGRES_PASSWORD=ggkip \
  -e POSTGRES_DB=ggkip -p 5432:5432 postgres:17-alpine

# an OpenAI-compatible model server, for example:
#   vllm serve Qwen/Qwen3-8B --port 8000
#   llama-server -m qwen.gguf --port 8080 --jinja

go run ./src/main --env-file src/resources/env/local.env --llm-runtime=vllm
```

The [Makefile](Makefile) wraps the usual commands (`make` lists them):

```sh
make db-up                                        # PostgreSQL in Docker
make run                                          # foreground, ENV=src/resources/env/local.env
make start ENV=src/resources/env/llamacpp.env     # background, log in logs/ggkip.log (LOG=)
make status | make logs | make stop | make restart
make batch [COUNTRIES=242,1] | make batch-status ID=1 | make batch-cancel ID=1 | make runs BATCH=1
make check                                        # gofmt, vet, race tests, build
```

On start the app runs pending migrations, loads the seed data (when
`SEED_ON_START=true`) and starts the scheduler and the HTTP API on `HTTP_PORT`
(8090). Stop it with Ctrl-C: running crawls are cancelled and recorded as
`cancelled`.

With the scheduler on, every active official site in `sites.json` is crawled
right away. To try the pipeline on a single site, set `SCHEDULER_ENABLED=false`
and call `POST /v1/sites/{id}/crawl`.

## Configuration

Environment variables, optionally pre-filled from `--env-file` (real environment
variables win). See [example.env](src/resources/env/example.env) for the full
list and defaults.

| Flag | Overrides | Meaning |
|---|---|---|
| `--llm-runtime=vllm\|llamacpp` | `LLM_RUNTIME` | model server type |
| `--migrate=true\|false` | `MIGRATE_ON_START` | apply migrations on start (default true) |
| `--seed=true\|false` | `SEED_ON_START` | load seed JSON on start (default false) |
| `--env-file=path` | | KEY=VALUE defaults |

`SEED_COUNTRIES=UZ,JP` seeds only those countries' organizations and sites
(all countries and categories are still seeded). Rows already in the database
are kept. [llamacpp.env](src/resources/env/llamacpp.env) is a ready test setup:
llama-server on port 8080, Uzbekistan only, scheduler off.

`RESOURCES_DIR` points at `src/resources` on disk. Without it, the copy
embedded in the binary is used, so a built binary needs no extra files.

Crawl limits, file extensions to skip, tracking parameters and the minimum text
length are in [crawler.json](src/resources/settings/crawler.json). Record
types and the fallback category code are in
[taxonomy.json](src/resources/settings/taxonomy.json).

## Layout

```text
src/main/
  main.go            wiring only
  app/               scheduler, seed loader, settings, HTTP server
  infrastructure/    config, logger, PostgreSQL, migration runner, HTTP helpers
  modules/<module>/{dto,model,service,handler,repo}
    country, organization_type, organization, category, site, page, record
    crawl       crawl runs + the pipeline runner (crawl → dedup → LLM → store)
    crawler     URL normalization, robots.txt, sitemaps, fetcher, frontier
    extractor   HTML → clean text, content filter
    llm         Client interface, vLLM / llama.cpp clients, schema, validator
src/resources/      migrations, seed, settings, env (embedded via resources.go)
```

Only repositories run SQL. Handlers call services; services own the business rules.

## Rules worth knowing

- **Dedup.** `pages.content_hash` is set only after a record has been stored. A
  page with the same hash is `unchanged` and never reaches the LLM. A page whose
  LLM call failed keeps its old hash, so the next crawl tries it again.
- **One record per page.** A changed page updates its record (`records_updated`)
  instead of creating a new one.
- **Taxonomy.** The LLM gets the active categories (with descriptions) and the
  record types. Its output is constrained by a JSON schema with enums and then
  validated again: codes must exist, the subcategory must belong to the category,
  and `UNKNOWN` has no subcategory. Invalid output is retried, then the page is
  marked `failed`. It is never stored.
- **Scheduling.** Success sets `next_crawl_at = now + crawl_interval_sec`.
  Failure stores `last_error` and retries after `SCHEDULER_FAILURE_RETRY_SEC`
  (capped by the interval). At most `CRAWL_MAX_SITES` sites crawl at once
  (scheduler, manual triggers and batches together; it defaults to
  `SCHEDULER_MAX_SITES`), and at most `LLM_WORKERS` pages are at the LLM.
- **Crawling.** It stays on the site's domain and `www.` host, honours robots.txt
  (a 5xx or unreachable robots.txt stops the crawl, as RFC 9309 says), meta
  `noindex`/`nofollow` and `Crawl-delay`, and stops after `max_pages_per_run`
  URLs. Sitemap entries with the newest `lastmod` are crawled first. JavaScript
  is not executed.
- **Seeds** are upserted by id (`sites` by domain) in one transaction. The seed
  files are the source of truth: a site set inactive by hand becomes active
  again on the next seeded start. Crawl state columns are never touched.
  `states.json` has no table yet and is not loaded.
- **Panics never stop the app.** A panic is recovered where it happens, logged
  with its stack (`stack=`), and fails only the smallest unit: a page in a
  crawl worker or LLM worker (`pages_failed`), a site's run (`failed`, with
  `panic: …` in `error_message`), one site of a batch, one scheduler tick, or
  one HTTP request (500). Every level waits for the goroutines it started, so
  a crawl cannot hang on a worker that died and no slot leaks.
- Runs and batches left `running` by a crashed process are closed as `failed` at startup.
  The "already running" check is in memory, so run one instance at a time.

## HTTP API

| Endpoint | Notes |
|---|---|
| `GET /health` | pings the database |
| `GET /v1/countries` `/{id}` | `?region`, `?is_active`, `?locale=uz\|ru\|kaa`; detail includes translations |
| `GET /v1/organization-types` | |
| `GET /v1/organizations` `/{id}` | `?country_id`, `?organization_type_id`, `?parent_id`, `?status` |
| `GET /v1/categories` `/{id}` | `?parent_id`, `?top_level=true`, `?tree=true` |
| `GET /v1/sites` `/{id}` | `?country_id`, `?organization_id`, `?status` |
| `POST /v1/sites/{id}/crawl` | 202 + crawl run; 409 if running, 503 if at capacity |
| `POST /v1/crawl-batches` | crawl every active country at once (see below); body `{"country_ids": [...]}` is optional; 202 + batch, 409 if a batch runs, 400 if no sites |
| `GET /v1/crawl-batches` `/{id}` | `?status`; the detail adds `runs` (crawl runs per status) and `sites_pending` |
| `POST /v1/crawl-batches/{id}/cancel` | 202; waiting sites are not started, running crawls end `cancelled`; 409 if finished |
| `GET /v1/crawl-runs` `/{id}` | `?site_id`, `?batch_id`, `?status` |
| `GET /v1/pages` | `?site_id`, `?status` |
| `GET /v1/records` `/{id}` | `?country_id`, `?organization_id`, `?site_id`, `?category_id`, `?subcategory_id`, `?record_type`, `?language`, `?published_from`, `?published_to` |

To save records to a JSON file (with category codes, organization and domain
names and the full content):
`python3 scripts/export_records.py --site-id 8 --out stat_uz.json`
(`--help` lists the filters).

Lists return `{"items": [...], "next_cursor": N}`. Pass `?cursor=N` to get the
next page and `?limit=` to set the page size. Records, pages and crawl runs are
newest first.

## Crawling all active countries

`POST /v1/crawl-batches` crawls every active official site of every active
country, ignoring `next_crawl_at`:

```text
POST /v1/crawl-batches
  ├── goroutine: country UZ ──► its sites, up to CRAWL_MAX_SITES_PER_COUNTRY at once
  ├── goroutine: country JP ──► …
  └── goroutine: country GB ──► …
        every site also waits for one of CRAWL_MAX_SITES global slots
        every page waits for one of LLM_WORKERS LLM slots (shared by all sites)
```

```sh
curl -s -X POST localhost:8090/v1/crawl-batches                              # all active countries
curl -s -X POST localhost:8090/v1/crawl-batches -d '{"country_ids": [242]}'   # only these
curl -s localhost:8090/v1/crawl-batches/1           # runs per status, sites_pending
curl -s "localhost:8090/v1/crawl-runs?batch_id=1"   # one row per site
curl -s -X POST localhost:8090/v1/crawl-batches/1/cancel
```

The per-country limit keeps one country with many sites from taking every
slot. One batch runs at a time. A site that is already being crawled (by the
scheduler or a manual trigger) is skipped and counted in `sites_skipped`. A
batch is `completed` when every site was tried, even if some runs failed; the
per-site outcome is in `runs`. Stopping the app ends the batch as `cancelled`.

## Tests

```sh
go test ./... && go vet ./... && go build ./...
```

Unit tests use `httptest` servers and fake LLM replies; nothing contacts real sites.
