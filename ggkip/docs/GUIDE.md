# ggkip guide: architecture, business logic, commands, API and server sizing

1. [Architecture](#1-architecture): the parts of the system, code layers, data model, concurrency.
2. [Business logic](#2-business-logic): how a crawl, a page and an LLM call are processed, statuses, scheduling, panics.
3. [Makefile commands](#3-makefile-commands): what each `make` target does and when to use it.
4. [HTTP API](#4-http-api): every endpoint, its parameters and replies.
5. [Server sizing](#5-server-sizing): how much GPU, CPU, RAM and disk the project needs, and why.

The diagrams are Mermaid; GitHub and VS Code (Markdown preview) draw them.

---

## 1. Architecture

### System context

ggkip is one Go process. It talks to four things: the operator (through the HTTP API), the government websites, an LLM server and PostgreSQL.

```mermaid
flowchart LR
    op["Operator<br/>curl / make / export script"]
    subgraph host["Server"]
        app["ggkip<br/>Go, :8090"]
        llm["LLM server<br/>llama-server :8080<br/>or vLLM :8000"]
        db[("PostgreSQL<br/>:5432")]
        gpu[["GPU"]]
    end
    web["Government websites<br/>robots.txt, sitemaps, HTML"]

    op -- "HTTP API (JSON)" --> app
    app -- "HTTPS GET, 2 req/s per site" --> web
    app -- "POST /v1/chat/completions<br/>JSON schema / json_object" --> llm
    llm --- gpu
    app -- "SQL (lib/pq)" --> db
```

| Part | Role | Why it is separate |
|---|---|---|
| ggkip | crawls, extracts text, deduplicates, calls the LLM, validates, stores, serves the API | one binary; migrations, seed and settings are embedded in it |
| LLM server | turns page text into a category, type, title, summary and keywords | needs the GPU; can be swapped (`--llm-runtime=llamacpp` or `vllm`) without code changes, since both speak the OpenAI chat API |
| PostgreSQL | reference data, pages, records, crawl runs and batches | durable state; dedup hashes and schedules survive restarts |

### Code layout and layers

```mermaid
flowchart TB
    main["src/main/main.go<br/>wiring only"]
    subgraph app["src/main/app"]
        server["server.go<br/>HTTP server, logging, panic recovery"]
        sched["scheduler.go<br/>due sites every tick"]
        seed["seed.go / settings.go<br/>seed JSON, crawler and taxonomy settings"]
    end
    subgraph infra["src/main/infrastructure"]
        cfg["config, env file, logger"]
        pg["postgres, InTx, migrations"]
        helpers["HTTP helpers, SQL builder,<br/>SHA-256, panic helpers"]
    end
    subgraph mods["src/main/modules"]
        data["data modules<br/>country, organization_type, organization,<br/>category, site, page, record"]
        crawl["crawl<br/>runs, batches, Runner pipeline"]
        crawler["crawler<br/>URL rules, robots.txt, sitemaps,<br/>fetcher, frontier"]
        extractor["extractor<br/>HTML → clean text, filter"]
        llmm["llm<br/>clients, prompt, schema, validator,<br/>Normalizer"]
    end
    res[("src/resources<br/>migrations, seed, settings, env<br/>(embedded)")]

    main --> app
    main --> mods
    main --> infra
    app --> mods
    crawl --> crawler
    crawl --> extractor
    crawl --> llmm
    crawl --> data
    mods --> infra
    seed --> res
```

Every module has the same five folders, and calls only go downwards:

```mermaid
flowchart LR
    h["handler<br/>HTTP: parse query, write JSON"] --> s["service<br/>business rules"] --> r["repo<br/>the only place with SQL"] --> d[("PostgreSQL")]
    dto["dto<br/>request / response shapes"] -.-> h
    m["model<br/>domain types, statuses"] -.-> s
    m -.-> r
```

- **Handlers** never touch SQL; **repos** never contain rules.
- A service that needs another module asks for a small interface (`SiteTracker`, `CountryGetter`, …) instead of the concrete type, so tests use fakes.
- Repos take `infrastructure.DBTX`, which works for both `*sql.DB` and `*sql.Tx`, so the same repo runs inside a transaction (`InTx`).

### Data model

```mermaid
erDiagram
    countries ||--o{ country_translations : "has (uz, ru, kaa)"
    countries ||--o{ organizations : has
    organization_types ||--o{ organizations : classifies
    organizations ||--o{ organizations : "parent of"
    organizations ||--o{ sites : runs
    countries ||--o{ sites : in
    categories ||--o{ categories : "parent of"
    sites ||--o{ pages : contains
    sites ||--o{ crawl_runs : "crawled by"
    crawl_batches ||--o{ crawl_runs : groups
    pages ||--o| records : "normalized into"
    categories ||--o{ records : "category / subcategory"

    countries {
        bigint id PK
        string iso2 UK
        string iso3 UK
        string soato UK
        bool is_active
    }
    organizations {
        bigint id PK
        bigint country_id FK
        bigint organization_type_id FK
        bigint parent_id FK
        string status
    }
    categories {
        bigint id PK
        bigint parent_id FK
        string code UK
        bool is_active
    }
    sites {
        bigint id PK
        string domain UK
        string base_url
        bool is_official
        string status
        bigint crawl_interval_sec
        timestamptz next_crawl_at
        text last_error
    }
    pages {
        bigint id PK
        bigint site_id FK
        bytea url_hash
        bytea content_hash
        string status
        smallint http_status
    }
    crawl_batches {
        bigint id PK
        string status
        int sites_total
        int sites_skipped
    }
    crawl_runs {
        bigint id PK
        bigint site_id FK
        bigint batch_id FK
        string status
        int pages_found
        int pages_failed
    }
    records {
        bigint id PK
        bigint page_id FK
        bigint category_id FK
        bigint subcategory_id FK
        string record_type
        text content
        text summary
        float confidence
    }
```

| Rule | Where it is enforced |
|---|---|
| One record per page | `UNIQUE (page_id)` on `records`; a changed page updates its record |
| One page per URL per site | `UNIQUE (site_id, url_hash)` on `pages` |
| A record always belongs to a real country, organization, site and category | foreign keys, filled from the site, never from the LLM |
| Only two category levels | checked by the seed loader |
| Reference data comes from the seed files | upserted by id (sites by domain) at start; unchanged rows are not rewritten |

### Concurrency

```mermaid
flowchart TB
    api["POST /v1/sites/{id}/crawl"] --> slots
    schedT["Scheduler tick<br/>every SCHEDULER_INTERVAL_SEC"] --> slots
    batch["POST /v1/crawl-batches"] --> cg["one goroutine per country<br/>≤ CRAWL_MAX_SITES_PER_COUNTRY sites each"]
    cg -- "waits for a slot" --> slots

    slots{{"Runner: CRAWL_MAX_SITES slots<br/>never two runs of one site"}}
    slots --> run1["crawl run: site A"]
    slots --> run2["crawl run: site B"]

    subgraph one["inside one crawl run"]
        fw["CRAWLER_WORKERS fetch goroutines<br/>rate limit 2 req/s, robots Crawl-delay"]
        q[["page queue<br/>LLM_WORKERS × 2"]]
        lw["LLM_WORKERS workers"]
        fw -- "new or changed text" --> q --> lw
    end
    run1 --> fw

    lw --> sem{{"Normalizer: LLM_WORKERS slots<br/>shared by ALL runs"}}
    run2 -. "same" .-> sem
    sem --> llmS["LLM server"]
```

| Limit | Setting | Protects |
|---|---|---|
| Sites crawled at once (all sources) | `CRAWL_MAX_SITES` | network, database connections |
| Sites of one country at once (batch) | `CRAWL_MAX_SITES_PER_COUNTRY` | fairness between countries |
| Requests per site | `requests_per_second` in `crawler.json`, robots `Crawl-delay` | the government sites |
| Pages at the LLM at once | `LLM_WORKERS` (global semaphore) | the LLM server; set it to llama-server's `-np` |

A manual trigger or scheduler start that finds no free slot is refused (503 / tried next tick). A batch site waits in the queue instead. When the page queue is full, fetching pauses, so a slow LLM slows the crawl rather than piling up memory.

### Startup and shutdown

```mermaid
sequenceDiagram
    participant M as main
    participant DB as PostgreSQL
    participant L as LLM client
    M->>M: load config (env file + env vars + flags)
    M->>M: load settings (crawler.json, taxonomy.json)
    M->>L: create client (vllm or llamacpp)
    M->>DB: connect (retries for 30 s)
    M->>DB: migrations (advisory lock, schema_migrations)
    M->>DB: seed upserts in one transaction (SEED_ON_START)
    M->>DB: close runs and batches left 'running' as 'failed'
    M->>M: start scheduler (if enabled) and HTTP server
    Note over M: SIGINT / SIGTERM
    M->>M: stop HTTP server (no new crawls)
    M->>M: wait for scheduler, batch, runs<br/>(they end as 'cancelled')
    M->>DB: close
```

---

## 2. Business logic

### Crawl run, end to end

```mermaid
flowchart TD
    start(["crawl run starts<br/>row: running"]) --> prep["load country, organization,<br/>active categories → taxonomy"]
    prep --> robots{"robots.txt"}
    robots -- "5xx or unreachable" --> fail(["run failed<br/>RFC 9309: stop"])
    robots -- "4xx: allow all<br/>200: parse rules" --> seedq["queue base URL<br/>+ sitemap URLs, newest lastmod first"]
    seedq --> front["frontier<br/>≤ max_pages_per_run, depth ≤ max_depth"]
    front --> fetch["fetch with retries on 429 / 5xx,<br/>≤ max_redirects, charset decoding"]
    fetch --> page["handle page<br/>(next diagram)"]
    page -- "links: same domain or www.,<br/>allowed by robots, not skipped ext,<br/>tracking params removed" --> front
    page --> done{"frontier empty?"}
    done -- no --> front
    done -- yes --> wait["wait for LLM workers"]
    wait --> status{"result"}
    status -- "pages_failed = 0" --> c([completed])
    status -- "pages_failed > 0" --> p([partial])
    status -- "app stopped / batch cancelled" --> x([cancelled])
```

### What happens to one page

```mermaid
flowchart TD
    r["fetch result"] --> sk{"skip reason?<br/>redirect off-site,<br/>content type, too large"}
    sk -- yes --> s1["pages_skipped"]
    sk -- no --> er{"fetch error?"}
    er -- yes --> f1["pages_failed<br/>page status: failed"]
    er -- no --> ex["extract clean text<br/>prefer main / article,<br/>drop nav, footer, menus, cookie banners"]
    ex --> obs["upsert page row<br/>(site_id, url_hash)"]
    obs --> fl{"enough text and no noindex?<br/>≥ min_text_chars and<br/>≥ min_text_words"}
    fl -- no --> s2["pages_skipped<br/>status: skipped"]
    fl -- yes --> h["SHA-256 of text"]
    h --> cmp{"compare with<br/>pages.content_hash"}
    cmp -- "same" --> u["pages_unchanged<br/>status: unchanged<br/>NO LLM call"]
    cmp -- "no hash yet" --> n["pages_new"]
    cmp -- "different" --> ch["pages_changed"]
    n --> llmq["LLM queue"]
    ch --> llmq
    llmq --> norm["normalize<br/>(next diagram)"]
    norm -- valid --> tx["one transaction:<br/>upsert record + set content_hash"]
    tx --> rc["records_created / records_updated"]
    norm -- "invalid after retries" --> f2["pages_failed<br/>status: failed<br/>hash NOT saved → retried next crawl"]
```

The hash is saved **only together with the record**. This is what makes the dedup safe: a page whose LLM call failed still has its old hash, so the next crawl sees it as new or changed and tries again.

### LLM normalization

```mermaid
sequenceDiagram
    participant W as LLM worker
    participant N as Normalizer
    participant C as Client (llamacpp / vllm)
    participant S as LLM server
    W->>N: page text, title, URL, country, organization
    N->>N: cut text to LLM_MAX_INPUT_CHARS,<br/>add categories + record types,<br/>build JSON schema with enums
    N->>N: wait for one of LLM_WORKERS slots
    loop up to 1 + LLM_MAX_RETRIES attempts
        N->>C: Normalize
        C->>S: chat/completions, temperature 0,<br/>thinking off, schema
        S-->>C: JSON
        alt HTTP 429/5xx, bad JSON, timeout
            C-->>N: retryable error → wait, retry
        else cut off at max_tokens
            C-->>N: ErrTruncated → retry with omit_content
        else reply
            N->>N: validate against taxonomy
            alt valid
                N-->>W: result (codes mapped to ids)
            else invalid
                N->>N: log "llm output rejected", retry
            end
        end
    end
    N-->>W: error → page failed
```

What the LLM decides and what it does not:

| Decided by the LLM | Fixed by ggkip |
|---|---|
| `record_type`, `category_code`, `subcategory_code` | country, organization, site, page, source URL |
| `title`, `summary`, `keywords`, `language`, `published_at`, `confidence` | content falls back to the extracted text when the LLM gives none |
| `content` (cleaned text) | `llm_runtime`, `llm_model`, `metadata` (headings, run id, change type) |

Taxonomy validation, applied to every reply even when the server enforced the schema:

| Rule | Example of a rejected reply |
|---|---|
| `record_type` must be one of `taxonomy.json` | `"record_type": "ARTICLE"` |
| `category_code` must exist and be top-level | `"FINANCE"`, or `"FIN.TAX"` as category |
| `subcategory_code` must exist, be a subcategory and belong to the category | `"category_code": "HEALTH", "subcategory_code": "FIN.TAX"` |
| `UNKNOWN` cannot have a subcategory | `"UNKNOWN"` + `"FIN.TAX"` |

### Statuses

Crawl run:

```mermaid
stateDiagram-v2
    [*] --> running
    running --> completed: no page failed
    running --> partial: some pages failed
    running --> failed: robots.txt 5xx, setup error, panic
    running --> cancelled: app stopped / batch cancelled
    running --> failed: found 'running' at next start (crash)
```

Page (the outcome of its last crawl):

```mermaid
stateDiagram-v2
    [*] --> new: first record stored
    new --> unchanged: same text
    new --> changed: text changed, record updated
    changed --> unchanged
    unchanged --> changed
    new --> failed
    changed --> failed
    failed --> new: retried, no record yet
    failed --> changed: retried, had a record
    [*] --> skipped: too little text or noindex
    [*] --> failed: fetch or LLM failed
```

Crawl batch:

```mermaid
stateDiagram-v2
    [*] --> running: POST /v1/crawl-batches
    running --> completed: every site tried (some runs may have failed)
    running --> cancelled: POST .../cancel or app stopped
    running --> failed: coordinator panic, or found 'running' at next start
```

### Site scheduling

```mermaid
flowchart LR
    due["scheduler tick:<br/>active, official sites with<br/>next_crawl_at ≤ now"] --> run["crawl run"]
    run -- "completed / partial" --> ok["last_success_at = now<br/>next_crawl_at = now + crawl_interval_sec"]
    run -- failed --> bad["last_error = reason<br/>next_crawl_at = now + min(FAILURE_RETRY, interval)"]
    run -- cancelled --> same["schedule unchanged<br/>(picked up after restart)"]
```

A manual trigger or a batch crawls a site regardless of `next_crawl_at` and updates the schedule the same way afterwards.

### Panic handling

A panic is recovered where it happens, logged with its stack (`grep panicked`), and fails only the smallest unit:

```mermaid
flowchart TB
    http["HTTP handler"] -- panic --> r500["500 reply, server keeps running"]
    tick["scheduler tick"] -- panic --> next["next tick runs normally"]
    bsite["one site of a batch"] -- panic --> bnext["other sites go on"]
    runp["crawl run (setup, robots, sitemaps)"] -- panic --> rf["run failed, error_message = panic: …"]
    fwp["fetch worker / page handler"] -- panic --> pf["that URL failed, worker goes on"]
    lwp["LLM worker"] -- panic --> lf["that page failed, worker goes on"]
    txp["inside a DB transaction"] -- panic --> rb["rolled back, connection returned"]
```

Every level waits for the goroutines it started, so a crawl cannot hang on a dead worker and no slot leaks.

---

## 3. Makefile commands

Run every command from the `ggkip/` folder. `make` with no target prints the list.

### Variables

Set them on the command line, for example `make start ENV=src/resources/env/llamacpp.env`.

| Variable | Default | Used by | Meaning |
|---|---|---|---|
| `ENV` | `src/resources/env/local.env` | `run`, `start` | env file passed to the app as `--env-file` |
| `LOG` | `logs/ggkip.log` | `start`, `logs` | log file of the background app (appended, not overwritten) |
| `API` | `http://localhost:8090` | API shortcuts, `status` | where the running app listens |
| `PG` | `ggkip-pg` | `db-*`, `psql` | name of the PostgreSQL Docker container |
| `COUNTRIES` | empty (all) | `batch` | comma-separated country ids, e.g. `242,115` |
| `ID` | — | `batch-status`, `batch-cancel` | batch id |
| `BATCH`, `SITE` | — | `runs` | filter crawl runs by batch or site id |
| `ARGS` | — | `export` | arguments for `scripts/export_records.py` |

On the GPU server, always pass `ENV=src/resources/env/llamacpp.env`. It points at llama-server on port 8080, seeds only Uzbekistan and turns the scheduler off.

### Build and run

| Command | What it does | When to use it |
|---|---|---|
| `make build` | Compiles `bin/ggkip` (`go build -trimpath`). | After `git pull` or a code change. The other run targets build first anyway. |
| `make run` | Builds, then runs the app in the foreground. Ctrl-C stops it; running crawls end as `cancelled`. | Local development, when you want the log on screen. |
| `make start` | Builds, then starts the app in the background with `nohup`. Saves the process id in `bin/ggkip.pid` and logs to `LOG`. Refuses to start if it is already running. After one second it checks the process is alive; if not, it prints the last 20 log lines. | On the server, so the app keeps running after you close the terminal. |
| `make stop` | Sends SIGTERM and waits up to 30 s, so running crawls close their rows as `cancelled`; only then kills it. Clears a stale pid file whose process is not ggkip. | Before a restart or update. |
| `make restart` | `stop`, then `start` (which rebuilds). | After `git pull`. |
| `make status` | Says whether the process is running and calls `/health`. | Quick check that everything is up. |
| `make logs` | `tail -f` on the log file. Ctrl-C stops watching only. | Following a crawl. |

`make start` and `make stop` manage only the process recorded in `bin/ggkip.pid`. If you started ggkip by hand with `nohup ./bin/ggkip ...`, stop that one yourself first (`pkill -f 'bin/ggkip --env-file'`).

### Quality checks

| Command | What it does |
|---|---|
| `make test` | All unit tests (`go test ./...`). They use fake HTTP servers and a fake LLM; no database, network or GPU needed. |
| `make race` | The same tests with the race detector. This matters here because crawls, LLM workers and batches run in many goroutines. |
| `make vet` | `go vet ./...`: static checks, and it compiles the test files too. |
| `make fmt` | Formats all Go files (`gofmt -w`). |
| `make fmt-check` | Fails and lists the files that need formatting. |
| `make check` | `fmt-check` + `vet` + `race` + `build`: what every commit should pass. |
| `make clean` | Removes `bin/` (binary and pid file). |

### Database (Docker only)

These need Docker, so they work on a laptop. They don't work on the GPU server, which runs PostgreSQL without Docker.

| Command | What it does |
|---|---|
| `make db-up` | Starts PostgreSQL 17 in a container: user, password and database are all `ggkip`, on port 5432. Reuses an existing container. |
| `make db-down` | Stops the container. The data stays. |
| `make db-logs` | Follows the PostgreSQL log. |
| `make psql` | Opens a `psql` shell inside the container. |

On the server, use `psql "postgres://ggkip:ggkip@localhost:5432/ggkip"` instead of `make psql`.

### API shortcuts

They call the running app and print the reply as indented JSON.

| Command | Calls | What it does |
|---|---|---|
| `make health` | `GET /health` | App and database status. |
| `make batch` | `POST /v1/crawl-batches` | Crawls every active official site of every active country. |
| `make batch COUNTRIES=242` | same, with `{"country_ids":[242]}` | Only Uzbekistan (242). |
| `make batches` | `GET /v1/crawl-batches` | Lists batches, newest first. |
| `make batch-status ID=1` | `GET /v1/crawl-batches/1` | Runs per status, sites pending and skipped. |
| `make batch-cancel ID=1` | `POST /v1/crawl-batches/1/cancel` | Stops the batch. |
| `make runs` | `GET /v1/crawl-runs?limit=50` | Latest crawl runs. Add `BATCH=1` or `SITE=8` to filter. |
| `make export ARGS="--site-id 8 --out stat_uz.json"` | `scripts/export_records.py` | Saves records to a JSON file with category codes, organization and domain names and the full content. `ARGS="--help"` lists the filters. |

### Typical session on the GPU server

```sh
cd /workspace/scientific-research && git pull && cd ggkip
make restart ENV=src/resources/env/llamacpp.env
make status
make batch COUNTRIES=242
make batch-status ID=1          # repeat until status is not "running"
make runs BATCH=1
make export ARGS="--country-id 242 --out /workspace/exports/uz.json"
```

---

## 4. HTTP API

Base URL: `http://localhost:8090`. All replies are JSON.

### Common rules

| Rule | Detail |
|---|---|
| Lists | `{"items": [...], "next_cursor": N}`. Pass `?cursor=N` for the next page; `next_cursor` is `null` on the last page. `?limit=` sets the page size. |
| Order | Records, pages, crawl runs and batches: newest first. Reference data: ascending by id. |
| Errors | `{"error": "..."}` with **400** for bad input, **404** when not found and **500** for an internal error. A panic in a handler is recovered and returns 500; the app keeps running. |
| Background work | Crawl endpoints return **202** at once. The work continues in the background; follow it with the GET endpoints. |

### Health

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Pings PostgreSQL and shows the LLM runtime (`vllm` or `llamacpp`). |

### Reference data (loaded from the seed files)

| Method | Endpoint | Description | Parameters |
|---|---|---|---|
| GET | `/v1/countries` | Countries (250) | `?region`, `?is_active`, `?locale=uz\|ru\|kaa` (returns the translated name) |
| GET | `/v1/countries/{id}` | One country with its uz/ru/kaa translations | — |
| GET | `/v1/organization-types` | Types: ministry, agency, committee, … | — |
| GET | `/v1/organizations` | Government organizations | `?country_id`, `?organization_type_id`, `?parent_id`, `?status` |
| GET | `/v1/organizations/{id}` | One organization | — |
| GET | `/v1/categories` | Categories and subcategories (the taxonomy the LLM must use) | `?parent_id`, `?top_level=true`, `?tree=true` (subcategories nested under parents) |
| GET | `/v1/categories/{id}` | One category | — |
| GET | `/v1/sites` | Official websites, with crawl schedule (`next_crawl_at`) and last error | `?country_id`, `?organization_id`, `?status` |
| GET | `/v1/sites/{id}` | One site | — |

Uzbekistan's country id is **242**. To find a site id: `curl -s "localhost:8090/v1/sites?country_id=242"`.

### Crawling one site

| Method | Endpoint | Description | Replies |
|---|---|---|---|
| POST | `/v1/sites/{id}/crawl` | Crawls this site now, ignoring `next_crawl_at`. | **202** run created · **409** this site is already being crawled · **503** all `CRAWL_MAX_SITES` slots are busy (try later) |
| GET | `/v1/crawl-runs` | Crawl runs, newest first | `?site_id`, `?batch_id`, `?status` |
| GET | `/v1/crawl-runs/{id}` | One run | — |

The fields of a crawl run:

| Field | Meaning |
|---|---|
| `status` | `running`; `completed` (no failures); `partial` (some pages failed); `failed` (the crawl itself failed, see `error_message`); `cancelled` (app stopped or batch cancelled) |
| `pages_found` | URLs fetched (limited by `max_pages_per_run` in `crawler.json`) |
| `pages_new` / `pages_changed` | Pages sent to the LLM: first time seen / text changed since the last record |
| `pages_unchanged` | Same text as before: no LLM call (the dedup) |
| `pages_skipped` | Too little text (under `min_text_chars` / `min_text_words`) or not an allowed content type |
| `pages_failed` | Download failed, or the LLM answer was invalid after all retries |
| `records_created` / `records_updated` | Records stored |
| `batch_id` | The batch the run belongs to, or `null` |

### Crawling all active countries (batches)

| Method | Endpoint | Description | Replies |
|---|---|---|---|
| POST | `/v1/crawl-batches` | Crawls every active official site of every active country. Optional body `{"country_ids": [242]}` limits it to those countries. | **202** batch started · **409** another batch is running · **400** no matching sites |
| GET | `/v1/crawl-batches` | Batches, newest first | `?status` |
| GET | `/v1/crawl-batches/{id}` | One batch; adds `runs` (crawl runs per status) and `sites_pending` | — |
| POST | `/v1/crawl-batches/{id}/cancel` | Sites still waiting are not started; running crawls end `cancelled` | **202** cancelling · **409** already finished |

How a batch runs:

```text
POST /v1/crawl-batches
  ├── goroutine: country UZ ──► its sites, up to CRAWL_MAX_SITES_PER_COUNTRY at once
  ├── goroutine: country JP ──► …
  └── goroutine: country GB ──► …
        every site also waits for one of the CRAWL_MAX_SITES global slots
        every page waits for one of the LLM_WORKERS LLM slots (shared by all sites)
```

- **Per-country limit:** it keeps one country with many sites from taking every slot.
- **One batch at a time:** a second `POST` returns 409.
- **Skipped sites:** a site already being crawled (by the scheduler or a manual trigger) is skipped and counted in `sites_skipped`.
- **Status:** a batch is `completed` once every site was tried, even if some runs failed; the per-site outcome is in `runs`. Stopping the app ends it as `cancelled`. A batch left `running` by a crash is closed as `failed` at the next start.

### Results

| Method | Endpoint | Description | Parameters |
|---|---|---|---|
| GET | `/v1/pages` | Every URL seen, with HTTP status and the outcome of its last crawl | `?site_id`, `?status` (`new`, `changed`, `unchanged`, `skipped`, `failed`) |
| GET | `/v1/records` | The normalized records, without the long content | `?country_id`, `?organization_id`, `?site_id`, `?category_id`, `?subcategory_id`, `?record_type`, `?language`, `?published_from`, `?published_to` (dates `YYYY-MM-DD`) |
| GET | `/v1/records/{id}` | One record with full `content` and `metadata` | — |

Where the record fields come from:

| Field | Source |
|---|---|
| `record_type` | LLM: one of the types in `taxonomy.json` |
| `category_id`, `subcategory_id` | LLM returns a code (e.g. `FIN.TAX`); ggkip checks it exists and maps it to the id |
| `title`, `summary`, `keywords`, `confidence` | LLM |
| `content` | LLM; if empty (e.g. after a `max_tokens` cut-off) the text extracted from the page |
| `language_code` | LLM; if empty, the page's `lang` attribute |
| `published_at` | LLM, read from the page text |
| `llm_runtime`, `llm_model` | which server and model produced the record |
| `country_id`, `organization_id`, `site_id`, `page_id`, `source_url`, `metadata` | not from the LLM: ggkip takes them from the site and the page |

An LLM answer that breaks the taxonomy (unknown code, a subcategory of another category, `UNKNOWN` with a subcategory) is retried. If it never passes, the page is marked `failed` and no record is stored.

---

## 5. Server sizing

### Summary

The LLM does about 95% of the work, so the GPU decides the server size. The Go app and PostgreSQL are light. The B200 used for testing (174 GB of GPU memory) is far bigger than the current setup needs.

Numbers marked *measured* come from the test server. Speeds marked *estimate* are typical values for this model size; check them with the commands in [Measuring on your server](#measuring-on-your-server).

### What each part needs

| Part | Work | CPU | RAM | Disk | GPU |
|---|---|---|---|---|---|
| ggkip (Go) | downloads pages, parses HTML, hashes text, calls the LLM over HTTP, writes to PostgreSQL | 2–4 cores | 0.5–1 GB | ~50 MB | — |
| PostgreSQL | pages, records, crawl runs and batches | 2 cores | 2–4 GB | grows with the data (below) | — |
| LLM server (llama.cpp or vLLM) | one call per new or changed page | 4–8 cores | 16–32 GB, for loading the model | 10–20 GB per model | **main cost** |

Why ggkip is light:
- **Fetch rate:** each site is limited to `requests_per_second: 2`, so at `CRAWL_MAX_SITES=8` the app fetches at most about 16 pages a second. Parsing a page takes milliseconds.
- **Memory:** the worst case is 8 sites × 4 workers × 5 MB (`CRAWLER_MAX_BODY_BYTES`), about 160 MB.

### GPU memory

A model on the GPU needs room for two things:
- **Weights:** the model itself. Fixed size.
- **Context memory (KV cache):** for every token of every request being processed. For Qwen3-8B that is about 144 KB per token (36 layers × 8 KV heads × 128 dims × 2 for keys and values × 2 bytes).

Each ggkip request is about 7–9k tokens of input (the system prompt, the 60 categories with descriptions and up to 12,000 characters of page text) plus up to 6k tokens of output. So every parallel request needs a slot of about 16k tokens, which is **~2.4 GB** of context memory.

Check against the test server (*measured*: llama-server used 17.7 GB):

| Item | Size |
|---|---|
| Qwen3-8B Q8_0 weights | ~8.7 GB |
| Context memory: `-c 65536` × 144 KB | ~9.4 GB |
| **Total** | **~18 GB** ✓ |

| Setup | Weights | Context (16k per request) | GPU memory |
|---|---|---|---|
| llama.cpp, Qwen3-8B Q8, 4 parallel (current) | 8.7 GB | 4 × 2.4 GB | ~18 GB |
| vLLM, Qwen3-8B bf16, 8 parallel | 16 GB | 8 × 2.4 GB | ~36 GB |
| vLLM, Qwen3-8B bf16, 16 parallel | 16 GB | 16 × 2.4 GB | ~55 GB |
| vLLM, Qwen3-30B-A3B bf16, 16 parallel | 61 GB | 16 × ~1.5 GB | ~85 GB |

More parallel requests means more pages per hour (next section). Memory grows linearly with the number of parallel requests, which is why the production rows need 48–80 GB.

### Speed

The model reads the input fast. Writing the output is what takes time, because output tokens are produced one after another.

The current prompt asks the LLM to rewrite the whole page into `content`, which is 1–4k output tokens per page. This is also why 5 of the first 23 `stat.uz` pages failed with `cut off at max_tokens` before the limit was raised. *Measured*: the first crawl of `stat.uz` handled 23 LLM pages in about 3 minutes with llama.cpp and 4 slots.

| Runtime | Parallel requests | Output speed, all requests together (*estimate*) | Pages per hour at ~2k output tokens |
|---|---|---|---|
| llama.cpp, `-np 4` | 4 | ~400–600 tokens/s | ~800–1,000 |
| vLLM | 16–32 | ~3,000–6,000 tokens/s | ~6,000–10,000 |

Why vLLM is faster on the same GPU: it batches many requests into each GPU step (continuous batching) and manages context memory in pages, so it can keep 16–32 requests busy at once. llama.cpp is simpler and works well for tests, but it gains less from more parallel requests.

**Biggest improvement for the cost:** stop asking the LLM to write `content`, and store the extracted page text, which ggkip already has. The output then shrinks to about 200–400 tokens (category, title, summary, keywords), making each page **5–10× faster** on the same GPU and ending the `max_tokens` failures. This is a small code change that hasn't been made yet.

### How much work the project creates

A page goes to the LLM only when it is new or its text changed. Unchanged pages are skipped by their SHA-256 hash, so the first full crawl is expensive and later crawls are cheap.

| Scale | First full crawl | Then every 6 hours (~5% of pages change) |
|---|---|---|
| Current seed: 26 sites × 500 pages | ~13k LLM calls → ~13 h with llama.cpp, ~1.5–2 h with vLLM | ~650 calls → minutes |
| 300 sites (about 10 countries) | ~150k calls → about a day with vLLM | ~7.5k calls → ~1 h |
| 300 sites, LLM without `content` | ~150k calls → ~3–4 h with vLLM | minutes |

### Database size

| Data | Per row | 100k pages and records | 1M |
|---|---|---|---|
| `records` (content, summary, keywords, metadata) | ~10–25 KB | ~2.5 GB | ~25 GB |
| `pages`, `crawl_runs`, indexes | ~1 KB | ~0.1 GB | ~1–2 GB |
| **Total with indexes** | | **~3–5 GB** | **~30–50 GB** |

Leave room for backups and the model files (10–20 GB each) when choosing the disk.

### Recommended servers

| Stage | GPU | CPU | RAM | Disk | LLM setup | Why |
|---|---|---|---|---|---|---|
| **Test / MVP**: 1–3 countries, tens of sites | 1× 24 GB (L4, A10, RTX 4090) | 8 vCPU | 32 GB | 200 GB SSD | llama.cpp, Qwen3-8B Q8, `-np 4` | 18 GB of GPU memory fits; daily updates take minutes |
| **Production**: 10+ countries, hundreds of sites | 1× L40S 48 GB or 1× H100 80 GB | 16 vCPU | 64 GB | 500 GB NVMe | vLLM, Qwen3-8B bf16, 16–32 parallel | the first full crawl of ~150k pages finishes in about a day, not a week |
| **Higher quality** | 1× H100 80 GB, H200 or B200 | 16–32 vCPU | 128 GB | 1 TB NVMe | vLLM, Qwen3-30B-A3B | a larger model picks categories more accurately; it needs ~85 GB |

The current B200 fits even the last row and could run the 8B and 30B models side by side to compare their categories.

### Network and location

- **Location:** many `.uz` government sites block or slow down foreign IP addresses. About 4 of the first 30 `stat.uz` pages failed to download. A server in Uzbekistan, or one with an Uzbek address, will have fewer download failures.
- **Bandwidth:** small. 8 sites × 2 requests/s × ~200 KB is about 3 MB/s at peak.
- **Ports:** the LLM server (8080 or 8000) and PostgreSQL (5432) should listen on `127.0.0.1` only. Only the ggkip API (8090) needs to be reachable, and only from your own network.

### Measuring on your server

llama-server prints its speed after every request:

```sh
grep "eval time" /workspace/logs/llama-server.log | tail -5
```

`prompt eval time` is the input speed and `eval time` the output speed, both in tokens per second. To see the time per page in ggkip, set `LOG_LEVEL=debug` in the env file, restart, and run:

```sh
grep "llm normalized page" logs/ggkip.log | tail -5    # duration= per page
```

To check GPU memory and use while a crawl runs:

```sh
watch -n 1 nvidia-smi
```
