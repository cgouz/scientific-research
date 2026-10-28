-- ============================================================
-- 10. CRAWL BATCHES
-- ============================================================
-- One POST /v1/crawl-batches: every active official site of the active
-- countries (or of the countries in country_ids) is crawled, each country
-- in its own goroutine. Per-site results are the crawl_runs with this batch_id.

CREATE TABLE crawl_batches (
    id                  BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,

    status              VARCHAR(20) NOT NULL DEFAULT 'running',

    -- the request filter; empty = all active countries
    country_ids         BIGINT[] NOT NULL DEFAULT '{}',

    countries_total     INTEGER NOT NULL DEFAULT 0,
    sites_total         INTEGER NOT NULL DEFAULT 0,
    -- sites not crawled because they were already running (scheduler, manual trigger)
    sites_skipped       INTEGER NOT NULL DEFAULT 0,

    started_at          TIMESTAMPTZ NOT NULL,
    finished_at         TIMESTAMPTZ,

    error_message       TEXT,

    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

ALTER TABLE crawl_runs
    ADD COLUMN batch_id BIGINT REFERENCES crawl_batches(id) ON DELETE SET NULL;

CREATE INDEX idx_crawl_runs_batch_id ON crawl_runs (batch_id, status) WHERE batch_id IS NOT NULL;
