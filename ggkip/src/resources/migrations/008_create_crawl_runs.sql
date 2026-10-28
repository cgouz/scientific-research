-- ============================================================
-- 8. CRAWL RUNS
-- ============================================================

CREATE TABLE crawl_runs (
    id                  BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,

    site_id             BIGINT NOT NULL
                        REFERENCES sites(id),

    status              VARCHAR(20) NOT NULL DEFAULT 'running',

    started_at          TIMESTAMPTZ,

    finished_at         TIMESTAMPTZ,

    pages_found         BIGINT NOT NULL DEFAULT 0,
    pages_new           BIGINT NOT NULL DEFAULT 0,
    pages_changed       BIGINT NOT NULL DEFAULT 0,
    pages_unchanged     BIGINT NOT NULL DEFAULT 0,
    pages_skipped       BIGINT NOT NULL DEFAULT 0,
    pages_failed        BIGINT NOT NULL DEFAULT 0,

    records_created     BIGINT NOT NULL DEFAULT 0,
    records_updated     BIGINT NOT NULL DEFAULT 0,

    error_message       TEXT,

    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_crawl_runs_site_id ON crawl_runs (site_id, id DESC);
