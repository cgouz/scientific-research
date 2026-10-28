-- ============================================================
-- 6. SITES
-- ============================================================

CREATE TABLE sites (
    id                  BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,

    country_id          BIGINT NOT NULL
                        REFERENCES countries(id),

    organization_id     BIGINT NOT NULL
                        REFERENCES organizations(id),

    domain              VARCHAR(255) NOT NULL UNIQUE,

    base_url            TEXT NOT NULL,

    is_official         BOOLEAN NOT NULL DEFAULT FALSE,

    verification_score  NUMERIC(5,4) NOT NULL DEFAULT 0,

    status              VARCHAR(20) NOT NULL DEFAULT 'active',

    crawl_interval_sec  BIGINT NOT NULL DEFAULT 86400
                        CHECK (crawl_interval_sec > 0),

    last_crawled_at     TIMESTAMPTZ,

    next_crawl_at       TIMESTAMPTZ,

    last_success_at     TIMESTAMPTZ,

    last_error_at       TIMESTAMPTZ,

    last_error          TEXT,

    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_sites_country_id      ON sites (country_id);
CREATE INDEX idx_sites_organization_id ON sites (organization_id);

-- Scheduler: due official active sites.
CREATE INDEX idx_sites_next_crawl_at   ON sites (next_crawl_at)
    WHERE status = 'active' AND is_official = TRUE;
