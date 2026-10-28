-- ============================================================
-- 7. PAGES
-- ============================================================

-- Answers only: have we seen this URL, and has its text changed?
CREATE TABLE pages (
    id                  BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,

    site_id             BIGINT NOT NULL
                        REFERENCES sites(id),

    url                 TEXT NOT NULL,
    url_hash            BYTEA NOT NULL,             -- SHA-256(normalized URL)

    content_hash        BYTEA,                      -- SHA-256(clean text), set once a record is stored

    status              VARCHAR(20) NOT NULL DEFAULT 'new',

    http_status         SMALLINT,
    language_code       VARCHAR(10),

    first_seen_at       TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_seen_at        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_crawled_at     TIMESTAMPTZ,

    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    UNIQUE (site_id, url_hash)      -- also serves lookups by site_id
);

CREATE INDEX idx_pages_url_hash     ON pages (url_hash);
CREATE INDEX idx_pages_content_hash ON pages (content_hash);
