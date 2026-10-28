-- ============================================================
-- 9. RECORDS
-- ============================================================

CREATE TABLE records (
    id                  BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,

    country_id          BIGINT NOT NULL
                        REFERENCES countries(id),

    organization_id     BIGINT NOT NULL
                        REFERENCES organizations(id),

    site_id             BIGINT NOT NULL
                        REFERENCES sites(id),

    page_id             BIGINT NOT NULL
                        REFERENCES pages(id),

    category_id         BIGINT NOT NULL
                        REFERENCES categories(id),

    subcategory_id      BIGINT
                        REFERENCES categories(id),

    record_type         VARCHAR(50) NOT NULL,

    title               TEXT,

    language_code       VARCHAR(10),

    published_at        TIMESTAMPTZ,

    content             TEXT NOT NULL,

    summary             TEXT,

    keywords            JSONB NOT NULL DEFAULT '[]'::jsonb,

    metadata            JSONB NOT NULL DEFAULT '{}'::jsonb,

    confidence          NUMERIC(5,4),

    content_hash        BYTEA NOT NULL,

    llm_runtime         VARCHAR(30),

    llm_model           VARCHAR(200),

    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    UNIQUE (page_id)
);

-- Main query pattern: Country -> Category -> Subcategory (also serves country_id alone).
CREATE INDEX idx_records_country_category ON records (country_id, category_id, subcategory_id);

CREATE INDEX idx_records_organization_id ON records (organization_id);
CREATE INDEX idx_records_category_id     ON records (category_id);
CREATE INDEX idx_records_subcategory_id  ON records (subcategory_id);
CREATE INDEX idx_records_content_hash    ON records (content_hash);
CREATE INDEX idx_records_published_at    ON records (published_at);
