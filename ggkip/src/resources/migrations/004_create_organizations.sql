-- ============================================================
-- 4. ORGANIZATIONS
-- ============================================================

CREATE TABLE organizations (
    id                      BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,

    country_id              BIGINT NOT NULL
                            REFERENCES countries(id),

    organization_type_id    BIGINT NOT NULL
                            REFERENCES organization_types(id),

    parent_id               BIGINT
                            REFERENCES organizations(id),

    name                    VARCHAR(300) NOT NULL,
    official_name           VARCHAR(500),

    description             TEXT,

    status                  VARCHAR(20) NOT NULL DEFAULT 'active',

    created_at              TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    UNIQUE (country_id, name)       -- also serves lookups by country_id
);

CREATE INDEX idx_organizations_parent_id ON organizations (parent_id);
