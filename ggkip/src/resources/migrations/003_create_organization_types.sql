-- ============================================================
-- 3. ORGANIZATION TYPES
-- ============================================================

CREATE TABLE organization_types (
    id              BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,

    code            VARCHAR(50) NOT NULL UNIQUE,
    name            VARCHAR(100) NOT NULL,

    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
