-- ============================================================
-- 5. CATEGORIES
-- ============================================================

CREATE TABLE categories (
    id              BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,

    parent_id       BIGINT
                    REFERENCES categories(id),

    code            VARCHAR(100) NOT NULL UNIQUE,

    name            VARCHAR(200) NOT NULL,

    description     TEXT,

    is_active       BOOLEAN NOT NULL DEFAULT TRUE,

    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_categories_parent_id ON categories (parent_id);
