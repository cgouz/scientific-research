-- ============================================================
-- 2. COUNTRY TRANSLATIONS
-- ============================================================

CREATE TABLE country_translations (
    id              BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,

    country_id      BIGINT NOT NULL REFERENCES countries(id) ON DELETE CASCADE,
    locale          VARCHAR(5) NOT NULL,            -- uz, ru, kaa

    name            VARCHAR(150) NOT NULL,

    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    UNIQUE (country_id, locale)
);
