-- ============================================================
-- 1. COUNTRIES
-- ============================================================

CREATE TABLE countries (
    id              BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,

    iso2            CHAR(2) NOT NULL UNIQUE,
    iso3            CHAR(3) NOT NULL UNIQUE,

    name            VARCHAR(150) NOT NULL,
    soato           VARCHAR(10) UNIQUE,

    region          VARCHAR(100),
    subregion       VARCHAR(100),

    is_active       BOOLEAN NOT NULL DEFAULT TRUE,

    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
