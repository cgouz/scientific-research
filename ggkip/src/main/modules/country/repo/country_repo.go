package repo

import (
	"context"
	"database/sql"
	"errors"
	"fmt"

	"ggkip/src/main/infrastructure"
	"ggkip/src/main/modules/country/model"
)

type Repo struct {
	db infrastructure.DBTX
}

func New(db infrastructure.DBTX) *Repo {
	return &Repo{db: db}
}

const columns = `c.id, c.iso2, c.iso3, c.name, c.soato, c.region, c.subregion, c.is_active, c.created_at, c.updated_at`

func scan(row interface{ Scan(...any) error }) (model.Country, error) {
	var c model.Country
	err := row.Scan(&c.ID, &c.ISO2, &c.ISO3, &c.Name, &c.Soato, &c.Region, &c.Subregion,
		&c.IsActive, &c.CreatedAt, &c.UpdatedAt)
	return c, err
}

// List returns up to f.Limit+1 countries ordered by id, after f.Cursor.
func (r *Repo) List(ctx context.Context, f model.ListFilter) ([]model.Country, error) {
	var w infrastructure.Where
	from := `countries c`
	cols := columns
	if f.Locale != nil {
		from += ` LEFT JOIN country_translations t ON t.country_id = c.id AND t.locale = ` + w.Arg(*f.Locale)
		cols = `c.id, c.iso2, c.iso3, COALESCE(t.name, c.name), c.soato, c.region, c.subregion, c.is_active, c.created_at, c.updated_at`
	}
	w.Add("c.id > ?", f.Cursor)
	if f.Region != nil {
		w.Add("c.region = ?", *f.Region)
	}
	if f.IsActive != nil {
		w.Add("c.is_active = ?", *f.IsActive)
	}
	q := `SELECT ` + cols + ` FROM ` + from + w.SQL() + ` ORDER BY c.id LIMIT ` + w.Arg(f.Limit+1)

	rows, err := r.db.QueryContext(ctx, q, w.Args...)
	if err != nil {
		return nil, fmt.Errorf("list countries: %w", err)
	}
	defer rows.Close()
	var out []model.Country
	for rows.Next() {
		c, err := scan(rows)
		if err != nil {
			return nil, err
		}
		out = append(out, c)
	}
	return out, rows.Err()
}

func (r *Repo) Get(ctx context.Context, id int64) (model.Country, error) {
	c, err := scan(r.db.QueryRowContext(ctx, `SELECT `+columns+` FROM countries c WHERE c.id = $1`, id))
	if errors.Is(err, sql.ErrNoRows) {
		return c, fmt.Errorf("country %d: %w", id, infrastructure.ErrNotFound)
	}
	return c, err
}

func (r *Repo) Translations(ctx context.Context, countryID int64) ([]model.Translation, error) {
	rows, err := r.db.QueryContext(ctx,
		`SELECT id, country_id, locale, name FROM country_translations WHERE country_id = $1 ORDER BY locale`, countryID)
	if err != nil {
		return nil, fmt.Errorf("country translations: %w", err)
	}
	defer rows.Close()
	var out []model.Translation
	for rows.Next() {
		var t model.Translation
		if err := rows.Scan(&t.ID, &t.CountryID, &t.Locale, &t.Name); err != nil {
			return nil, err
		}
		out = append(out, t)
	}
	return out, rows.Err()
}

// UpsertSeed inserts or updates countries by id. Rows that did not change are
// left untouched, so running the seed again does not bump updated_at.
func (r *Repo) UpsertSeed(ctx context.Context, items []model.Country) error {
	for _, c := range items {
		_, err := r.db.ExecContext(ctx, `
			INSERT INTO countries (id, iso2, iso3, name, soato, region, subregion, is_active)
			OVERRIDING SYSTEM VALUE
			VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
			ON CONFLICT (id) DO UPDATE SET
				iso2 = EXCLUDED.iso2, iso3 = EXCLUDED.iso3, name = EXCLUDED.name, soato = EXCLUDED.soato,
				region = EXCLUDED.region, subregion = EXCLUDED.subregion, is_active = EXCLUDED.is_active,
				updated_at = NOW()
			WHERE (countries.iso2, countries.iso3, countries.name, countries.soato, countries.region,
			       countries.subregion, countries.is_active)
			      IS DISTINCT FROM
			      (EXCLUDED.iso2, EXCLUDED.iso3, EXCLUDED.name, EXCLUDED.soato, EXCLUDED.region,
			       EXCLUDED.subregion, EXCLUDED.is_active)`,
			c.ID, c.ISO2, c.ISO3, c.Name, c.Soato, c.Region, c.Subregion, c.IsActive)
		if err != nil {
			return fmt.Errorf("seed country %d (%s): %w", c.ID, c.ISO2, err)
		}
	}
	return infrastructure.ResetIdentity(ctx, r.db, "countries")
}

func (r *Repo) UpsertTranslationsSeed(ctx context.Context, items []model.Translation) error {
	for _, t := range items {
		_, err := r.db.ExecContext(ctx, `
			INSERT INTO country_translations (id, country_id, locale, name)
			OVERRIDING SYSTEM VALUE
			VALUES ($1, $2, $3, $4)
			ON CONFLICT (id) DO UPDATE SET
				country_id = EXCLUDED.country_id, locale = EXCLUDED.locale, name = EXCLUDED.name,
				updated_at = NOW()
			WHERE (country_translations.country_id, country_translations.locale, country_translations.name)
			      IS DISTINCT FROM (EXCLUDED.country_id, EXCLUDED.locale, EXCLUDED.name)`,
			t.ID, t.CountryID, t.Locale, t.Name)
		if err != nil {
			return fmt.Errorf("seed country translation %d: %w", t.ID, err)
		}
	}
	return infrastructure.ResetIdentity(ctx, r.db, "country_translations")
}
