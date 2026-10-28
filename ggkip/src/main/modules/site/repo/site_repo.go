package repo

import (
	"context"
	"database/sql"
	"errors"
	"fmt"
	"strings"
	"time"

	"github.com/lib/pq"

	"ggkip/src/main/infrastructure"
	"ggkip/src/main/modules/site/model"
)

type Repo struct {
	db infrastructure.DBTX
}

func New(db infrastructure.DBTX) *Repo {
	return &Repo{db: db}
}

const columns = `id, country_id, organization_id, domain, base_url, is_official, verification_score, status,
	crawl_interval_sec, last_crawled_at, next_crawl_at, last_success_at, last_error_at, last_error, created_at, updated_at`

func scan(row interface{ Scan(...any) error }) (model.Site, error) {
	var s model.Site
	err := row.Scan(&s.ID, &s.CountryID, &s.OrganizationID, &s.Domain, &s.BaseURL, &s.IsOfficial,
		&s.VerificationScore, &s.Status, &s.CrawlIntervalSec, &s.LastCrawledAt, &s.NextCrawlAt,
		&s.LastSuccessAt, &s.LastErrorAt, &s.LastError, &s.CreatedAt, &s.UpdatedAt)
	return s, err
}

func (r *Repo) query(ctx context.Context, q string, args ...any) ([]model.Site, error) {
	rows, err := r.db.QueryContext(ctx, q, args...)
	if err != nil {
		return nil, fmt.Errorf("query sites: %w", err)
	}
	defer rows.Close()
	var out []model.Site
	for rows.Next() {
		s, err := scan(rows)
		if err != nil {
			return nil, err
		}
		out = append(out, s)
	}
	return out, rows.Err()
}

func (r *Repo) List(ctx context.Context, f model.ListFilter) ([]model.Site, error) {
	var w infrastructure.Where
	w.Add("id > ?", f.Cursor)
	if f.CountryID != nil {
		w.Add("country_id = ?", *f.CountryID)
	}
	if f.OrganizationID != nil {
		w.Add("organization_id = ?", *f.OrganizationID)
	}
	if f.Status != nil {
		w.Add("status = ?", *f.Status)
	}
	return r.query(ctx, `SELECT `+columns+` FROM sites`+w.SQL()+` ORDER BY id LIMIT `+w.Arg(f.Limit+1), w.Args...)
}

func (r *Repo) Get(ctx context.Context, id int64) (model.Site, error) {
	s, err := scan(r.db.QueryRowContext(ctx, `SELECT `+columns+` FROM sites WHERE id = $1`, id))
	if errors.Is(err, sql.ErrNoRows) {
		return s, fmt.Errorf("site %d: %w", id, infrastructure.ErrNotFound)
	}
	return s, err
}

// Due returns active official sites whose next crawl time has passed (or was
// never set), oldest first.
func (r *Repo) Due(ctx context.Context, now time.Time, limit int) ([]model.Site, error) {
	return r.query(ctx, `
		SELECT `+columns+` FROM sites
		WHERE status = 'active'
		  AND is_official = TRUE
		  AND (next_crawl_at IS NULL OR next_crawl_at <= $1)
		ORDER BY next_crawl_at NULLS FIRST, id
		LIMIT $2`, now, limit)
}

// Crawlable returns active official sites whose country is active, by country
// then id, regardless of next_crawl_at. countryIDs narrows it (empty = all).
func (r *Repo) Crawlable(ctx context.Context, countryIDs []int64) ([]model.Site, error) {
	q := `
		SELECT ` + siteColumns("s.") + ` FROM sites s
		JOIN countries c ON c.id = s.country_id
		WHERE s.status = 'active'
		  AND s.is_official = TRUE
		  AND c.is_active = TRUE`
	var args []any
	if len(countryIDs) > 0 {
		q += ` AND s.country_id = ANY($1)`
		args = append(args, pq.Array(countryIDs))
	}
	return r.query(ctx, q+` ORDER BY s.country_id, s.id`, args...)
}

// siteColumns is columns with every name prefixed (for queries with a join).
func siteColumns(prefix string) string {
	names := strings.Split(columns, ",")
	for i, n := range names {
		names[i] = prefix + strings.TrimSpace(n)
	}
	return strings.Join(names, ", ")
}

func (r *Repo) MarkSuccess(ctx context.Context, id int64, at, next time.Time) error {
	_, err := r.db.ExecContext(ctx, `
		UPDATE sites SET last_crawled_at = $2, last_success_at = $2, next_crawl_at = $3, updated_at = NOW()
		WHERE id = $1`, id, at, next)
	if err != nil {
		return fmt.Errorf("mark site %d success: %w", id, err)
	}
	return nil
}

func (r *Repo) MarkFailure(ctx context.Context, id int64, at, next time.Time, msg string) error {
	_, err := r.db.ExecContext(ctx, `
		UPDATE sites SET last_crawled_at = $2, last_error_at = $2, last_error = $3, next_crawl_at = $4, updated_at = NOW()
		WHERE id = $1`, id, at, msg, next)
	if err != nil {
		return fmt.Errorf("mark site %d failure: %w", id, err)
	}
	return nil
}

// UpsertSeed inserts or updates sites by domain. Crawl state columns
// (last_*/next_crawl_at) are never touched by seeding.
func (r *Repo) UpsertSeed(ctx context.Context, items []model.Site) error {
	for _, s := range items {
		_, err := r.db.ExecContext(ctx, `
			INSERT INTO sites (country_id, organization_id, domain, base_url, is_official, verification_score,
			                   status, crawl_interval_sec)
			VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
			ON CONFLICT (domain) DO UPDATE SET
				country_id = EXCLUDED.country_id, organization_id = EXCLUDED.organization_id,
				base_url = EXCLUDED.base_url, is_official = EXCLUDED.is_official,
				verification_score = EXCLUDED.verification_score, status = EXCLUDED.status,
				crawl_interval_sec = EXCLUDED.crawl_interval_sec, updated_at = NOW()
			WHERE (sites.country_id, sites.organization_id, sites.base_url, sites.is_official,
			       sites.verification_score, sites.status, sites.crawl_interval_sec)
			      IS DISTINCT FROM
			      (EXCLUDED.country_id, EXCLUDED.organization_id, EXCLUDED.base_url, EXCLUDED.is_official,
			       EXCLUDED.verification_score, EXCLUDED.status, EXCLUDED.crawl_interval_sec)`,
			s.CountryID, s.OrganizationID, s.Domain, s.BaseURL, s.IsOfficial, s.VerificationScore,
			s.Status, s.CrawlIntervalSec)
		if err != nil {
			return fmt.Errorf("seed site %s: %w", s.Domain, err)
		}
	}
	return nil
}
