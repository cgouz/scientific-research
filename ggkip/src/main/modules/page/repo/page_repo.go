package repo

import (
	"context"
	"fmt"

	"ggkip/src/main/infrastructure"
	"ggkip/src/main/modules/page/model"
)

type Repo struct {
	db infrastructure.DBTX
}

func New(db infrastructure.DBTX) *Repo {
	return &Repo{db: db}
}

// Observe inserts the page if its (site_id, url_hash) is new, otherwise
// updates last_seen_at/last_crawled_at. It returns the stored page and
// whether it was inserted. Status and content_hash of an existing page are
// not changed here.
func (r *Repo) Observe(ctx context.Context, o model.Observation) (model.Page, bool, error) {
	p := model.Page{SiteID: o.SiteID, URL: o.URL, URLHash: o.URLHash, HTTPStatus: o.HTTPStatus}
	var inserted bool
	err := r.db.QueryRowContext(ctx, `
		INSERT INTO pages (site_id, url, url_hash, status, http_status, language_code,
		                   first_seen_at, last_seen_at, last_crawled_at)
		VALUES ($1, $2, $3, 'new', $4, $5, $6, $6, $6)
		ON CONFLICT (site_id, url_hash) DO UPDATE SET
			last_seen_at = EXCLUDED.last_seen_at,
			last_crawled_at = EXCLUDED.last_crawled_at,
			http_status = EXCLUDED.http_status,
			language_code = COALESCE(EXCLUDED.language_code, pages.language_code),
			updated_at = NOW()
		RETURNING id, content_hash, status, language_code, first_seen_at, last_seen_at, (xmax = 0)`,
		o.SiteID, o.URL, o.URLHash, o.HTTPStatus, o.LanguageCode, o.At,
	).Scan(&p.ID, &p.ContentHash, &p.Status, &p.LanguageCode, &p.FirstSeenAt, &p.LastSeenAt, &inserted)
	if err != nil {
		return p, false, fmt.Errorf("observe page %s: %w", o.URL, err)
	}
	return p, inserted, nil
}

func (r *Repo) SetStatus(ctx context.Context, id int64, status string) error {
	_, err := r.db.ExecContext(ctx, `UPDATE pages SET status = $2, updated_at = NOW() WHERE id = $1`, id, status)
	if err != nil {
		return fmt.Errorf("set page %d status: %w", id, err)
	}
	return nil
}

// MarkProcessed stores the content hash of text that now has a record.
func (r *Repo) MarkProcessed(ctx context.Context, id int64, contentHash []byte, status string) error {
	_, err := r.db.ExecContext(ctx,
		`UPDATE pages SET content_hash = $2, status = $3, updated_at = NOW() WHERE id = $1`, id, contentHash, status)
	if err != nil {
		return fmt.Errorf("mark page %d processed: %w", id, err)
	}
	return nil
}

// List returns pages newest first (cursor = continue below this id).
func (r *Repo) List(ctx context.Context, f model.ListFilter) ([]model.Page, error) {
	var w infrastructure.Where
	if f.Cursor > 0 {
		w.Add("id < ?", f.Cursor)
	}
	if f.SiteID != nil {
		w.Add("site_id = ?", *f.SiteID)
	}
	if f.Status != nil {
		w.Add("status = ?", *f.Status)
	}
	rows, err := r.db.QueryContext(ctx, `
		SELECT id, site_id, url, url_hash, content_hash, status, http_status, language_code,
		       first_seen_at, last_seen_at, last_crawled_at, created_at, updated_at
		FROM pages`+w.SQL()+` ORDER BY id DESC LIMIT `+w.Arg(f.Limit+1), w.Args...)
	if err != nil {
		return nil, fmt.Errorf("list pages: %w", err)
	}
	defer rows.Close()
	var out []model.Page
	for rows.Next() {
		var p model.Page
		if err := rows.Scan(&p.ID, &p.SiteID, &p.URL, &p.URLHash, &p.ContentHash, &p.Status, &p.HTTPStatus,
			&p.LanguageCode, &p.FirstSeenAt, &p.LastSeenAt, &p.LastCrawledAt, &p.CreatedAt, &p.UpdatedAt); err != nil {
			return nil, err
		}
		out = append(out, p)
	}
	return out, rows.Err()
}
