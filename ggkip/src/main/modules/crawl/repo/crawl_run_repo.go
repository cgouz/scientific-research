package repo

import (
	"context"
	"database/sql"
	"errors"
	"fmt"
	"time"

	"ggkip/src/main/infrastructure"
	"ggkip/src/main/modules/crawl/model"
)

type Repo struct {
	db infrastructure.DBTX
}

func New(db infrastructure.DBTX) *Repo {
	return &Repo{db: db}
}

const columns = `id, site_id, batch_id, status, started_at, finished_at, pages_found, pages_new, pages_changed,
	pages_unchanged, pages_skipped, pages_failed, records_created, records_updated, error_message, created_at`

func scan(row interface{ Scan(...any) error }) (model.CrawlRun, error) {
	var r model.CrawlRun
	c := &r.Counters
	err := row.Scan(&r.ID, &r.SiteID, &r.BatchID, &r.Status, &r.StartedAt, &r.FinishedAt, &c.PagesFound, &c.PagesNew,
		&c.PagesChanged, &c.PagesUnchanged, &c.PagesSkipped, &c.PagesFailed, &c.RecordsCreated,
		&c.RecordsUpdated, &r.ErrorMessage, &r.CreatedAt)
	return r, err
}

// Create inserts a 'running' row; batchID is nil for runs outside a batch.
func (r *Repo) Create(ctx context.Context, siteID int64, batchID *int64, startedAt time.Time) (model.CrawlRun, error) {
	run, err := scan(r.db.QueryRowContext(ctx, `
		INSERT INTO crawl_runs (site_id, batch_id, status, started_at) VALUES ($1, $2, 'running', $3)
		RETURNING `+columns, siteID, batchID, startedAt))
	if err != nil {
		return run, fmt.Errorf("create crawl run for site %d: %w", siteID, err)
	}
	return run, nil
}

func (r *Repo) Finish(ctx context.Context, id int64, status string, finishedAt time.Time, c model.Counters, errMsg *string) error {
	_, err := r.db.ExecContext(ctx, `
		UPDATE crawl_runs SET status = $2, finished_at = $3, pages_found = $4, pages_new = $5,
			pages_changed = $6, pages_unchanged = $7, pages_skipped = $8, pages_failed = $9,
			records_created = $10, records_updated = $11, error_message = $12
		WHERE id = $1`,
		id, status, finishedAt, c.PagesFound, c.PagesNew, c.PagesChanged, c.PagesUnchanged, c.PagesSkipped,
		c.PagesFailed, c.RecordsCreated, c.RecordsUpdated, errMsg)
	if err != nil {
		return fmt.Errorf("finish crawl run %d: %w", id, err)
	}
	return nil
}

// FailInterrupted closes runs left in 'running' by a previous process that
// stopped without finishing them. Call it once at startup.
func (r *Repo) FailInterrupted(ctx context.Context) (int64, error) {
	res, err := r.db.ExecContext(ctx, `
		UPDATE crawl_runs SET status = 'failed', finished_at = NOW(), error_message = 'interrupted: application stopped'
		WHERE status = 'running'`)
	if err != nil {
		return 0, fmt.Errorf("close interrupted crawl runs: %w", err)
	}
	return res.RowsAffected()
}

// List returns runs newest first (cursor = continue below this id).
func (r *Repo) List(ctx context.Context, f model.ListFilter) ([]model.CrawlRun, error) {
	var w infrastructure.Where
	if f.Cursor > 0 {
		w.Add("id < ?", f.Cursor)
	}
	if f.SiteID != nil {
		w.Add("site_id = ?", *f.SiteID)
	}
	if f.BatchID != nil {
		w.Add("batch_id = ?", *f.BatchID)
	}
	if f.Status != nil {
		w.Add("status = ?", *f.Status)
	}
	rows, err := r.db.QueryContext(ctx,
		`SELECT `+columns+` FROM crawl_runs`+w.SQL()+` ORDER BY id DESC LIMIT `+w.Arg(f.Limit+1), w.Args...)
	if err != nil {
		return nil, fmt.Errorf("list crawl runs: %w", err)
	}
	defer rows.Close()
	var out []model.CrawlRun
	for rows.Next() {
		run, err := scan(rows)
		if err != nil {
			return nil, err
		}
		out = append(out, run)
	}
	return out, rows.Err()
}

func (r *Repo) Get(ctx context.Context, id int64) (model.CrawlRun, error) {
	run, err := scan(r.db.QueryRowContext(ctx, `SELECT `+columns+` FROM crawl_runs WHERE id = $1`, id))
	if errors.Is(err, sql.ErrNoRows) {
		return run, fmt.Errorf("crawl run %d: %w", id, infrastructure.ErrNotFound)
	}
	return run, err
}
