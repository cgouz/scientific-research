package repo

import (
	"context"
	"database/sql"
	"errors"
	"fmt"
	"time"

	"github.com/lib/pq"

	"ggkip/src/main/infrastructure"
	"ggkip/src/main/modules/crawl/model"
)

type BatchRepo struct {
	db infrastructure.DBTX
}

func NewBatchRepo(db infrastructure.DBTX) *BatchRepo {
	return &BatchRepo{db: db}
}

const batchColumns = `id, status, country_ids, countries_total, sites_total, sites_skipped,
	started_at, finished_at, error_message, created_at`

func scanBatch(row interface{ Scan(...any) error }) (model.Batch, error) {
	var b model.Batch
	err := row.Scan(&b.ID, &b.Status, (*pq.Int64Array)(&b.CountryIDs), &b.CountriesTotal, &b.SitesTotal,
		&b.SitesSkipped, &b.StartedAt, &b.FinishedAt, &b.ErrorMessage, &b.CreatedAt)
	return b, err
}

// Create inserts a 'running' batch.
func (r *BatchRepo) Create(ctx context.Context, b model.Batch) (model.Batch, error) {
	ids := b.CountryIDs
	if ids == nil {
		ids = []int64{}
	}
	out, err := scanBatch(r.db.QueryRowContext(ctx, `
		INSERT INTO crawl_batches (status, country_ids, countries_total, sites_total, started_at)
		VALUES ('running', $1, $2, $3, $4)
		RETURNING `+batchColumns, pq.Array(ids), b.CountriesTotal, b.SitesTotal, b.StartedAt))
	if err != nil {
		return out, fmt.Errorf("create crawl batch: %w", err)
	}
	return out, nil
}

func (r *BatchRepo) Finish(ctx context.Context, id int64, status string, finishedAt time.Time, sitesSkipped int, errMsg *string) error {
	_, err := r.db.ExecContext(ctx, `
		UPDATE crawl_batches SET status = $2, finished_at = $3, sites_skipped = $4, error_message = $5
		WHERE id = $1`, id, status, finishedAt, sitesSkipped, errMsg)
	if err != nil {
		return fmt.Errorf("finish crawl batch %d: %w", id, err)
	}
	return nil
}

// FailInterrupted closes batches left 'running' by a previous process.
// Call it once at startup, like Repo.FailInterrupted for runs.
func (r *BatchRepo) FailInterrupted(ctx context.Context) (int64, error) {
	res, err := r.db.ExecContext(ctx, `
		UPDATE crawl_batches SET status = 'failed', finished_at = NOW(), error_message = 'interrupted: application stopped'
		WHERE status = 'running'`)
	if err != nil {
		return 0, fmt.Errorf("close interrupted crawl batches: %w", err)
	}
	return res.RowsAffected()
}

func (r *BatchRepo) Get(ctx context.Context, id int64) (model.Batch, error) {
	b, err := scanBatch(r.db.QueryRowContext(ctx, `SELECT `+batchColumns+` FROM crawl_batches WHERE id = $1`, id))
	if errors.Is(err, sql.ErrNoRows) {
		return b, fmt.Errorf("crawl batch %d: %w", id, infrastructure.ErrNotFound)
	}
	return b, err
}

// List returns batches newest first (cursor = continue below this id).
func (r *BatchRepo) List(ctx context.Context, f model.BatchListFilter) ([]model.Batch, error) {
	var w infrastructure.Where
	if f.Cursor > 0 {
		w.Add("id < ?", f.Cursor)
	}
	if f.Status != nil {
		w.Add("status = ?", *f.Status)
	}
	rows, err := r.db.QueryContext(ctx,
		`SELECT `+batchColumns+` FROM crawl_batches`+w.SQL()+` ORDER BY id DESC LIMIT `+w.Arg(f.Limit+1), w.Args...)
	if err != nil {
		return nil, fmt.Errorf("list crawl batches: %w", err)
	}
	defer rows.Close()
	var out []model.Batch
	for rows.Next() {
		b, err := scanBatch(rows)
		if err != nil {
			return nil, err
		}
		out = append(out, b)
	}
	return out, rows.Err()
}

// RunCounts returns how many of the batch's crawl runs are in each status.
func (r *BatchRepo) RunCounts(ctx context.Context, batchID int64) (model.RunCounts, error) {
	rows, err := r.db.QueryContext(ctx,
		`SELECT status, COUNT(*) FROM crawl_runs WHERE batch_id = $1 GROUP BY status`, batchID)
	if err != nil {
		return nil, fmt.Errorf("count runs of crawl batch %d: %w", batchID, err)
	}
	defer rows.Close()
	counts := model.RunCounts{}
	for rows.Next() {
		var status string
		var n int64
		if err := rows.Scan(&status, &n); err != nil {
			return nil, err
		}
		counts[status] = n
	}
	return counts, rows.Err()
}
