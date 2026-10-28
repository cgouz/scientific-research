package repo

import (
	"context"
	"database/sql"
	"encoding/json"
	"errors"
	"fmt"

	"ggkip/src/main/infrastructure"
	"ggkip/src/main/modules/record/model"
)

type Repo struct {
	db infrastructure.DBTX
}

func New(db infrastructure.DBTX) *Repo {
	return &Repo{db: db}
}

// Upsert stores the record for a page (one record per page). It returns the
// record id and whether a new row was created rather than an existing one updated.
func (r *Repo) Upsert(ctx context.Context, rec model.Record) (int64, bool, error) {
	keywords, err := json.Marshal(rec.Keywords)
	if err != nil {
		return 0, false, err
	}
	if rec.Keywords == nil {
		keywords = []byte("[]")
	}
	metadata := []byte(rec.Metadata)
	if len(metadata) == 0 {
		metadata = []byte("{}")
	}

	var id int64
	var created bool
	err = r.db.QueryRowContext(ctx, `
		INSERT INTO records (country_id, organization_id, site_id, page_id, category_id, subcategory_id,
		                     record_type, title, language_code, published_at, content, summary,
		                     keywords, metadata, confidence, content_hash, llm_runtime, llm_model)
		VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18)
		ON CONFLICT (page_id) DO UPDATE SET
			country_id = EXCLUDED.country_id, organization_id = EXCLUDED.organization_id,
			site_id = EXCLUDED.site_id, category_id = EXCLUDED.category_id,
			subcategory_id = EXCLUDED.subcategory_id, record_type = EXCLUDED.record_type,
			title = EXCLUDED.title, language_code = EXCLUDED.language_code,
			published_at = EXCLUDED.published_at, content = EXCLUDED.content, summary = EXCLUDED.summary,
			keywords = EXCLUDED.keywords, metadata = EXCLUDED.metadata, confidence = EXCLUDED.confidence,
			content_hash = EXCLUDED.content_hash, llm_runtime = EXCLUDED.llm_runtime,
			llm_model = EXCLUDED.llm_model, updated_at = NOW()
		RETURNING id, (xmax = 0)`,
		rec.CountryID, rec.OrganizationID, rec.SiteID, rec.PageID, rec.CategoryID, rec.SubcategoryID,
		rec.RecordType, rec.Title, rec.LanguageCode, rec.PublishedAt, rec.Content, rec.Summary,
		keywords, metadata, rec.Confidence, rec.ContentHash, rec.LLMRuntime, rec.LLMModel,
	).Scan(&id, &created)
	if err != nil {
		return 0, false, fmt.Errorf("upsert record for page %d: %w", rec.PageID, err)
	}
	return id, created, nil
}

const selectRecord = `
	SELECT r.id, r.country_id, r.organization_id, r.site_id, r.page_id, r.category_id, r.subcategory_id,
	       r.record_type, r.title, r.language_code, r.published_at, r.content, r.summary, r.keywords,
	       r.metadata, r.confidence, r.content_hash, r.llm_runtime, r.llm_model, r.created_at, r.updated_at,
	       p.url
	FROM records r JOIN pages p ON p.id = r.page_id`

func scan(row interface{ Scan(...any) error }) (model.Record, error) {
	var rec model.Record
	var keywords, metadata []byte
	err := row.Scan(&rec.ID, &rec.CountryID, &rec.OrganizationID, &rec.SiteID, &rec.PageID, &rec.CategoryID,
		&rec.SubcategoryID, &rec.RecordType, &rec.Title, &rec.LanguageCode, &rec.PublishedAt, &rec.Content,
		&rec.Summary, &keywords, &metadata, &rec.Confidence, &rec.ContentHash, &rec.LLMRuntime, &rec.LLMModel,
		&rec.CreatedAt, &rec.UpdatedAt, &rec.SourceURL)
	if err != nil {
		return rec, err
	}
	rec.Metadata = json.RawMessage(metadata)
	if err := json.Unmarshal(keywords, &rec.Keywords); err != nil {
		return rec, fmt.Errorf("record %d keywords: %w", rec.ID, err)
	}
	return rec, nil
}

// List returns records newest first (cursor = continue below this id).
func (r *Repo) List(ctx context.Context, f model.ListFilter) ([]model.Record, error) {
	var w infrastructure.Where
	if f.Cursor > 0 {
		w.Add("r.id < ?", f.Cursor)
	}
	if f.CountryID != nil {
		w.Add("r.country_id = ?", *f.CountryID)
	}
	if f.OrganizationID != nil {
		w.Add("r.organization_id = ?", *f.OrganizationID)
	}
	if f.SiteID != nil {
		w.Add("r.site_id = ?", *f.SiteID)
	}
	if f.CategoryID != nil {
		w.Add("r.category_id = ?", *f.CategoryID)
	}
	if f.SubcategoryID != nil {
		w.Add("r.subcategory_id = ?", *f.SubcategoryID)
	}
	if f.RecordType != nil {
		w.Add("r.record_type = ?", *f.RecordType)
	}
	if f.Language != nil {
		w.Add("r.language_code = ?", *f.Language)
	}
	if f.PublishedFrom != nil {
		w.Add("r.published_at >= ?", *f.PublishedFrom)
	}
	if f.PublishedTo != nil {
		w.Add("r.published_at < ?", *f.PublishedTo)
	}
	rows, err := r.db.QueryContext(ctx, selectRecord+w.SQL()+` ORDER BY r.id DESC LIMIT `+w.Arg(f.Limit+1), w.Args...)
	if err != nil {
		return nil, fmt.Errorf("list records: %w", err)
	}
	defer rows.Close()
	var out []model.Record
	for rows.Next() {
		rec, err := scan(rows)
		if err != nil {
			return nil, err
		}
		out = append(out, rec)
	}
	return out, rows.Err()
}

func (r *Repo) Get(ctx context.Context, id int64) (model.Record, error) {
	rec, err := scan(r.db.QueryRowContext(ctx, selectRecord+` WHERE r.id = $1`, id))
	if errors.Is(err, sql.ErrNoRows) {
		return rec, fmt.Errorf("record %d: %w", id, infrastructure.ErrNotFound)
	}
	return rec, err
}
