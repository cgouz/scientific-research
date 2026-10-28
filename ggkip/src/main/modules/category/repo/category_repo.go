package repo

import (
	"context"
	"database/sql"
	"errors"
	"fmt"

	"ggkip/src/main/infrastructure"
	"ggkip/src/main/modules/category/model"
)

type Repo struct {
	db infrastructure.DBTX
}

func New(db infrastructure.DBTX) *Repo {
	return &Repo{db: db}
}

const columns = `id, parent_id, code, name, description, is_active, created_at, updated_at`

func scan(row interface{ Scan(...any) error }) (model.Category, error) {
	var c model.Category
	err := row.Scan(&c.ID, &c.ParentID, &c.Code, &c.Name, &c.Description, &c.IsActive, &c.CreatedAt, &c.UpdatedAt)
	return c, err
}

// List returns all matching categories; the taxonomy is small and not paginated.
func (r *Repo) List(ctx context.Context, f model.ListFilter) ([]model.Category, error) {
	var w infrastructure.Where
	if f.ParentID != nil {
		w.Add("parent_id = ?", *f.ParentID)
	}
	if f.IsActive != nil {
		w.Add("is_active = ?", *f.IsActive)
	}
	if f.TopLevel {
		w.Raw("parent_id IS NULL")
	}
	rows, err := r.db.QueryContext(ctx, `SELECT `+columns+` FROM categories`+w.SQL()+` ORDER BY id`, w.Args...)
	if err != nil {
		return nil, fmt.Errorf("list categories: %w", err)
	}
	defer rows.Close()
	var out []model.Category
	for rows.Next() {
		c, err := scan(rows)
		if err != nil {
			return nil, err
		}
		out = append(out, c)
	}
	return out, rows.Err()
}

func (r *Repo) Get(ctx context.Context, id int64) (model.Category, error) {
	c, err := scan(r.db.QueryRowContext(ctx, `SELECT `+columns+` FROM categories WHERE id = $1`, id))
	if errors.Is(err, sql.ErrNoRows) {
		return c, fmt.Errorf("category %d: %w", id, infrastructure.ErrNotFound)
	}
	return c, err
}

// UpsertSeed expects parents to come before their children in items.
func (r *Repo) UpsertSeed(ctx context.Context, items []model.Category) error {
	for _, c := range items {
		_, err := r.db.ExecContext(ctx, `
			INSERT INTO categories (id, parent_id, code, name, description, is_active)
			OVERRIDING SYSTEM VALUE
			VALUES ($1, $2, $3, $4, $5, $6)
			ON CONFLICT (id) DO UPDATE SET
				parent_id = EXCLUDED.parent_id, code = EXCLUDED.code, name = EXCLUDED.name,
				description = EXCLUDED.description, is_active = EXCLUDED.is_active, updated_at = NOW()
			WHERE (categories.parent_id, categories.code, categories.name, categories.description, categories.is_active)
			      IS DISTINCT FROM
			      (EXCLUDED.parent_id, EXCLUDED.code, EXCLUDED.name, EXCLUDED.description, EXCLUDED.is_active)`,
			c.ID, c.ParentID, c.Code, c.Name, c.Description, c.IsActive)
		if err != nil {
			return fmt.Errorf("seed category %d (%s): %w", c.ID, c.Code, err)
		}
	}
	return infrastructure.ResetIdentity(ctx, r.db, "categories")
}
