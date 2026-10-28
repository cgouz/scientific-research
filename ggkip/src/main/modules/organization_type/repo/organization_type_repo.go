package repo

import (
	"context"
	"fmt"

	"ggkip/src/main/infrastructure"
	"ggkip/src/main/modules/organization_type/model"
)

type Repo struct {
	db infrastructure.DBTX
}

func New(db infrastructure.DBTX) *Repo {
	return &Repo{db: db}
}

// List returns all organization types; the table is small and not paginated.
func (r *Repo) List(ctx context.Context) ([]model.OrganizationType, error) {
	rows, err := r.db.QueryContext(ctx, `SELECT id, code, name, created_at FROM organization_types ORDER BY id`)
	if err != nil {
		return nil, fmt.Errorf("list organization types: %w", err)
	}
	defer rows.Close()
	var out []model.OrganizationType
	for rows.Next() {
		var t model.OrganizationType
		if err := rows.Scan(&t.ID, &t.Code, &t.Name, &t.CreatedAt); err != nil {
			return nil, err
		}
		out = append(out, t)
	}
	return out, rows.Err()
}

func (r *Repo) UpsertSeed(ctx context.Context, items []model.OrganizationType) error {
	for _, t := range items {
		_, err := r.db.ExecContext(ctx, `
			INSERT INTO organization_types (id, code, name)
			OVERRIDING SYSTEM VALUE
			VALUES ($1, $2, $3)
			ON CONFLICT (id) DO UPDATE SET code = EXCLUDED.code, name = EXCLUDED.name
			WHERE (organization_types.code, organization_types.name) IS DISTINCT FROM (EXCLUDED.code, EXCLUDED.name)`,
			t.ID, t.Code, t.Name)
		if err != nil {
			return fmt.Errorf("seed organization type %d (%s): %w", t.ID, t.Code, err)
		}
	}
	return infrastructure.ResetIdentity(ctx, r.db, "organization_types")
}
