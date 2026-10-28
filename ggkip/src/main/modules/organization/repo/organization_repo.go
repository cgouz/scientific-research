package repo

import (
	"context"
	"database/sql"
	"errors"
	"fmt"

	"ggkip/src/main/infrastructure"
	"ggkip/src/main/modules/organization/model"
)

type Repo struct {
	db infrastructure.DBTX
}

func New(db infrastructure.DBTX) *Repo {
	return &Repo{db: db}
}

const columns = `id, country_id, organization_type_id, parent_id, name, official_name, description, status, created_at, updated_at`

func scan(row interface{ Scan(...any) error }) (model.Organization, error) {
	var o model.Organization
	err := row.Scan(&o.ID, &o.CountryID, &o.OrganizationTypeID, &o.ParentID, &o.Name, &o.OfficialName,
		&o.Description, &o.Status, &o.CreatedAt, &o.UpdatedAt)
	return o, err
}

func (r *Repo) List(ctx context.Context, f model.ListFilter) ([]model.Organization, error) {
	var w infrastructure.Where
	w.Add("id > ?", f.Cursor)
	if f.CountryID != nil {
		w.Add("country_id = ?", *f.CountryID)
	}
	if f.OrganizationTypeID != nil {
		w.Add("organization_type_id = ?", *f.OrganizationTypeID)
	}
	if f.ParentID != nil {
		w.Add("parent_id = ?", *f.ParentID)
	}
	if f.Status != nil {
		w.Add("status = ?", *f.Status)
	}
	rows, err := r.db.QueryContext(ctx,
		`SELECT `+columns+` FROM organizations`+w.SQL()+` ORDER BY id LIMIT `+w.Arg(f.Limit+1), w.Args...)
	if err != nil {
		return nil, fmt.Errorf("list organizations: %w", err)
	}
	defer rows.Close()
	var out []model.Organization
	for rows.Next() {
		o, err := scan(rows)
		if err != nil {
			return nil, err
		}
		out = append(out, o)
	}
	return out, rows.Err()
}

func (r *Repo) Get(ctx context.Context, id int64) (model.Organization, error) {
	o, err := scan(r.db.QueryRowContext(ctx, `SELECT `+columns+` FROM organizations WHERE id = $1`, id))
	if errors.Is(err, sql.ErrNoRows) {
		return o, fmt.Errorf("organization %d: %w", id, infrastructure.ErrNotFound)
	}
	return o, err
}

// UpsertSeed expects parents to come before their children in items.
func (r *Repo) UpsertSeed(ctx context.Context, items []model.Organization) error {
	for _, o := range items {
		_, err := r.db.ExecContext(ctx, `
			INSERT INTO organizations (id, country_id, organization_type_id, parent_id, name, official_name, description, status)
			OVERRIDING SYSTEM VALUE
			VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
			ON CONFLICT (id) DO UPDATE SET
				country_id = EXCLUDED.country_id, organization_type_id = EXCLUDED.organization_type_id,
				parent_id = EXCLUDED.parent_id, name = EXCLUDED.name, official_name = EXCLUDED.official_name,
				description = EXCLUDED.description, status = EXCLUDED.status, updated_at = NOW()
			WHERE (organizations.country_id, organizations.organization_type_id, organizations.parent_id,
			       organizations.name, organizations.official_name, organizations.description, organizations.status)
			      IS DISTINCT FROM
			      (EXCLUDED.country_id, EXCLUDED.organization_type_id, EXCLUDED.parent_id,
			       EXCLUDED.name, EXCLUDED.official_name, EXCLUDED.description, EXCLUDED.status)`,
			o.ID, o.CountryID, o.OrganizationTypeID, o.ParentID, o.Name, o.OfficialName, o.Description, o.Status)
		if err != nil {
			return fmt.Errorf("seed organization %d (%s): %w", o.ID, o.Name, err)
		}
	}
	return infrastructure.ResetIdentity(ctx, r.db, "organizations")
}
