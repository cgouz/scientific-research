package app

import (
	"context"
	"database/sql"
	"errors"
	"fmt"
	"io/fs"
	"log/slog"
	"net/url"
	"strings"

	"ggkip/src/main/infrastructure"
	categorymodel "ggkip/src/main/modules/category/model"
	categoryrepo "ggkip/src/main/modules/category/repo"
	countrymodel "ggkip/src/main/modules/country/model"
	countryrepo "ggkip/src/main/modules/country/repo"
	organizationmodel "ggkip/src/main/modules/organization/model"
	organizationrepo "ggkip/src/main/modules/organization/repo"
	orgtypemodel "ggkip/src/main/modules/organization_type/model"
	orgtyperepo "ggkip/src/main/modules/organization_type/repo"
	sitemodel "ggkip/src/main/modules/site/model"
	siterepo "ggkip/src/main/modules/site/repo"
)

// SeedData is the parsed content of resources/seed/*.json.
type SeedData struct {
	Countries           []countrymodel.Country
	CountryTranslations []countrymodel.Translation
	OrganizationTypes   []orgtypemodel.OrganizationType
	Categories          []categorymodel.Category
	Organizations       []organizationmodel.Organization
	Sites               []sitemodel.Site
}

// LoadSeed reads and validates all seed files, including every foreign key
// between them, before anything touches the database.
func LoadSeed(resources fs.FS) (*SeedData, error) {
	var d SeedData
	files := []struct {
		name     string
		dst      any
		optional bool
	}{
		{"seed/countries.json", &d.Countries, false},
		{"seed/countries_translations.json", &d.CountryTranslations, true},
		{"seed/organization_types.json", &d.OrganizationTypes, false},
		{"seed/categories.json", &d.Categories, false},
		{"seed/organizations.json", &d.Organizations, false},
		{"seed/sites.json", &d.Sites, false},
	}
	for _, f := range files {
		if err := readJSON(resources, f.name, f.dst); err != nil {
			if f.optional && errors.Is(err, fs.ErrNotExist) {
				continue
			}
			return nil, err
		}
	}
	if err := d.validate(); err != nil {
		return nil, err
	}
	var err error
	if d.Categories, err = parentsFirst(d.Categories, func(c categorymodel.Category) (int64, *int64) { return c.ID, c.ParentID }); err != nil {
		return nil, fmt.Errorf("categories.json: %w", err)
	}
	if d.Organizations, err = parentsFirst(d.Organizations, func(o organizationmodel.Organization) (int64, *int64) { return o.ID, o.ParentID }); err != nil {
		return nil, fmt.Errorf("organizations.json: %w", err)
	}
	return &d, nil
}

func (d *SeedData) validate() error {
	var errs []error
	bad := func(file string, i int, format string, args ...any) {
		errs = append(errs, fmt.Errorf("%s[%d]: %s", file, i, fmt.Sprintf(format, args...)))
	}

	countries := map[int64]bool{}
	iso2 := map[string]bool{}
	for i, c := range d.Countries {
		switch {
		case c.ID <= 0 || countries[c.ID]:
			bad("countries.json", i, "id %d is missing or duplicated", c.ID)
		case len(c.ISO2) != 2 || len(c.ISO3) != 3:
			bad("countries.json", i, "iso2/iso3 must have 2/3 letters")
		case iso2[c.ISO2]:
			bad("countries.json", i, "iso2 %s is duplicated", c.ISO2)
		case strings.TrimSpace(c.Name) == "":
			bad("countries.json", i, "name is empty")
		}
		countries[c.ID], iso2[c.ISO2] = true, true
	}

	translationIDs := map[int64]bool{}
	for i, t := range d.CountryTranslations {
		switch {
		case t.ID <= 0 || translationIDs[t.ID]:
			bad("countries_translations.json", i, "id %d is missing or duplicated", t.ID)
		case !countries[t.CountryID]:
			bad("countries_translations.json", i, "country_id %d not found in countries.json", t.CountryID)
		case t.Locale == "" || t.Name == "":
			bad("countries_translations.json", i, "locale and name are required")
		}
		translationIDs[t.ID] = true
	}

	orgTypes := map[int64]bool{}
	for i, t := range d.OrganizationTypes {
		if t.ID <= 0 || orgTypes[t.ID] || t.Code == "" || t.Name == "" {
			bad("organization_types.json", i, "id must be unique and code/name are required")
		}
		orgTypes[t.ID] = true
	}

	cats := map[int64]categorymodel.Category{}
	codes := map[string]bool{}
	for i, c := range d.Categories {
		if c.ID <= 0 || cats[c.ID].ID != 0 || c.Code == "" || codes[c.Code] || c.Name == "" {
			bad("categories.json", i, "id and code must be unique and code/name are required")
		}
		cats[c.ID], codes[c.Code] = c, true
	}
	for i, c := range d.Categories {
		if c.ParentID == nil {
			continue
		}
		parent, ok := cats[*c.ParentID]
		switch {
		case !ok:
			bad("categories.json", i, "parent_id %d not found", *c.ParentID)
		case parent.ParentID != nil:
			bad("categories.json", i, "parent %s is itself a subcategory; only two levels are allowed", parent.Code)
		}
	}

	orgs := map[int64]organizationmodel.Organization{}
	names := map[string]bool{}
	for i, o := range d.Organizations {
		key := fmt.Sprintf("%d/%s", o.CountryID, o.Name)
		switch {
		case o.ID <= 0 || orgs[o.ID].ID != 0:
			bad("organizations.json", i, "id %d is missing or duplicated", o.ID)
		case !countries[o.CountryID]:
			bad("organizations.json", i, "country_id %d not found in countries.json", o.CountryID)
		case !orgTypes[o.OrganizationTypeID]:
			bad("organizations.json", i, "organization_type_id %d not found", o.OrganizationTypeID)
		case o.Name == "" || names[key]:
			bad("organizations.json", i, "name is empty or duplicated within its country")
		}
		if o.Status == "" {
			d.Organizations[i].Status = organizationmodel.StatusActive
		}
		orgs[o.ID], names[key] = o, true
	}
	for i, o := range d.Organizations {
		if o.ParentID == nil {
			continue
		}
		parent, ok := orgs[*o.ParentID]
		switch {
		case !ok:
			bad("organizations.json", i, "parent_id %d not found", *o.ParentID)
		case parent.CountryID != o.CountryID:
			bad("organizations.json", i, "parent %d belongs to another country", *o.ParentID)
		}
	}

	domains := map[string]bool{}
	for i, s := range d.Sites {
		org, orgOK := orgs[s.OrganizationID]
		u, err := url.Parse(s.BaseURL)
		host := ""
		if err == nil {
			host = strings.TrimPrefix(strings.ToLower(u.Hostname()), "www.")
		}
		switch {
		case s.Domain == "" || s.Domain != strings.ToLower(s.Domain) || strings.Contains(s.Domain, "/"):
			bad("sites.json", i, "domain %q must be a lowercase host name", s.Domain)
		case domains[s.Domain]:
			bad("sites.json", i, "domain %s is duplicated", s.Domain)
		case err != nil || (u.Scheme != "http" && u.Scheme != "https") || host != strings.TrimPrefix(s.Domain, "www."):
			bad("sites.json", i, "base_url %q must be http(s) on %s or www.%s", s.BaseURL, s.Domain, s.Domain)
		case !countries[s.CountryID]:
			bad("sites.json", i, "country_id %d not found in countries.json", s.CountryID)
		case !orgOK:
			bad("sites.json", i, "organization_id %d not found in organizations.json", s.OrganizationID)
		case org.CountryID != s.CountryID:
			bad("sites.json", i, "organization %d belongs to country %d, not %d", s.OrganizationID, org.CountryID, s.CountryID)
		case s.CrawlIntervalSec <= 0:
			bad("sites.json", i, "crawl_interval_sec must be > 0")
		case s.VerificationScore < 0 || s.VerificationScore > 1:
			bad("sites.json", i, "verification_score must be between 0 and 1")
		}
		if s.Status == "" {
			d.Sites[i].Status = sitemodel.StatusActive
		}
		domains[s.Domain] = true
	}
	return errors.Join(errs...)
}

// parentsFirst orders rows so every parent is inserted before its children.
func parentsFirst[T any](items []T, key func(T) (int64, *int64)) ([]T, error) {
	placed := map[int64]bool{}
	out := make([]T, 0, len(items))
	for len(out) < len(items) {
		progress := false
		for _, it := range items {
			id, parent := key(it)
			if placed[id] || (parent != nil && !placed[*parent]) {
				continue
			}
			placed[id] = true
			out = append(out, it)
			progress = true
		}
		if !progress {
			return nil, errors.New("parent_id references form a cycle")
		}
	}
	return out, nil
}

// ApplySeed upserts all seed data in one transaction. It is idempotent:
// unchanged rows are not rewritten, and crawl state of sites is never touched.
func ApplySeed(ctx context.Context, db *sql.DB, d *SeedData, log *slog.Logger) error {
	err := infrastructure.InTx(ctx, db, func(tx *sql.Tx) error {
		if err := countryrepo.New(tx).UpsertSeed(ctx, d.Countries); err != nil {
			return err
		}
		if err := countryrepo.New(tx).UpsertTranslationsSeed(ctx, d.CountryTranslations); err != nil {
			return err
		}
		if err := orgtyperepo.New(tx).UpsertSeed(ctx, d.OrganizationTypes); err != nil {
			return err
		}
		if err := categoryrepo.New(tx).UpsertSeed(ctx, d.Categories); err != nil {
			return err
		}
		if err := organizationrepo.New(tx).UpsertSeed(ctx, d.Organizations); err != nil {
			return err
		}
		return siterepo.New(tx).UpsertSeed(ctx, d.Sites)
	})
	if err != nil {
		return fmt.Errorf("seed: %w", err)
	}
	log.Info("seed applied", "countries", len(d.Countries), "country_translations", len(d.CountryTranslations),
		"organization_types", len(d.OrganizationTypes), "categories", len(d.Categories),
		"organizations", len(d.Organizations), "sites", len(d.Sites))
	return nil
}
