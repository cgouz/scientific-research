package app

import (
	"encoding/json"
	"errors"
	"fmt"
	"io/fs"

	crawlermodel "ggkip/src/main/modules/crawler/model"
	llmmodel "ggkip/src/main/modules/llm/model"
)

type Settings struct {
	Crawler  crawlermodel.Settings
	Taxonomy llmmodel.TaxonomySettings
}

// LoadSettings reads settings/crawler.json and settings/taxonomy.json.
func LoadSettings(resources fs.FS) (Settings, error) {
	var s Settings
	if err := readJSON(resources, "settings/crawler.json", &s.Crawler); err != nil {
		return s, err
	}
	if err := readJSON(resources, "settings/taxonomy.json", &s.Taxonomy); err != nil {
		return s, err
	}
	return s, s.validate()
}

func (s Settings) validate() error {
	var errs []error
	check := func(ok bool, msg string) {
		if !ok {
			errs = append(errs, errors.New(msg))
		}
	}
	c := s.Crawler
	check(c.MaxPagesPerRun > 0, "crawler.json: max_pages_per_run must be > 0")
	check(c.MaxDepth >= 0, "crawler.json: max_depth must be >= 0")
	check(c.MaxSitemaps >= 0, "crawler.json: max_sitemaps must be >= 0")
	check(c.MaxRedirects >= 0, "crawler.json: max_redirects must be >= 0")
	check(c.MaxRetries >= 0, "crawler.json: max_retries must be >= 0")
	check(c.RequestsPerSecond > 0, "crawler.json: requests_per_second must be > 0")
	check(len(c.AllowedContentTypes) > 0, "crawler.json: allowed_content_types is empty")
	check(s.Taxonomy.UnknownCategoryCode != "", "taxonomy.json: unknown_category_code is required")
	check(len(s.Taxonomy.RecordTypes) > 0, "taxonomy.json: record_types is empty")
	return errors.Join(errs...)
}

func readJSON(fsys fs.FS, name string, v any) error {
	data, err := fs.ReadFile(fsys, name)
	if err != nil {
		return err
	}
	if err := json.Unmarshal(data, v); err != nil {
		return fmt.Errorf("%s: %w", name, err)
	}
	return nil
}
