package model

import (
	"errors"
	"fmt"
	"strings"
	"time"

	"ggkip/src/main/modules/llm/dto"
)

// TaxonomySettings come from resources/settings/taxonomy.json.
type TaxonomySettings struct {
	UnknownCategoryCode string   `json:"unknown_category_code"`
	RecordTypes         []string `json:"record_types"`
}

// Entry is one category row as loaded from the database.
type Entry struct {
	ID          int64
	ParentID    *int64
	Code        string
	Name        string
	Description string
}

// Taxonomy is the fixed set of codes the LLM may choose from.
type Taxonomy struct {
	UnknownCode string
	RecordTypes []string
	Categories  []dto.TaxonomyCategory // prompt form, top-level with nested subcategories

	byCode      map[string]Entry
	recordTypes map[string]bool
}

// NewTaxonomy builds the taxonomy from category rows (two levels) and checks
// that the "unknown" category exists, because records.category_id is required.
func NewTaxonomy(entries []Entry, s TaxonomySettings) (*Taxonomy, error) {
	t := &Taxonomy{
		UnknownCode: s.UnknownCategoryCode,
		RecordTypes: s.RecordTypes,
		byCode:      map[string]Entry{},
		recordTypes: map[string]bool{},
	}
	if len(s.RecordTypes) == 0 {
		return nil, errors.New("taxonomy: record_types is empty")
	}
	for _, rt := range s.RecordTypes {
		t.recordTypes[rt] = true
	}
	byID := map[int64]Entry{}
	for _, e := range entries {
		t.byCode[e.Code] = e
		byID[e.ID] = e
	}
	index := map[int64]int{}
	for _, e := range entries {
		if e.ParentID == nil {
			index[e.ID] = len(t.Categories)
			t.Categories = append(t.Categories, dto.TaxonomyCategory{Code: e.Code, Name: e.Name, Description: e.Description})
		}
	}
	for _, e := range entries {
		if e.ParentID == nil {
			continue
		}
		i, ok := index[*e.ParentID]
		if !ok {
			return nil, fmt.Errorf("taxonomy: %s has parent %d which is not an active top-level category", e.Code, *e.ParentID)
		}
		t.Categories[i].Subcategories = append(t.Categories[i].Subcategories,
			dto.TaxonomyCategory{Code: e.Code, Name: e.Name, Description: e.Description})
	}
	unknown, ok := t.byCode[s.UnknownCategoryCode]
	if !ok || unknown.ParentID != nil {
		return nil, fmt.Errorf("taxonomy: top-level category %q is required", s.UnknownCategoryCode)
	}
	return t, nil
}

func (t *Taxonomy) Lookup(code string) (Entry, bool) {
	e, ok := t.byCode[code]
	return e, ok
}

func (t *Taxonomy) HasRecordType(rt string) bool {
	return t.recordTypes[rt]
}

// CategoryCodes and SubcategoryCodes are used for the JSON schema enums.
func (t *Taxonomy) CategoryCodes() []string {
	out := make([]string, 0, len(t.Categories))
	for _, c := range t.Categories {
		out = append(out, c.Code)
	}
	return out
}

func (t *Taxonomy) SubcategoryCodes() []string {
	var out []string
	for _, c := range t.Categories {
		for _, s := range c.Subcategories {
			out = append(out, s.Code)
		}
	}
	return out
}

// Result is a validated LLM output, with codes resolved to category ids.
type Result struct {
	RecordType    string
	CategoryID    int64
	SubcategoryID *int64
	Title         string
	Language      string
	PublishedAt   *time.Time
	Content       string
	Summary       string
	Keywords      []string
	Confidence    float64
}

// ValidationError marks LLM output that does not satisfy the taxonomy rules.
type ValidationError struct {
	Problems []string
}

func (e *ValidationError) Error() string {
	return "invalid llm output: " + strings.Join(e.Problems, "; ")
}
