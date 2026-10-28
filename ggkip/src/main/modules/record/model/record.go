package model

import (
	"encoding/json"
	"time"
)

// Record is one normalized piece of government content. Country,
// organization, site, page, category and subcategory are real foreign keys,
// so a record always stays tied to its source no matter what the LLM said.
type Record struct {
	ID             int64
	CountryID      int64
	OrganizationID int64
	SiteID         int64
	PageID         int64
	CategoryID     int64
	SubcategoryID  *int64
	RecordType     string
	Title          *string
	LanguageCode   *string
	PublishedAt    *time.Time
	Content        string
	Summary        *string
	Keywords       []string
	Metadata       json.RawMessage
	Confidence     *float64
	ContentHash    []byte
	LLMRuntime     *string
	LLMModel       *string
	CreatedAt      time.Time
	UpdatedAt      time.Time

	// SourceURL is read from pages.url; it is not a records column.
	SourceURL string
}

type ListFilter struct {
	CountryID      *int64
	OrganizationID *int64
	SiteID         *int64
	CategoryID     *int64
	SubcategoryID  *int64
	RecordType     *string
	Language       *string
	PublishedFrom  *time.Time
	PublishedTo    *time.Time
	Cursor         int64
	Limit          int
}
