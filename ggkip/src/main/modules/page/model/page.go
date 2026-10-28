package model

import "time"

// Page answers two questions: have we seen this URL, and has its text changed?
// ContentHash is only set once a record for that text has been stored, so a
// page whose LLM step failed is retried on the next crawl.
type Page struct {
	ID            int64
	SiteID        int64
	URL           string
	URLHash       []byte
	ContentHash   []byte
	Status        string
	HTTPStatus    *int
	LanguageCode  *string
	FirstSeenAt   time.Time
	LastSeenAt    time.Time
	LastCrawledAt *time.Time
	CreatedAt     time.Time
	UpdatedAt     time.Time
}

// Page statuses: the outcome of the last crawl of the page.
const (
	StatusNew       = "new"
	StatusChanged   = "changed"
	StatusUnchanged = "unchanged"
	StatusSkipped   = "skipped"
	StatusFailed    = "failed"
)

// Observation is what a crawl learned about a URL before any content comparison.
type Observation struct {
	SiteID       int64
	URL          string
	URLHash      []byte
	HTTPStatus   *int
	LanguageCode *string
	At           time.Time
}

type ListFilter struct {
	SiteID *int64
	Status *string
	Cursor int64
	Limit  int
}
