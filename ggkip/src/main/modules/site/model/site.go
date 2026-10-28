package model

import "time"

// Site is a row of sites. JSON tags match seed/sites.json (which has no id;
// sites are matched by domain).
type Site struct {
	ID                int64      `json:"id"`
	CountryID         int64      `json:"country_id"`
	OrganizationID    int64      `json:"organization_id"`
	Domain            string     `json:"domain"`
	BaseURL           string     `json:"base_url"`
	IsOfficial        bool       `json:"is_official"`
	VerificationScore float64    `json:"verification_score"`
	Status            string     `json:"status"`
	CrawlIntervalSec  int64      `json:"crawl_interval_sec"`
	LastCrawledAt     *time.Time `json:"last_crawled_at"`
	NextCrawlAt       *time.Time `json:"next_crawl_at"`
	LastSuccessAt     *time.Time `json:"last_success_at"`
	LastErrorAt       *time.Time `json:"last_error_at"`
	LastError         *string    `json:"last_error"`
	CreatedAt         time.Time  `json:"created_at"`
	UpdatedAt         time.Time  `json:"updated_at"`
}

const StatusActive = "active"

func (s Site) CrawlInterval() time.Duration {
	return time.Duration(s.CrawlIntervalSec) * time.Second
}

type ListFilter struct {
	CountryID      *int64
	OrganizationID *int64
	Status         *string
	Cursor         int64
	Limit          int
}
