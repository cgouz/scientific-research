package dto

import (
	"time"

	"ggkip/src/main/modules/site/model"
)

type SiteResponse struct {
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
}

func FromModel(s model.Site) SiteResponse {
	return SiteResponse{
		ID: s.ID, CountryID: s.CountryID, OrganizationID: s.OrganizationID, Domain: s.Domain, BaseURL: s.BaseURL,
		IsOfficial: s.IsOfficial, VerificationScore: s.VerificationScore, Status: s.Status,
		CrawlIntervalSec: s.CrawlIntervalSec, LastCrawledAt: s.LastCrawledAt, NextCrawlAt: s.NextCrawlAt,
		LastSuccessAt: s.LastSuccessAt, LastErrorAt: s.LastErrorAt, LastError: s.LastError,
	}
}
