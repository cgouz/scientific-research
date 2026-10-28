package dto

import (
	"encoding/hex"
	"time"

	"ggkip/src/main/modules/page/model"
)

type PageResponse struct {
	ID            int64      `json:"id"`
	SiteID        int64      `json:"site_id"`
	URL           string     `json:"url"`
	ContentHash   *string    `json:"content_hash"`
	Status        string     `json:"status"`
	HTTPStatus    *int       `json:"http_status"`
	LanguageCode  *string    `json:"language_code"`
	FirstSeenAt   time.Time  `json:"first_seen_at"`
	LastSeenAt    time.Time  `json:"last_seen_at"`
	LastCrawledAt *time.Time `json:"last_crawled_at"`
}

func FromModel(p model.Page) PageResponse {
	resp := PageResponse{
		ID: p.ID, SiteID: p.SiteID, URL: p.URL, Status: p.Status, HTTPStatus: p.HTTPStatus,
		LanguageCode: p.LanguageCode, FirstSeenAt: p.FirstSeenAt, LastSeenAt: p.LastSeenAt,
		LastCrawledAt: p.LastCrawledAt,
	}
	if p.ContentHash != nil {
		h := hex.EncodeToString(p.ContentHash)
		resp.ContentHash = &h
	}
	return resp
}
