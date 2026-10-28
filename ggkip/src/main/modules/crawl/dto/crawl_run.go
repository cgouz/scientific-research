package dto

import (
	"time"

	"ggkip/src/main/modules/crawl/model"
)

type CrawlRunResponse struct {
	ID             int64      `json:"id"`
	SiteID         int64      `json:"site_id"`
	BatchID        *int64     `json:"batch_id"`
	Status         string     `json:"status"`
	StartedAt      *time.Time `json:"started_at"`
	FinishedAt     *time.Time `json:"finished_at"`
	PagesFound     int64      `json:"pages_found"`
	PagesNew       int64      `json:"pages_new"`
	PagesChanged   int64      `json:"pages_changed"`
	PagesUnchanged int64      `json:"pages_unchanged"`
	PagesSkipped   int64      `json:"pages_skipped"`
	PagesFailed    int64      `json:"pages_failed"`
	RecordsCreated int64      `json:"records_created"`
	RecordsUpdated int64      `json:"records_updated"`
	ErrorMessage   *string    `json:"error_message"`
	CreatedAt      time.Time  `json:"created_at"`
}

func FromModel(r model.CrawlRun) CrawlRunResponse {
	c := r.Counters
	return CrawlRunResponse{
		ID: r.ID, SiteID: r.SiteID, BatchID: r.BatchID, Status: r.Status, StartedAt: r.StartedAt, FinishedAt: r.FinishedAt,
		PagesFound: c.PagesFound, PagesNew: c.PagesNew, PagesChanged: c.PagesChanged,
		PagesUnchanged: c.PagesUnchanged, PagesSkipped: c.PagesSkipped, PagesFailed: c.PagesFailed,
		RecordsCreated: c.RecordsCreated, RecordsUpdated: c.RecordsUpdated, ErrorMessage: r.ErrorMessage,
		CreatedAt: r.CreatedAt,
	}
}
