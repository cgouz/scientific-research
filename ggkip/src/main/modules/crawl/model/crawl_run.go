package model

import (
	"sync/atomic"
	"time"
)

// CrawlRun is the history row of one crawl of one site.
type CrawlRun struct {
	ID           int64
	SiteID       int64
	Status       string
	StartedAt    *time.Time
	FinishedAt   *time.Time
	Counters     Counters
	ErrorMessage *string
	CreatedAt    time.Time
}

// Crawl run statuses.
const (
	StatusRunning   = "running"
	StatusCompleted = "completed"
	StatusPartial   = "partial" // finished, but some pages failed
	StatusFailed    = "failed"
	StatusCancelled = "cancelled"
)

// Counters are the per-run page and record totals.
type Counters struct {
	PagesFound     int64
	PagesNew       int64
	PagesChanged   int64
	PagesUnchanged int64
	PagesSkipped   int64
	PagesFailed    int64
	RecordsCreated int64
	RecordsUpdated int64
}

// LiveCounters are updated concurrently by crawl and LLM workers.
type LiveCounters struct {
	PagesFound, PagesNew, PagesChanged, PagesUnchanged, PagesSkipped, PagesFailed atomic.Int64
	RecordsCreated, RecordsUpdated                                                atomic.Int64
}

func (c *LiveCounters) Snapshot() Counters {
	return Counters{
		PagesFound: c.PagesFound.Load(), PagesNew: c.PagesNew.Load(), PagesChanged: c.PagesChanged.Load(),
		PagesUnchanged: c.PagesUnchanged.Load(), PagesSkipped: c.PagesSkipped.Load(),
		PagesFailed: c.PagesFailed.Load(), RecordsCreated: c.RecordsCreated.Load(),
		RecordsUpdated: c.RecordsUpdated.Load(),
	}
}

type ListFilter struct {
	SiteID *int64
	Status *string
	Cursor int64
	Limit  int
}
