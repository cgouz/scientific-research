package model

import "time"

// Batch is one "crawl every active country" request. Each country is crawled
// in its own goroutine; the per-site results are the crawl runs with its ID.
type Batch struct {
	ID             int64
	Status         string
	CountryIDs     []int64 // the request filter; empty = all active countries
	CountriesTotal int
	SitesTotal     int
	SitesSkipped   int // already running elsewhere, so not crawled by this batch
	StartedAt      time.Time
	FinishedAt     *time.Time
	ErrorMessage   *string
	CreatedAt      time.Time
}

// Batch statuses. A completed batch can still have failed or partial runs;
// the per-site outcome is in RunCounts.
const (
	BatchRunning   = "running"
	BatchCompleted = "completed"
	BatchCancelled = "cancelled"
	BatchFailed    = "failed" // the batch itself broke (e.g. a panic in its coordinator)
)

// RunCounts is the number of a batch's crawl runs per run status.
type RunCounts map[string]int64

type BatchListFilter struct {
	Status *string
	Cursor int64
	Limit  int
}
