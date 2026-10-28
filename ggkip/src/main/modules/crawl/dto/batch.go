package dto

import (
	"time"

	"ggkip/src/main/modules/crawl/model"
)

// StartBatchRequest is the optional body of POST /v1/crawl-batches.
type StartBatchRequest struct {
	CountryIDs []int64 `json:"country_ids"` // empty or missing = all active countries
}

type BatchResponse struct {
	ID             int64      `json:"id"`
	Status         string     `json:"status"`
	CountryIDs     []int64    `json:"country_ids"`
	CountriesTotal int        `json:"countries_total"`
	SitesTotal     int        `json:"sites_total"`
	SitesSkipped   int        `json:"sites_skipped"`
	StartedAt      time.Time  `json:"started_at"`
	FinishedAt     *time.Time `json:"finished_at"`
	ErrorMessage   *string    `json:"error_message"`
	CreatedAt      time.Time  `json:"created_at"`

	// Detail only (GET /v1/crawl-batches/{id}).
	Runs         map[string]int64 `json:"runs,omitempty"`          // crawl runs per status
	SitesPending *int64           `json:"sites_pending,omitempty"` // not started yet (or never, when cancelled)
}

func BatchFromModel(b model.Batch) BatchResponse {
	ids := b.CountryIDs
	if ids == nil {
		ids = []int64{}
	}
	return BatchResponse{
		ID: b.ID, Status: b.Status, CountryIDs: ids, CountriesTotal: b.CountriesTotal, SitesTotal: b.SitesTotal,
		SitesSkipped: b.SitesSkipped, StartedAt: b.StartedAt, FinishedAt: b.FinishedAt,
		ErrorMessage: b.ErrorMessage, CreatedAt: b.CreatedAt,
	}
}

// BatchDetail adds the run counts (every run status, 0 when absent) and the
// number of sites that have no run yet.
func BatchDetail(b model.Batch, counts model.RunCounts) BatchResponse {
	out := BatchFromModel(b)
	out.Runs = map[string]int64{}
	var runs int64
	for _, s := range []string{model.StatusRunning, model.StatusCompleted, model.StatusPartial,
		model.StatusFailed, model.StatusCancelled} {
		out.Runs[s] = counts[s]
	}
	for _, n := range counts {
		runs += n
	}
	pending := max(0, int64(b.SitesTotal-b.SitesSkipped)-runs)
	out.SitesPending = &pending
	return out
}
