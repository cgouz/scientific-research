package dto

import (
	"testing"

	"ggkip/src/main/modules/crawl/model"
)

func TestBatchDetailCountsRunsAndPending(t *testing.T) {
	b := model.Batch{ID: 1, Status: model.BatchRunning, SitesTotal: 10, SitesSkipped: 1}
	out := BatchDetail(b, model.RunCounts{model.StatusCompleted: 3, model.StatusRunning: 2})
	if out.Runs[model.StatusFailed] != 0 || len(out.Runs) != 5 {
		t.Errorf("every run status must be listed: %v", out.Runs)
	}
	if out.SitesPending == nil || *out.SitesPending != 4 {
		t.Errorf("pending = %v, want 10 - 1 skipped - 5 runs = 4", out.SitesPending)
	}
	if out.CountryIDs == nil {
		t.Error("country_ids must be [] not null")
	}
}
