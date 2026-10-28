package app

import (
	"context"
	"errors"
	"log/slog"
	"time"

	crawlmodel "ggkip/src/main/modules/crawl/model"
	crawlservice "ggkip/src/main/modules/crawl/service"
	sitemodel "ggkip/src/main/modules/site/model"
)

type DueSites interface {
	Due(ctx context.Context, limit int) ([]sitemodel.Site, error)
}

type CrawlStarter interface {
	Start(site sitemodel.Site) (crawlmodel.CrawlRun, error)
}

// Scheduler checks for due sites on every tick and starts crawls for them
// until the runner is full. Sites that are still running are skipped.
type Scheduler struct {
	sites    DueSites
	runner   CrawlStarter
	interval time.Duration
	batch    int
	log      *slog.Logger
}

func NewScheduler(sites DueSites, runner CrawlStarter, interval time.Duration, maxSites int, log *slog.Logger) *Scheduler {
	return &Scheduler{sites: sites, runner: runner, interval: interval, batch: maxSites * 4, log: log}
}

// Run ticks until ctx is cancelled. The first tick runs immediately.
func (s *Scheduler) Run(ctx context.Context) {
	ticker := time.NewTicker(s.interval)
	defer ticker.Stop()
	for {
		s.Tick(ctx)
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
		}
	}
}

// Tick starts crawls for due sites and returns how many were started.
func (s *Scheduler) Tick(ctx context.Context) int {
	sites, err := s.sites.Due(ctx, s.batch)
	if err != nil {
		if ctx.Err() == nil {
			s.log.Error("scheduler: loading due sites failed", "error", err)
		}
		return 0
	}
	started := 0
	for _, site := range sites {
		run, err := s.runner.Start(site)
		switch {
		case err == nil:
			started++
			s.log.Info("scheduler: crawl started", "site_id", site.ID, "crawl_run_id", run.ID, "domain", site.Domain)
		case errors.Is(err, crawlservice.ErrAlreadyRunning):
			continue
		case errors.Is(err, crawlservice.ErrBusy):
			return started
		default:
			s.log.Error("scheduler: crawl not started", "site_id", site.ID, "error", err)
		}
	}
	return started
}
