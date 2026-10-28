package service

import (
	"context"

	"ggkip/src/main/modules/crawl/model"
	"ggkip/src/main/modules/crawl/repo"
	sitemodel "ggkip/src/main/modules/site/model"
)

type SiteGetter interface {
	Get(ctx context.Context, id int64) (sitemodel.Site, error)
}

// Service serves crawl history and manual crawl triggers.
type Service struct {
	repo   *repo.Repo
	sites  SiteGetter
	runner *Runner
}

func New(r *repo.Repo, sites SiteGetter, runner *Runner) *Service {
	return &Service{repo: r, sites: sites, runner: runner}
}

func (s *Service) List(ctx context.Context, f model.ListFilter) ([]model.CrawlRun, error) {
	return s.repo.List(ctx, f)
}

func (s *Service) Get(ctx context.Context, id int64) (model.CrawlRun, error) {
	return s.repo.Get(ctx, id)
}

// TriggerSite starts a crawl now, regardless of next_crawl_at. Inactive or
// unofficial sites can be crawled this way on purpose.
func (s *Service) TriggerSite(ctx context.Context, siteID int64) (model.CrawlRun, error) {
	site, err := s.sites.Get(ctx, siteID)
	if err != nil {
		return model.CrawlRun{}, err
	}
	return s.runner.Start(site)
}
