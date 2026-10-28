package service

import (
	"context"
	"time"

	"ggkip/src/main/modules/site/model"
	"ggkip/src/main/modules/site/repo"
)

// maxErrorLen keeps last_error readable; full errors are in the logs.
const maxErrorLen = 2000

type Service struct {
	repo *repo.Repo
	now  func() time.Time
	// failureRetry is how long a failed site waits before the scheduler picks
	// it again (never longer than the site's own crawl interval).
	failureRetry time.Duration
}

func New(r *repo.Repo, failureRetry time.Duration) *Service {
	return &Service{repo: r, now: time.Now, failureRetry: failureRetry}
}

func (s *Service) List(ctx context.Context, f model.ListFilter) ([]model.Site, error) {
	return s.repo.List(ctx, f)
}

func (s *Service) Get(ctx context.Context, id int64) (model.Site, error) {
	return s.repo.Get(ctx, id)
}

func (s *Service) Due(ctx context.Context, limit int) ([]model.Site, error) {
	return s.repo.Due(ctx, s.now(), limit)
}

// RecordSuccess sets last_crawled_at = last_success_at = at and schedules the
// next crawl one interval later.
func (s *Service) RecordSuccess(ctx context.Context, site model.Site, at time.Time) error {
	return s.repo.MarkSuccess(ctx, site.ID, at, NextCrawlAt(at, site.CrawlInterval()))
}

// RecordFailure stores the error and schedules a retry, so one broken site is
// neither hammered every scheduler tick nor dropped.
func (s *Service) RecordFailure(ctx context.Context, site model.Site, at time.Time, cause error) error {
	msg := cause.Error()
	if len(msg) > maxErrorLen {
		msg = msg[:maxErrorLen]
	}
	return s.repo.MarkFailure(ctx, site.ID, at, RetryAt(at, site.CrawlInterval(), s.failureRetry), msg)
}

// NextCrawlAt is the next scheduled crawl after a successful one.
func NextCrawlAt(at time.Time, interval time.Duration) time.Time {
	return at.Add(interval)
}

// RetryAt is the next attempt after a failure: the retry delay, capped by the
// site's normal crawl interval.
func RetryAt(at time.Time, interval, retry time.Duration) time.Time {
	if retry <= 0 || retry > interval {
		retry = interval
	}
	return at.Add(retry)
}
