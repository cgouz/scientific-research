package service

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"sync"
	"sync/atomic"
	"time"

	"ggkip/src/main/infrastructure"
	"ggkip/src/main/modules/crawl/model"
	sitemodel "ggkip/src/main/modules/site/model"
)

var (
	// ErrBatchRunning: only one crawl batch runs at a time.
	ErrBatchRunning = errors.New("a crawl batch is already running")
	// ErrBatchNotRunning: the batch exists but has already finished.
	ErrBatchNotRunning = errors.New("crawl batch is not running")
	// ErrNoSites: the filter matched no active official site of an active country.
	ErrNoSites = fmt.Errorf("%w: no active official sites in the selected active countries", infrastructure.ErrInvalidInput)
)

// What the batch service needs from other modules.
type (
	CrawlableSites interface {
		Crawlable(ctx context.Context, countryIDs []int64) ([]sitemodel.Site, error)
	}
	SiteRunner interface {
		Run(ctx context.Context, site sitemodel.Site, batchID *int64) (model.CrawlRun, error)
	}
	BatchStore interface {
		Create(ctx context.Context, b model.Batch) (model.Batch, error)
		Finish(ctx context.Context, id int64, status string, finishedAt time.Time, sitesSkipped int, errMsg *string) error
		Get(ctx context.Context, id int64) (model.Batch, error)
		List(ctx context.Context, f model.BatchListFilter) ([]model.Batch, error)
		RunCounts(ctx context.Context, batchID int64) (model.RunCounts, error)
	}
)

// BatchService crawls all active countries at once: one goroutine per
// country, and in each country up to sitesPerCountry sites in parallel, so a
// country with many sites cannot take every slot. The global limit is the
// runner's MaxSites: a site queues there for a free slot instead of being
// rejected. One batch runs at a time.
//
// No panic leaves this service: a panic while crawling a site fails that
// site, one in a country goroutine stops only that country, and one in the
// coordinator marks the batch failed. Every level still waits for the
// goroutines it started, so nothing keeps running unseen after a panic.
type BatchService struct {
	store           BatchStore
	sites           CrawlableSites
	runner          SiteRunner
	sitesPerCountry int
	ctx             context.Context // application lifetime; batches stop with it
	log             *slog.Logger
	now             func() time.Time

	mu     sync.Mutex
	active *activeBatch // nil while no batch runs
	wg     sync.WaitGroup
}

type activeBatch struct {
	id      int64
	cancel  context.CancelFunc
	skipped *atomic.Int64
}

type countrySites struct {
	countryID int64
	sites     []sitemodel.Site
}

func NewBatchService(ctx context.Context, store BatchStore, sites CrawlableSites, runner SiteRunner,
	sitesPerCountry int, log *slog.Logger) *BatchService {
	return &BatchService{
		store: store, sites: sites, runner: runner, sitesPerCountry: max(1, sitesPerCountry),
		ctx: ctx, log: log, now: time.Now,
	}
}

// Start stores a new batch for every active official site of the active
// countries (only countryIDs when given) and crawls it in the background.
func (s *BatchService) Start(ctx context.Context, countryIDs []int64) (model.Batch, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.active != nil {
		return model.Batch{}, ErrBatchRunning
	}
	if err := s.ctx.Err(); err != nil {
		return model.Batch{}, fmt.Errorf("shutting down: %w", err)
	}
	sites, err := s.sites.Crawlable(ctx, countryIDs)
	if err != nil {
		return model.Batch{}, err
	}
	if len(sites) == 0 {
		return model.Batch{}, ErrNoSites
	}
	groups := groupByCountry(sites)
	batch, err := s.store.Create(ctx, model.Batch{
		CountryIDs: countryIDs, CountriesTotal: len(groups), SitesTotal: len(sites), StartedAt: s.now(),
	})
	if err != nil {
		return batch, err
	}

	batchCtx, cancel := context.WithCancel(s.ctx)
	active := &activeBatch{id: batch.ID, cancel: cancel, skipped: new(atomic.Int64)}
	s.active = active
	s.wg.Go(func() {
		defer func() {
			s.mu.Lock()
			s.active = nil
			s.mu.Unlock()
			cancel()
		}()
		if err := infrastructure.Protect(func() { s.execute(batchCtx, batch, groups, active.skipped) }); err != nil {
			infrastructure.LogPanic(s.log, "crawl batch bookkeeping panicked", err, "crawl_batch_id", batch.ID)
		}
	})
	return batch, nil
}

// execute crawls every country in its own goroutine and closes the batch row.
func (s *BatchService) execute(ctx context.Context, batch model.Batch, groups []countrySites, skipped *atomic.Int64) {
	log := s.log.With("crawl_batch_id", batch.ID)
	start := time.Now()
	log.Info("crawl batch started", "countries", len(groups), "sites", batch.SitesTotal)

	err := infrastructure.Protect(func() {
		var countries sync.WaitGroup
		defer countries.Wait() // also after a panic in this loop: no country goroutine outlives the batch
		for _, g := range groups {
			countries.Go(func() {
				err := infrastructure.Protect(func() { s.crawlCountry(ctx, batch.ID, g, skipped, log) })
				if err != nil {
					infrastructure.LogPanic(log, "country crawl panicked, its remaining sites are not crawled", err,
						"country_id", g.countryID)
				}
			})
		}
	})

	status, errMsg := model.BatchCompleted, (*string)(nil)
	switch {
	case err != nil:
		infrastructure.LogPanic(log, "crawl batch panicked", err)
		msg := err.Error()
		status, errMsg = model.BatchFailed, &msg
	case ctx.Err() != nil:
		status = model.BatchCancelled
	}
	// Bookkeeping must survive shutdown and cancel, so it does not use ctx.
	finishCtx, cancel := context.WithTimeout(context.WithoutCancel(ctx), 15*time.Second)
	defer cancel()
	if err := s.store.Finish(finishCtx, batch.ID, status, s.now(), int(skipped.Load()), errMsg); err != nil {
		log.Error("could not finish crawl batch", "error", err)
	}
	log.Info("crawl batch finished", "status", status, "duration", time.Since(start), "sites_skipped", skipped.Load())
}

// crawlCountry crawls one country's sites, at most sitesPerCountry at once;
// each site then also waits for one of the runner's global slots.
func (s *BatchService) crawlCountry(ctx context.Context, batchID int64, g countrySites, skipped *atomic.Int64, log *slog.Logger) {
	log = log.With("country_id", g.countryID)
	slots := make(chan struct{}, s.sitesPerCountry)
	var sites sync.WaitGroup
	defer sites.Wait() // also after a panic: no site goroutine outlives its country
	for _, site := range g.sites {
		select {
		case slots <- struct{}{}:
		case <-ctx.Done():
		}
		// select picks at random when both are ready, so check again
		if ctx.Err() != nil {
			return // cancelled: the remaining sites are not started
		}
		sites.Go(func() {
			defer func() { <-slots }()
			s.crawlSite(ctx, batchID, site, skipped, log)
		})
	}
}

// crawlSite runs one site through the runner. The runner handles its own
// panics; the guard here is the last line, so a bug cannot stop the batch.
func (s *BatchService) crawlSite(ctx context.Context, batchID int64, site sitemodel.Site, skipped *atomic.Int64, log *slog.Logger) {
	var err error
	if p := infrastructure.Protect(func() { _, err = s.runner.Run(ctx, site, &batchID) }); p != nil {
		infrastructure.LogPanic(log, "site crawl panicked", p, "site_id", site.ID, "domain", site.Domain)
		return
	}
	switch {
	case err == nil:
	case errors.Is(err, ErrAlreadyRunning):
		skipped.Add(1)
		log.Info("site skipped: already being crawled", "site_id", site.ID, "domain", site.Domain)
	case ctx.Err() != nil:
		// cancelled while waiting for a crawl slot
	default:
		log.Error("site crawl not started", "site_id", site.ID, "domain", site.Domain, "error", err)
	}
}

// Cancel stops a running batch: sites still waiting are not started and
// running crawls end as 'cancelled'. It returns before they have stopped.
func (s *BatchService) Cancel(ctx context.Context, id int64) (model.Batch, error) {
	s.mu.Lock()
	active := s.active
	s.mu.Unlock()
	if active == nil || active.id != id {
		b, err := s.store.Get(ctx, id)
		if err != nil {
			return b, err
		}
		return b, ErrBatchNotRunning
	}
	active.cancel()
	return s.store.Get(ctx, id)
}

// Get returns the batch and how many of its runs are in each status. For the
// running batch, SitesSkipped is the live count.
func (s *BatchService) Get(ctx context.Context, id int64) (model.Batch, model.RunCounts, error) {
	b, err := s.store.Get(ctx, id)
	if err != nil {
		return b, nil, err
	}
	counts, err := s.store.RunCounts(ctx, id)
	if err != nil {
		return b, nil, err
	}
	s.mu.Lock()
	if s.active != nil && s.active.id == id {
		b.SitesSkipped = int(s.active.skipped.Load())
	}
	s.mu.Unlock()
	return b, counts, nil
}

func (s *BatchService) List(ctx context.Context, f model.BatchListFilter) ([]model.Batch, error) {
	return s.store.List(ctx, f)
}

// Wait blocks until the running batch (if any) has finished. Call it after
// the application context is cancelled and the HTTP server has stopped.
func (s *BatchService) Wait() { s.wg.Wait() }

// groupByCountry keeps the order of sites (by country, then id).
func groupByCountry(sites []sitemodel.Site) []countrySites {
	var out []countrySites
	index := map[int64]int{}
	for _, site := range sites {
		i, ok := index[site.CountryID]
		if !ok {
			i = len(out)
			index[site.CountryID] = i
			out = append(out, countrySites{countryID: site.CountryID})
		}
		out[i].sites = append(out[i].sites, site)
	}
	return out
}
