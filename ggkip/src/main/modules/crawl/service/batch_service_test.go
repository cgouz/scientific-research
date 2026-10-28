package service

import (
	"context"
	"errors"
	"io"
	"log/slog"
	"sync"
	"testing"
	"time"

	"ggkip/src/main/infrastructure"
	"ggkip/src/main/modules/crawl/model"
	sitemodel "ggkip/src/main/modules/site/model"
)

var quietLog = slog.New(slog.NewTextHandler(io.Discard, nil))

// fakeStore keeps batches in memory.
type fakeStore struct {
	mu       sync.Mutex
	batches  map[int64]model.Batch
	finished chan model.Batch
}

func newFakeStore() *fakeStore {
	return &fakeStore{batches: map[int64]model.Batch{}, finished: make(chan model.Batch, 10)}
}

func (f *fakeStore) Create(_ context.Context, b model.Batch) (model.Batch, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	b.ID, b.Status = int64(len(f.batches)+1), model.BatchRunning
	f.batches[b.ID] = b
	return b, nil
}

func (f *fakeStore) Finish(_ context.Context, id int64, status string, at time.Time, skipped int, errMsg *string) error {
	f.mu.Lock()
	b := f.batches[id]
	b.Status, b.FinishedAt, b.SitesSkipped, b.ErrorMessage = status, &at, skipped, errMsg
	f.batches[id] = b
	f.mu.Unlock()
	f.finished <- b
	return nil
}

func (f *fakeStore) Get(_ context.Context, id int64) (model.Batch, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	b, ok := f.batches[id]
	if !ok {
		return b, infrastructure.ErrNotFound
	}
	return b, nil
}

func (f *fakeStore) List(context.Context, model.BatchListFilter) ([]model.Batch, error) {
	return nil, nil
}
func (f *fakeStore) RunCounts(context.Context, int64) (model.RunCounts, error) { return nil, nil }

func (f *fakeStore) waitFinished(t *testing.T) model.Batch {
	t.Helper()
	select {
	case b := <-f.finished:
		return b
	case <-time.After(5 * time.Second):
		t.Fatal("batch did not finish")
		return model.Batch{}
	}
}

type fakeSites []sitemodel.Site

func (f fakeSites) Crawlable(_ context.Context, countryIDs []int64) ([]sitemodel.Site, error) {
	if len(countryIDs) == 0 {
		return f, nil
	}
	var out []sitemodel.Site
	for _, s := range f {
		for _, id := range countryIDs {
			if s.CountryID == id {
				out = append(out, s)
			}
		}
	}
	return out, nil
}

// fakeRunner records concurrency per country; run decides each site's outcome.
type fakeRunner struct {
	mu         sync.Mutex
	inFlight   map[int64]int // per country
	maxCountry map[int64]int
	total      int
	maxTotal   int
	ran        []int64
	run        func(ctx context.Context, site sitemodel.Site) error
}

func newFakeRunner(run func(ctx context.Context, site sitemodel.Site) error) *fakeRunner {
	return &fakeRunner{inFlight: map[int64]int{}, maxCountry: map[int64]int{}, run: run}
}

func (f *fakeRunner) Run(ctx context.Context, site sitemodel.Site, batchID *int64) (model.CrawlRun, error) {
	f.mu.Lock()
	f.inFlight[site.CountryID]++
	f.total++
	f.maxCountry[site.CountryID] = max(f.maxCountry[site.CountryID], f.inFlight[site.CountryID])
	f.maxTotal = max(f.maxTotal, f.total)
	f.ran = append(f.ran, site.ID)
	f.mu.Unlock()
	defer func() {
		f.mu.Lock()
		f.inFlight[site.CountryID]--
		f.total--
		f.mu.Unlock()
	}()
	return model.CrawlRun{SiteID: site.ID, BatchID: batchID}, f.run(ctx, site)
}

func sitesOf(perCountry map[int64]int) fakeSites {
	var out fakeSites
	id := int64(1)
	for _, country := range []int64{1, 2, 3} {
		for range perCountry[country] {
			out = append(out, sitemodel.Site{ID: id, CountryID: country, Domain: "site.example"})
			id++
		}
	}
	return out
}

func sleepRun(d time.Duration) func(context.Context, sitemodel.Site) error {
	return func(ctx context.Context, _ sitemodel.Site) error {
		select {
		case <-time.After(d):
		case <-ctx.Done():
		}
		return nil
	}
}

func TestBatchCrawlsCountriesInParallelWithinLimits(t *testing.T) {
	store, runner := newFakeStore(), newFakeRunner(sleepRun(20*time.Millisecond))
	sites := sitesOf(map[int64]int{1: 5, 2: 3, 3: 2})
	svc := NewBatchService(context.Background(), store, sites, runner, 2, quietLog)

	batch, err := svc.Start(context.Background(), nil)
	if err != nil {
		t.Fatal(err)
	}
	if batch.CountriesTotal != 3 || batch.SitesTotal != 10 {
		t.Fatalf("batch = %+v, want 3 countries and 10 sites", batch)
	}
	done := store.waitFinished(t)
	svc.Wait()
	if done.Status != model.BatchCompleted || len(runner.ran) != 10 {
		t.Fatalf("status %s, %d sites crawled; want completed, 10", done.Status, len(runner.ran))
	}
	for country, n := range runner.maxCountry {
		if n > 2 {
			t.Errorf("country %d had %d sites at once, limit is 2", country, n)
		}
	}
	if runner.maxTotal <= 2 {
		t.Errorf("countries must run in parallel: at most %d sites ran at once", runner.maxTotal)
	}
}

func TestBatchSurvivesPanickingSites(t *testing.T) {
	store := newFakeStore()
	runner := newFakeRunner(func(_ context.Context, site sitemodel.Site) error {
		switch site.ID {
		case 2:
			panic("extractor bug")
		case 6:
			var m map[int64]bool
			m[site.ID] = true // runtime error
		}
		return nil
	})
	svc := NewBatchService(context.Background(), store, sitesOf(map[int64]int{1: 3, 2: 3}), runner, 1, quietLog)
	if _, err := svc.Start(context.Background(), nil); err != nil {
		t.Fatal(err)
	}
	done := store.waitFinished(t)
	svc.Wait()
	if done.Status != model.BatchCompleted || done.ErrorMessage != nil {
		t.Fatalf("a panicking site must not fail the batch: %+v", done)
	}
	if len(runner.ran) != 6 {
		t.Fatalf("crawled %v, want all 6 sites: after a panic the country goes on", runner.ran)
	}
	if runner.maxCountry[1] != 1 || runner.maxCountry[2] != 1 {
		t.Errorf("the per-country slot must be freed after a panic: %v", runner.maxCountry)
	}
}

func TestBatchSkipsRunningSitesAndAllowsOneBatch(t *testing.T) {
	store := newFakeStore()
	release := make(chan struct{})
	runner := newFakeRunner(func(ctx context.Context, site sitemodel.Site) error {
		if site.ID == 1 {
			return ErrAlreadyRunning // e.g. the scheduler crawls it right now
		}
		<-release
		return nil
	})
	svc := NewBatchService(context.Background(), store, sitesOf(map[int64]int{1: 2}), runner, 4, quietLog)
	first, err := svc.Start(context.Background(), nil)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := svc.Start(context.Background(), nil); !errors.Is(err, ErrBatchRunning) {
		t.Fatalf("second Start = %v, want ErrBatchRunning", err)
	}
	close(release)
	done := store.waitFinished(t)
	svc.Wait()
	if done.SitesSkipped != 1 {
		t.Errorf("sites_skipped = %d, want 1", done.SitesSkipped)
	}
	if _, err := svc.Cancel(context.Background(), first.ID); !errors.Is(err, ErrBatchNotRunning) {
		t.Errorf("cancel of a finished batch = %v, want ErrBatchNotRunning", err)
	}
	if _, err := svc.Start(context.Background(), nil); err != nil {
		t.Errorf("a new batch must start after the first finished: %v", err)
	}
	store.waitFinished(t)
	svc.Wait()
}

func TestBatchCancelStopsWaitingSites(t *testing.T) {
	store, runner := newFakeStore(), newFakeRunner(sleepRun(time.Hour)) // only cancel ends a run
	svc := NewBatchService(context.Background(), store, sitesOf(map[int64]int{1: 5}), runner, 1, quietLog)
	batch, err := svc.Start(context.Background(), nil)
	if err != nil {
		t.Fatal(err)
	}
	for {
		runner.mu.Lock()
		started := len(runner.ran)
		runner.mu.Unlock()
		if started > 0 {
			break
		}
		time.Sleep(time.Millisecond)
	}
	if _, err := svc.Cancel(context.Background(), batch.ID); err != nil {
		t.Fatal(err)
	}
	done := store.waitFinished(t)
	svc.Wait()
	if done.Status != model.BatchCancelled || len(runner.ran) != 1 {
		t.Fatalf("status %s, %d sites started; want cancelled and only the first one", done.Status, len(runner.ran))
	}
}

func TestBatchStopsWithTheApplication(t *testing.T) {
	appCtx, stop := context.WithCancel(context.Background())
	store, runner := newFakeStore(), newFakeRunner(sleepRun(time.Hour))
	svc := NewBatchService(appCtx, store, sitesOf(map[int64]int{1: 2, 2: 2}), runner, 1, quietLog)
	if _, err := svc.Start(context.Background(), nil); err != nil {
		t.Fatal(err)
	}
	stop()
	if done := store.waitFinished(t); done.Status != model.BatchCancelled {
		t.Fatalf("status %s, want cancelled on shutdown", done.Status)
	}
	svc.Wait()
	if _, err := svc.Start(context.Background(), nil); err == nil {
		t.Error("no batch may start while shutting down")
	}
}

func TestBatchFiltersAndRejectsEmpty(t *testing.T) {
	store, runner := newFakeStore(), newFakeRunner(sleepRun(0))
	svc := NewBatchService(context.Background(), store, sitesOf(map[int64]int{1: 2, 2: 3}), runner, 2, quietLog)
	b, err := svc.Start(context.Background(), []int64{2})
	if err != nil {
		t.Fatal(err)
	}
	if b.CountriesTotal != 1 || b.SitesTotal != 3 {
		t.Fatalf("filtered batch = %+v, want 1 country with 3 sites", b)
	}
	store.waitFinished(t)
	svc.Wait()

	_, err = svc.Start(context.Background(), []int64{99})
	if !errors.Is(err, ErrNoSites) || !errors.Is(err, infrastructure.ErrInvalidInput) {
		t.Fatalf("no sites = %v, want ErrNoSites (a 400)", err)
	}
}

func TestGroupByCountryKeepsOrder(t *testing.T) {
	groups := groupByCountry([]sitemodel.Site{{ID: 1, CountryID: 5}, {ID: 2, CountryID: 5}, {ID: 3, CountryID: 7}})
	if len(groups) != 2 || groups[0].countryID != 5 || len(groups[0].sites) != 2 || groups[1].sites[0].ID != 3 {
		t.Fatalf("groups = %+v", groups)
	}
}
