package app

import (
	"context"
	"errors"
	"io"
	"log/slog"
	"os"
	"strings"
	"testing"
	"testing/fstest"
	"time"

	crawlmodel "ggkip/src/main/modules/crawl/model"
	crawlservice "ggkip/src/main/modules/crawl/service"
	sitemodel "ggkip/src/main/modules/site/model"
)

var quietLog = slog.New(slog.NewTextHandler(io.Discard, nil))

type fakeDue struct {
	sites []sitemodel.Site
	limit int
	err   error
}

func (f *fakeDue) Due(_ context.Context, limit int) ([]sitemodel.Site, error) {
	f.limit = limit
	return f.sites, f.err
}

// fakeStarter behaves like the runner: some sites are already running and
// only `capacity` new crawls fit.
type fakeStarter struct {
	running  map[int64]bool
	capacity int
	started  []int64
}

func (f *fakeStarter) Start(site sitemodel.Site) (crawlmodel.CrawlRun, error) {
	if f.running[site.ID] {
		return crawlmodel.CrawlRun{}, crawlservice.ErrAlreadyRunning
	}
	if len(f.started) >= f.capacity {
		return crawlmodel.CrawlRun{}, crawlservice.ErrBusy
	}
	f.started = append(f.started, site.ID)
	return crawlmodel.CrawlRun{ID: int64(100 + len(f.started)), SiteID: site.ID}, nil
}

func TestSchedulerStartsDueSites(t *testing.T) {
	due := &fakeDue{sites: []sitemodel.Site{{ID: 1}, {ID: 2}, {ID: 3}, {ID: 4}}}
	starter := &fakeStarter{running: map[int64]bool{2: true}, capacity: 2}
	s := NewScheduler(due, starter, time.Minute, 2, quietLog)

	if n := s.Tick(context.Background()); n != 2 {
		t.Fatalf("started %d crawls, want 2", n)
	}
	if got := starter.started; len(got) != 2 || got[0] != 1 || got[1] != 3 {
		t.Errorf("started %v, want [1 3]: running site 2 is skipped, site 4 waits for capacity", got)
	}
	if due.limit != 8 {
		t.Errorf("due batch = %d, want 4x max sites", due.limit)
	}
}

func TestSchedulerSurvivesErrors(t *testing.T) {
	s := NewScheduler(&fakeDue{err: errors.New("db down")}, &fakeStarter{capacity: 1}, time.Minute, 1, quietLog)
	if n := s.Tick(context.Background()); n != 0 {
		t.Fatalf("started %d crawls on a failing query", n)
	}
}

func TestSchedulerRunStopsOnCancel(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	due := &fakeDue{}
	done := make(chan struct{})
	go func() {
		NewScheduler(due, &fakeStarter{capacity: 1}, 10*time.Millisecond, 1, quietLog).Run(ctx)
		close(done)
	}()
	time.Sleep(30 * time.Millisecond)
	cancel()
	select {
	case <-done:
	case <-time.After(time.Second):
		t.Fatal("scheduler did not stop after cancel")
	}
}

func TestRealResourcesLoad(t *testing.T) {
	res := os.DirFS("../../resources")
	if _, err := LoadSettings(res); err != nil {
		t.Fatalf("settings: %v", err)
	}
	d, err := LoadSeed(res)
	if err != nil {
		t.Fatalf("seed: %v", err)
	}
	if len(d.Countries) < 200 || len(d.Sites) == 0 || len(d.Organizations) == 0 || len(d.Categories) == 0 {
		t.Fatalf("seed looks incomplete: %d countries, %d orgs, %d sites, %d categories",
			len(d.Countries), len(d.Organizations), len(d.Sites), len(d.Categories))
	}
}

func TestKeepCountries(t *testing.T) {
	d, err := LoadSeed(os.DirFS("../../resources"))
	if err != nil {
		t.Fatal(err)
	}
	countries := len(d.Countries)
	if err := d.KeepCountries([]string{"UZ"}); err != nil {
		t.Fatal(err)
	}
	if len(d.Countries) != countries || len(d.Organizations) == 0 || len(d.Sites) == 0 {
		t.Fatalf("got %d countries, %d orgs, %d sites", len(d.Countries), len(d.Organizations), len(d.Sites))
	}
	for _, s := range d.Sites {
		if s.CountryID != 242 {
			t.Fatalf("site %s of country %d kept", s.Domain, s.CountryID)
		}
	}
	if err := d.KeepCountries([]string{"XX"}); err == nil {
		t.Fatal("unknown country accepted")
	}
}

func seedFS(orgs, sites string) fstest.MapFS {
	return fstest.MapFS{
		"seed/countries.json": {Data: []byte(`[
			{"id": 1, "iso2": "UZ", "iso3": "UZB", "name": "Uzbekistan", "is_active": true},
			{"id": 2, "iso2": "JP", "iso3": "JPN", "name": "Japan", "is_active": true}]`)},
		"seed/organization_types.json": {Data: []byte(`[{"id": 1, "code": "ministry", "name": "Ministry"}]`)},
		"seed/categories.json": {Data: []byte(`[
			{"id": 2, "parent_id": 1, "code": "FIN.TAX", "name": "Taxation"},
			{"id": 1, "parent_id": null, "code": "FIN", "name": "Finance"}]`)},
		"seed/organizations.json": {Data: []byte(orgs)},
		"seed/sites.json":         {Data: []byte(sites)},
	}
}

const goodOrgs = `[
	{"id": 11, "country_id": 1, "organization_type_id": 1, "parent_id": 10, "name": "Tax Committee"},
	{"id": 10, "country_id": 1, "organization_type_id": 1, "name": "Ministry of Finance"}]`

func TestLoadSeedValidOrdersParentsFirst(t *testing.T) {
	d, err := LoadSeed(seedFS(goodOrgs, `[{"country_id": 1, "organization_id": 11, "domain": "soliq.uz",
		"base_url": "https://soliq.uz/", "is_official": true, "verification_score": 1, "crawl_interval_sec": 3600}]`))
	if err != nil {
		t.Fatal(err)
	}
	if d.Organizations[0].ID != 10 || d.Categories[0].Code != "FIN" {
		t.Errorf("parents must come first: orgs=%d,%d cats=%s,%s",
			d.Organizations[0].ID, d.Organizations[1].ID, d.Categories[0].Code, d.Categories[1].Code)
	}
	if d.Sites[0].Status != "active" || d.Organizations[0].Status != "active" {
		t.Error("missing status must default to active")
	}
}

func TestLoadSeedReportsBrokenReferences(t *testing.T) {
	cases := map[string]struct{ orgs, sites, want string }{
		"site with unknown organization": {goodOrgs,
			`[{"country_id": 1, "organization_id": 999, "domain": "a.uz", "base_url": "https://a.uz/", "crawl_interval_sec": 60}]`,
			"organization_id 999 not found"},
		"site country differs from organization": {goodOrgs,
			`[{"country_id": 2, "organization_id": 10, "domain": "a.uz", "base_url": "https://a.uz/", "crawl_interval_sec": 60}]`,
			"belongs to country 1, not 2"},
		"base_url on another domain": {goodOrgs,
			`[{"country_id": 1, "organization_id": 10, "domain": "a.uz", "base_url": "https://b.uz/", "crawl_interval_sec": 60}]`,
			"base_url"},
		"duplicate domain": {goodOrgs,
			`[{"country_id": 1, "organization_id": 10, "domain": "a.uz", "base_url": "https://a.uz/", "crawl_interval_sec": 60},
			  {"country_id": 1, "organization_id": 10, "domain": "a.uz", "base_url": "https://a.uz/", "crawl_interval_sec": 60}]`,
			"duplicated"},
		"organization with unknown country": {`[{"id": 10, "country_id": 77, "organization_type_id": 1, "name": "X"}]`, `[]`,
			"country_id 77 not found"},
		"organization parent cycle": {`[
			{"id": 10, "country_id": 1, "organization_type_id": 1, "parent_id": 11, "name": "A"},
			{"id": 11, "country_id": 1, "organization_type_id": 1, "parent_id": 10, "name": "B"}]`, `[]`,
			"cycle"},
	}
	for name, c := range cases {
		_, err := LoadSeed(seedFS(c.orgs, c.sites))
		if err == nil || !strings.Contains(err.Error(), c.want) {
			t.Errorf("%s: error %v, want it to mention %q", name, err, c.want)
		}
	}
}
