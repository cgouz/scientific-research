package service

import (
	"context"
	"database/sql"
	"encoding/json"
	"errors"
	"log/slog"
	"net/url"
	"sync"
	"time"

	"ggkip/src/main/infrastructure"
	categorymodel "ggkip/src/main/modules/category/model"
	countrymodel "ggkip/src/main/modules/country/model"
	"ggkip/src/main/modules/crawl/model"
	crawlrepo "ggkip/src/main/modules/crawl/repo"
	crawlermodel "ggkip/src/main/modules/crawler/model"
	crawlerservice "ggkip/src/main/modules/crawler/service"
	extractormodel "ggkip/src/main/modules/extractor/model"
	extractorservice "ggkip/src/main/modules/extractor/service"
	llmdto "ggkip/src/main/modules/llm/dto"
	llmmodel "ggkip/src/main/modules/llm/model"
	llmservice "ggkip/src/main/modules/llm/service"
	organizationmodel "ggkip/src/main/modules/organization/model"
	pagemodel "ggkip/src/main/modules/page/model"
	pagerepo "ggkip/src/main/modules/page/repo"
	pageservice "ggkip/src/main/modules/page/service"
	recordmodel "ggkip/src/main/modules/record/model"
	recordrepo "ggkip/src/main/modules/record/repo"
	sitemodel "ggkip/src/main/modules/site/model"
)

var (
	// ErrAlreadyRunning: the site is being crawled right now.
	ErrAlreadyRunning = errors.New("site is already being crawled")
	// ErrBusy: the maximum number of concurrent site crawls is reached.
	ErrBusy = errors.New("maximum number of concurrent crawls reached")
)

// Lookups the runner needs from other modules.
type (
	SiteTracker interface {
		RecordSuccess(ctx context.Context, site sitemodel.Site, at time.Time) error
		RecordFailure(ctx context.Context, site sitemodel.Site, at time.Time, cause error) error
	}
	CountryGetter interface {
		Get(ctx context.Context, id int64) (countrymodel.Country, error)
	}
	OrganizationGetter interface {
		Get(ctx context.Context, id int64) (organizationmodel.Organization, error)
	}
	TaxonomySource interface {
		Active(ctx context.Context) ([]categorymodel.Category, error)
	}
)

type RunnerDeps struct {
	DB               *sql.DB
	Sites            SiteTracker
	Countries        CountryGetter
	Organizations    OrganizationGetter
	Categories       TaxonomySource
	Crawler          *crawlerservice.Crawler
	Normalizer       *llmservice.Normalizer
	TaxonomySettings llmmodel.TaxonomySettings
	Filter           extractorservice.Filter
	LLMWorkers       int
	MaxSites         int
	Log              *slog.Logger
}

// Runner executes crawl runs in the background: at most MaxSites at once and
// never two for the same site. Its context is the application lifetime;
// cancelling it stops all runs.
type Runner struct {
	deps    RunnerDeps
	runs    *crawlrepo.Repo
	pages   *pagerepo.Repo
	ctx     context.Context
	sem     chan struct{}
	mu      sync.Mutex
	running map[int64]bool
	wg      sync.WaitGroup
}

func NewRunner(ctx context.Context, deps RunnerDeps) *Runner {
	return &Runner{
		deps:    deps,
		runs:    crawlrepo.New(deps.DB),
		pages:   pagerepo.New(deps.DB),
		ctx:     ctx,
		sem:     make(chan struct{}, deps.MaxSites),
		running: map[int64]bool{},
	}
}

// Start creates the crawl run row and crawls the site in the background.
// It returns ErrAlreadyRunning or ErrBusy instead of waiting.
func (r *Runner) Start(site sitemodel.Site) (model.CrawlRun, error) {
	r.mu.Lock()
	if r.running[site.ID] {
		r.mu.Unlock()
		return model.CrawlRun{}, ErrAlreadyRunning
	}
	select {
	case r.sem <- struct{}{}:
	default:
		r.mu.Unlock()
		return model.CrawlRun{}, ErrBusy
	}
	r.running[site.ID] = true
	r.mu.Unlock()

	release := func() {
		r.mu.Lock()
		delete(r.running, site.ID)
		r.mu.Unlock()
		<-r.sem
	}
	run, err := r.runs.Create(r.ctx, site.ID, time.Now())
	if err != nil {
		release()
		return run, err
	}
	r.wg.Go(func() {
		defer release()
		r.execute(site, run)
	})
	return run, nil
}

// Wait blocks until every started run has finished.
func (r *Runner) Wait() { r.wg.Wait() }

// pageJob is a new or changed page waiting for the LLM.
type pageJob struct {
	page    pagemodel.Page
	change  pageservice.Change
	url     string
	content extractormodel.Content
	hash    []byte
}

// run holds what one crawl run needs while it is executing.
type run struct {
	site     sitemodel.Site
	runID    int64
	country  countrymodel.Country
	org      organizationmodel.Organization
	taxonomy *llmmodel.Taxonomy
	counters model.LiveCounters
	jobs     chan pageJob
}

func (r *Runner) execute(site sitemodel.Site, crawlRun model.CrawlRun) {
	ctx := r.ctx
	log := r.deps.Log.With("site_id", site.ID, "crawl_run_id", crawlRun.ID, "domain", site.Domain)
	start := time.Now()
	log.Info("crawl started")

	rn, err := r.prepare(ctx, site, crawlRun.ID)
	var crawlErr error
	if err != nil {
		crawlErr = err
	} else {
		var workers sync.WaitGroup
		for range r.deps.LLMWorkers {
			workers.Go(func() {
				for job := range rn.jobs {
					r.normalizeAndStore(ctx, rn, job, log)
				}
			})
		}
		_, crawlErr = r.deps.Crawler.Crawl(ctx, crawlermodel.Target{Domain: site.Domain, BaseURL: site.BaseURL},
			func(ctx context.Context, res *crawlermodel.FetchResult) []string {
				return r.handlePage(ctx, rn, res, log)
			})
		close(rn.jobs)
		workers.Wait()
	}

	var counters model.Counters
	if rn != nil {
		counters = rn.counters.Snapshot()
	}
	status := model.StatusCompleted
	switch {
	case ctx.Err() != nil:
		status = model.StatusCancelled
	case crawlErr != nil:
		status = model.StatusFailed
	case counters.PagesFailed > 0:
		status = model.StatusPartial
	}

	// Bookkeeping must survive shutdown, so it does not use the cancelled context.
	finishCtx, cancel := context.WithTimeout(context.WithoutCancel(ctx), 15*time.Second)
	defer cancel()
	finished := time.Now()
	var errMsg *string
	if crawlErr != nil && status != model.StatusCancelled {
		msg := crawlErr.Error()
		errMsg = &msg
	}
	if err := r.runs.Finish(finishCtx, crawlRun.ID, status, finished, counters, errMsg); err != nil {
		log.Error("could not finish crawl run", "error", err)
	}
	switch status {
	case model.StatusFailed:
		err = r.deps.Sites.RecordFailure(finishCtx, site, finished, crawlErr)
	case model.StatusCompleted, model.StatusPartial:
		err = r.deps.Sites.RecordSuccess(finishCtx, site, finished)
	default:
		err = nil // cancelled: leave the schedule as is, the site is picked up after restart
	}
	if err != nil {
		log.Error("could not update site crawl state", "error", err)
	}

	log.Info("crawl finished", "status", status, "duration", time.Since(start),
		"pages_found", counters.PagesFound, "pages_new", counters.PagesNew, "pages_changed", counters.PagesChanged,
		"pages_unchanged", counters.PagesUnchanged, "pages_skipped", counters.PagesSkipped,
		"pages_failed", counters.PagesFailed, "records_created", counters.RecordsCreated,
		"records_updated", counters.RecordsUpdated, "error", crawlErr)
}

func (r *Runner) prepare(ctx context.Context, site sitemodel.Site, runID int64) (*run, error) {
	country, err := r.deps.Countries.Get(ctx, site.CountryID)
	if err != nil {
		return nil, err
	}
	org, err := r.deps.Organizations.Get(ctx, site.OrganizationID)
	if err != nil {
		return nil, err
	}
	cats, err := r.deps.Categories.Active(ctx)
	if err != nil {
		return nil, err
	}
	entries := make([]llmmodel.Entry, len(cats))
	for i, c := range cats {
		entries[i] = llmmodel.Entry{ID: c.ID, ParentID: c.ParentID, Code: c.Code, Name: c.Name}
		if c.Description != nil {
			entries[i].Description = *c.Description
		}
	}
	taxonomy, err := llmmodel.NewTaxonomy(entries, r.deps.TaxonomySettings)
	if err != nil {
		return nil, err
	}
	return &run{
		site: site, runID: runID, country: country, org: org, taxonomy: taxonomy,
		jobs: make(chan pageJob, r.deps.LLMWorkers*2),
	}, nil
}

// handlePage runs fetch result → extract → filter → hash → dedup. Only new
// or changed text is queued for the LLM; everything else stops here.
func (r *Runner) handlePage(ctx context.Context, rn *run, res *crawlermodel.FetchResult, log *slog.Logger) []string {
	rn.counters.PagesFound.Add(1)
	log = log.With("url", res.URL)

	if res.SkipReason != "" {
		rn.counters.PagesSkipped.Add(1)
		log.Debug("page skipped", "reason", res.SkipReason)
		return nil
	}

	obs := pagemodel.Observation{SiteID: rn.site.ID, URL: res.URL, URLHash: crawlerservice.HashURL(res.URL), At: time.Now()}
	if res.StatusCode != 0 {
		status := res.StatusCode
		obs.HTTPStatus = &status
	}

	if res.Err != nil {
		rn.counters.PagesFailed.Add(1)
		log.Debug("page fetch failed", "http_status", res.StatusCode, "error", res.Err)
		if res.StatusCode != 0 {
			if p, _, err := r.pages.Observe(ctx, obs); err == nil {
				r.pages.SetStatus(ctx, p.ID, pagemodel.StatusFailed)
			}
		}
		return nil
	}

	pageURL, _ := url.Parse(res.FinalURL)
	content := extractorservice.Extract(res.Body, res.ContentType, pageURL)
	var links []string
	if !content.NoFollow {
		links = content.Links
	}
	if content.Language != "" {
		obs.LanguageCode = &content.Language
	}

	page, _, err := r.pages.Observe(ctx, obs)
	if err != nil {
		rn.counters.PagesFailed.Add(1)
		log.Error("could not store page", "error", err)
		return links
	}

	if ok, reason := r.deps.Filter.Check(content); !ok {
		rn.counters.PagesSkipped.Add(1)
		r.pages.SetStatus(ctx, page.ID, pagemodel.StatusSkipped)
		log.Debug("page text skipped", "page_id", page.ID, "reason", reason)
		return links
	}

	hash := infrastructure.SHA256(content.Text)
	change := pageservice.Classify(page.ContentHash, hash)
	switch change {
	case pageservice.ChangeUnchanged:
		rn.counters.PagesUnchanged.Add(1)
		r.pages.SetStatus(ctx, page.ID, pagemodel.StatusUnchanged)
		return links
	case pageservice.ChangeNew:
		rn.counters.PagesNew.Add(1)
	case pageservice.ChangeChanged:
		rn.counters.PagesChanged.Add(1)
	}

	select {
	case rn.jobs <- pageJob{page: page, change: change, url: res.URL, content: content, hash: hash}:
	case <-ctx.Done():
	}
	return links
}

// normalizeAndStore sends one page to the LLM and stores the validated record
// and the new page content hash in one transaction.
func (r *Runner) normalizeAndStore(ctx context.Context, rn *run, job pageJob, log *slog.Logger) {
	log = log.With("page_id", job.page.ID, "url", job.url)
	result, err := r.deps.Normalizer.Normalize(ctx, llmdto.NormalizeInput{
		Country:      llmdto.NamedRef{ID: rn.country.ID, Name: rn.country.Name},
		Organization: llmdto.NamedRef{ID: rn.org.ID, Name: rn.org.Name},
		SourceURL:    job.url,
		PageTitle:    job.content.Title,
		Content:      job.content.Text,
	}, rn.taxonomy)
	if err != nil {
		if ctx.Err() == nil {
			rn.counters.PagesFailed.Add(1)
			r.pages.SetStatus(ctx, job.page.ID, pagemodel.StatusFailed)
			log.Warn("page not normalized", "error", err)
		}
		return
	}

	rec := buildRecord(rn, job, result, r.deps.Normalizer.Runtime(), r.deps.Normalizer.Model())
	var created bool
	err = infrastructure.InTx(ctx, r.deps.DB, func(tx *sql.Tx) error {
		var err error
		if _, created, err = recordrepo.New(tx).Upsert(ctx, rec); err != nil {
			return err
		}
		return pagerepo.New(tx).MarkProcessed(ctx, job.page.ID, job.hash, job.change.Status())
	})
	if err != nil {
		if ctx.Err() == nil {
			rn.counters.PagesFailed.Add(1)
			log.Error("could not store record", "error", err)
		}
		return
	}
	if created {
		rn.counters.RecordsCreated.Add(1)
	} else {
		rn.counters.RecordsUpdated.Add(1)
	}
}

func buildRecord(rn *run, job pageJob, res llmmodel.Result, runtime, modelName string) recordmodel.Record {
	content := res.Content
	if content == "" {
		content = job.content.Text // the source text is still the best content we have
	}
	language := res.Language
	if language == "" {
		language = job.content.Language
	}
	headings := job.content.Headings
	if len(headings) > 20 {
		headings = headings[:20]
	}
	metadata, _ := json.Marshal(map[string]any{
		"source_url":   job.url,
		"page_title":   job.content.Title,
		"headings":     headings,
		"crawl_run_id": rn.runID,
		"change":       job.change.Status(),
	})
	confidence := res.Confidence
	return recordmodel.Record{
		CountryID:      rn.site.CountryID,
		OrganizationID: rn.site.OrganizationID,
		SiteID:         rn.site.ID,
		PageID:         job.page.ID,
		CategoryID:     res.CategoryID,
		SubcategoryID:  res.SubcategoryID,
		RecordType:     res.RecordType,
		Title:          nonEmpty(res.Title),
		LanguageCode:   nonEmpty(language),
		PublishedAt:    res.PublishedAt,
		Content:        content,
		Summary:        nonEmpty(res.Summary),
		Keywords:       res.Keywords,
		Metadata:       metadata,
		Confidence:     &confidence,
		ContentHash:    job.hash,
		LLMRuntime:     &runtime,
		LLMModel:       nonEmpty(modelName),
	}
}

func nonEmpty(s string) *string {
	if s == "" {
		return nil
	}
	return &s
}
