package service

import (
	"bytes"
	"compress/gzip"
	"context"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"net/url"
	"sort"
	"strings"
	"sync"

	"golang.org/x/time/rate"

	"ggkip/src/main/modules/crawler/model"
)

// maxSitemapURLs caps how many sitemap entries are kept in memory per crawl.
const maxSitemapURLs = 50_000

// Handler is called for every URL the crawler tried, from up to
// Options.Workers goroutines at once. It returns the links found on the page
// (raw hrefs resolved against the page URL are fine; the crawler normalizes them).
type Handler func(ctx context.Context, res *model.FetchResult) (links []string)

type Crawler struct {
	settings   model.Settings
	opts       model.Options
	normalizer *URLNormalizer
	log        *slog.Logger
}

func New(settings model.Settings, opts model.Options, log *slog.Logger) *Crawler {
	return &Crawler{settings: settings, opts: opts, normalizer: NewURLNormalizer(settings), log: log}
}

func (c *Crawler) Normalizer() *URLNormalizer { return c.normalizer }

// Crawl visits the site breadth-first: robots.txt, then sitemap URLs (newest
// lastmod first), then links found on pages, up to MaxPagesPerRun URLs and
// MaxDepth link hops from the base URL. It returns an error only when the site
// cannot be crawled at all; single page failures go to the handler.
func (c *Crawler) Crawl(ctx context.Context, target model.Target, handle Handler) (model.Stats, error) {
	var stats model.Stats
	base, err := url.Parse(target.BaseURL)
	if err != nil {
		return stats, fmt.Errorf("base url: %w", err)
	}
	scope := NewScope(target.Domain)
	if !scope.Allows(base) {
		return stats, fmt.Errorf("base url %s is outside domain %s", target.BaseURL, target.Domain)
	}
	baseURL, err := c.normalizer.Normalize(base.String(), nil)
	if err != nil {
		return stats, err
	}

	limiter := rate.NewLimiter(rate.Limit(c.settings.RequestsPerSecond), 1)
	fetcher := newFetcher(c.settings, c.opts, scope, c.normalizer, limiter)

	robots, err := c.loadRobots(ctx, fetcher, base)
	if err != nil {
		return stats, err
	}
	stats.RobotsLoaded = true
	if robots.CrawlDelay > 0 {
		if perSec := rate.Every(robots.CrawlDelay); perSec < limiter.Limit() {
			limiter.SetLimit(perSec)
		}
	}

	f := newFrontier(c.settings.MaxPagesPerRun)
	stop := context.AfterFunc(ctx, f.close)
	defer stop()

	enqueue := func(raw string, ref *url.URL, depth int) {
		if depth > c.settings.MaxDepth {
			return
		}
		norm, err := c.normalizer.Normalize(raw, ref)
		if err != nil {
			return
		}
		u, err := url.Parse(norm)
		if err != nil || !scope.Allows(u) || c.normalizer.IsSkippable(u) || !robots.Allowed(u) {
			return
		}
		f.push(norm, depth)
	}

	enqueue(baseURL, nil, 0)
	sitemapURLs := c.loadSitemaps(ctx, fetcher, base, robots, scope)
	stats.SitemapURLs = len(sitemapURLs)
	for _, e := range sitemapURLs {
		enqueue(e.Loc, base, 1)
	}

	var wg sync.WaitGroup
	for range c.opts.Workers {
		wg.Go(func() {
			for {
				it, ok := f.pop()
				if !ok {
					return
				}
				res := fetcher.FetchPage(ctx, it.url)
				res.Depth = it.depth
				if ctx.Err() == nil {
					links := handle(ctx, res)
					if ref, err := url.Parse(res.FinalURL); err == nil {
						for _, l := range links {
							enqueue(l, ref, it.depth+1)
						}
					}
				}
				f.done()
			}
		})
	}
	wg.Wait()
	stats.URLsQueued = f.accepted()
	return stats, ctx.Err()
}

// loadRobots follows RFC 9309: a 4xx means no restrictions, while a 5xx or an
// unreachable server means the whole site must be treated as disallowed.
func (c *Crawler) loadRobots(ctx context.Context, fetcher *Fetcher, base *url.URL) (*Robots, error) {
	robotsURL := base.ResolveReference(&url.URL{Path: "/robots.txt"}).String()
	status, body, err := fetcher.FetchRaw(ctx, robotsURL)
	switch {
	case err != nil:
		return nil, fmt.Errorf("robots.txt unreachable, not crawling: %w", err)
	case status == http.StatusOK:
		return ParseRobots(body, c.opts.UserAgent), nil
	case status >= 400 && status < 500:
		return AllowAll(), nil
	default:
		return nil, fmt.Errorf("robots.txt returned status %d, not crawling", status)
	}
}

// loadSitemaps reads sitemaps named in robots.txt (or /sitemap.xml) and
// nested sitemap indexes, returning in-scope URLs newest first.
func (c *Crawler) loadSitemaps(ctx context.Context, fetcher *Fetcher, base *url.URL, robots *Robots, scope Scope) []SitemapEntry {
	queue := robots.Sitemaps
	if len(queue) == 0 {
		queue = []string{base.ResolveReference(&url.URL{Path: "/sitemap.xml"}).String()}
	}
	seen := map[string]bool{}
	var entries []SitemapEntry
	for fetched := 0; len(queue) > 0 && fetched < c.settings.MaxSitemaps && len(entries) < maxSitemapURLs; {
		loc := queue[0]
		queue = queue[1:]
		u, err := url.Parse(loc)
		if err != nil || seen[loc] || !scope.Allows(u) {
			continue
		}
		seen[loc] = true
		fetched++

		status, body, err := fetcher.FetchRaw(ctx, loc)
		if err != nil || status != http.StatusOK {
			c.log.Debug("sitemap not loaded", "url", loc, "status", status, "error", err)
			continue
		}
		var r io.Reader = bytes.NewReader(body)
		if strings.HasSuffix(u.Path, ".gz") || (len(body) > 2 && body[0] == 0x1f && body[1] == 0x8b) {
			gz, err := gzip.NewReader(r)
			if err != nil {
				continue
			}
			r = io.LimitReader(gz, c.opts.MaxBodyBytes*4)
		}
		urls, children, err := ParseSitemap(r)
		if err != nil {
			c.log.Debug("sitemap not parsed", "url", loc, "error", err)
			continue
		}
		entries = append(entries, urls...)
		queue = append(queue, children...)
	}
	if len(entries) > maxSitemapURLs {
		entries = entries[:maxSitemapURLs]
	}
	sort.SliceStable(entries, func(i, j int) bool { return entries[i].LastMod.After(entries[j].LastMod) })
	return entries
}

type frontierItem struct {
	url   string
	depth int
}

// frontier is a FIFO queue shared by the crawl workers. It never holds more
// than max URLs in total, and pop blocks until work arrives or the crawl is done.
type frontier struct {
	mu       sync.Mutex
	cond     *sync.Cond
	queue    []frontierItem
	seen     map[string]struct{}
	max      int
	inFlight int
	closed   bool
}

func newFrontier(max int) *frontier {
	f := &frontier{seen: map[string]struct{}{}, max: max}
	f.cond = sync.NewCond(&f.mu)
	return f
}

func (f *frontier) push(u string, depth int) {
	f.mu.Lock()
	defer f.mu.Unlock()
	if f.closed || len(f.seen) >= f.max {
		return
	}
	if _, dup := f.seen[u]; dup {
		return
	}
	f.seen[u] = struct{}{}
	f.queue = append(f.queue, frontierItem{url: u, depth: depth})
	f.cond.Signal()
}

// pop returns the next URL, or false when the queue is empty and no worker
// can add more (or the crawl was cancelled).
func (f *frontier) pop() (frontierItem, bool) {
	f.mu.Lock()
	defer f.mu.Unlock()
	for len(f.queue) == 0 && f.inFlight > 0 && !f.closed {
		f.cond.Wait()
	}
	if len(f.queue) == 0 || f.closed {
		return frontierItem{}, false
	}
	it := f.queue[0]
	f.queue = f.queue[1:]
	f.inFlight++
	return it, true
}

func (f *frontier) done() {
	f.mu.Lock()
	f.inFlight--
	if f.inFlight == 0 && len(f.queue) == 0 {
		f.cond.Broadcast()
	}
	f.mu.Unlock()
}

func (f *frontier) close() {
	f.mu.Lock()
	f.closed = true
	f.cond.Broadcast()
	f.mu.Unlock()
}

func (f *frontier) accepted() int {
	f.mu.Lock()
	defer f.mu.Unlock()
	return len(f.seen)
}
