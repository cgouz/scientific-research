package service

import (
	"context"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"net/url"
	"regexp"
	"sort"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"golang.org/x/time/rate"

	"ggkip/src/main/modules/crawler/model"
)

func TestParseRobots(t *testing.T) {
	body := []byte(`
User-agent: OtherBot
Disallow: /

User-agent: *
Disallow: /private/
Allow: /private/public-page
Disallow: /*.php$
Crawl-delay: 2

Sitemap: https://example.gov/sitemap_index.xml
`)
	r := ParseRobots(body, "GGKIPBot/0.1")
	for path, want := range map[string]bool{
		"/":                    true,
		"/news/1":              true,
		"/private/":            false,
		"/private/x":           false,
		"/private/public-page": true,
		"/index.php":           false,
		"/index.php?x=1":       true, // "$" anchors the end of the whole path+query
		"/index.phpx":          true,
	} {
		u, _ := url.Parse("https://example.gov" + path)
		if got := r.Allowed(u); got != want {
			t.Errorf("Allowed(%q) = %v, want %v", path, got, want)
		}
	}
	if r.CrawlDelay != 2*time.Second {
		t.Errorf("CrawlDelay = %v, want 2s", r.CrawlDelay)
	}
	if len(r.Sitemaps) != 1 {
		t.Errorf("Sitemaps = %v", r.Sitemaps)
	}
}

func TestParseRobotsOwnGroupWins(t *testing.T) {
	r := ParseRobots([]byte("User-agent: *\nDisallow: /\n\nUser-agent: ggkipbot\nDisallow: /tmp/\n"), "GGKIPBot/0.1 (+info)")
	ok, _ := url.Parse("https://example.gov/news")
	blocked, _ := url.Parse("https://example.gov/tmp/a")
	if !r.Allowed(ok) || r.Allowed(blocked) {
		t.Fatal("the group naming our user agent must replace the * group")
	}
}

func TestParseSitemap(t *testing.T) {
	urls, children, err := ParseSitemap(strings.NewReader(`<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
  <url><loc>https://example.gov/a</loc><lastmod>2024-01-02</lastmod></url>
  <url><loc> https://example.gov/b </loc></url>
</urlset>`))
	if err != nil || len(urls) != 2 || len(children) != 0 {
		t.Fatalf("urlset: urls=%v children=%v err=%v", urls, children, err)
	}
	if urls[1].Loc != "https://example.gov/b" || urls[0].LastMod.Year() != 2024 {
		t.Fatalf("unexpected entries: %+v", urls)
	}

	_, children, err = ParseSitemap(strings.NewReader(`<sitemapindex><sitemap><loc>https://example.gov/s1.xml</loc></sitemap></sitemapindex>`))
	if err != nil || len(children) != 1 {
		t.Fatalf("index: children=%v err=%v", children, err)
	}
}

// testSite serves a small site: robots.txt, a sitemap, linked pages, a PDF,
// an image, an off-site link and a page blocked by robots.txt.
func testSite(t *testing.T) (*httptest.Server, *sync.Map) {
	t.Helper()
	hits := &sync.Map{}
	mux := http.NewServeMux()
	var srv *httptest.Server
	page := func(body string) http.HandlerFunc {
		return func(w http.ResponseWriter, r *http.Request) {
			w.Header().Set("Content-Type", "text/html; charset=utf-8")
			io.WriteString(w, "<html><body>"+body+"</body></html>")
		}
	}
	mux.HandleFunc("/robots.txt", func(w http.ResponseWriter, r *http.Request) {
		fmt.Fprintf(w, "User-agent: *\nDisallow: /secret\nSitemap: %s/sitemap.xml\n", srv.URL)
	})
	mux.HandleFunc("/sitemap.xml", func(w http.ResponseWriter, r *http.Request) {
		fmt.Fprintf(w, `<urlset><url><loc>%s/from-sitemap</loc><lastmod>2025-01-01</lastmod></url></urlset>`, srv.URL)
	})
	mux.HandleFunc("/{$}", page(`<a href="/a">A</a> <a href="b?utm_source=x">B</a> <a href="/doc.pdf">PDF</a>
		<a href="/secret">S</a> <a href="https://elsewhere.example/">off</a> <a href="/a#top">A again</a>`))
	mux.HandleFunc("/a", page(`<a href="/deep">deep</a>`))
	mux.HandleFunc("/b", page(`<p>b</p>`))
	mux.HandleFunc("/deep", page(`<p>deep</p>`))
	mux.HandleFunc("/from-sitemap", page(`<p>sitemap page</p>`))
	mux.HandleFunc("/image", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "image/png")
		w.Write([]byte{0x89, 'P', 'N', 'G'})
	})
	srv = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		v, _ := hits.LoadOrStore(r.URL.Path, new(atomic.Int32))
		v.(*atomic.Int32).Add(1)
		mux.ServeHTTP(w, r)
	}))
	t.Cleanup(srv.Close)
	return srv, hits
}

func newTestCrawler() *Crawler {
	return New(testSettings, model.Options{
		Workers: 3, Timeout: 5 * time.Second, MaxBodyBytes: 1 << 20, UserAgent: "GGKIPBot/test",
	}, slog.New(slog.NewTextHandler(io.Discard, nil)))
}

var hrefRe = regexp.MustCompile(`href="([^"]+)"`)

func crawlAll(t *testing.T, c *Crawler, srv *httptest.Server) map[string]*model.FetchResult {
	t.Helper()
	u, _ := url.Parse(srv.URL)
	var mu sync.Mutex
	results := map[string]*model.FetchResult{}
	_, err := c.Crawl(context.Background(), model.Target{Domain: u.Hostname(), BaseURL: srv.URL + "/"},
		func(ctx context.Context, res *model.FetchResult) []string {
			mu.Lock()
			results[strings.TrimPrefix(res.URL, srv.URL)] = res
			mu.Unlock()
			var links []string
			for _, m := range hrefRe.FindAllStringSubmatch(string(res.Body), -1) {
				links = append(links, m[1])
			}
			return links
		})
	if err != nil {
		t.Fatal(err)
	}
	return results
}

func TestCrawlFollowsRulesAndScope(t *testing.T) {
	srv, hits := testSite(t)
	results := crawlAll(t, newTestCrawler(), srv)

	var got []string
	for path := range results {
		got = append(got, path)
	}
	sort.Strings(got)
	want := []string{"/", "/a", "/b", "/deep", "/from-sitemap"}
	if strings.Join(got, ",") != strings.Join(want, ",") {
		t.Fatalf("crawled %v, want %v", got, want)
	}
	for _, path := range []string{"/doc.pdf", "/secret"} {
		if _, ok := hits.Load(path); ok {
			t.Errorf("%s must not be requested", path)
		}
	}
	if v, _ := hits.Load("/a"); v.(*atomic.Int32).Load() != 1 {
		t.Error("/a and /a#top are the same page and must be fetched once")
	}
	if res := results["/b"]; res.Err != nil || len(res.Body) == 0 {
		t.Errorf("/b (tracking parameter removed) should be fetched: %+v", res)
	}
}

func TestCrawlRespectsMaxPagesAndDepth(t *testing.T) {
	srv, _ := testSite(t)
	c := newTestCrawler()
	c.settings.MaxPagesPerRun = 2
	if n := len(crawlAll(t, c, srv)); n != 2 {
		t.Errorf("max_pages_per_run=2 crawled %d pages", n)
	}

	c = newTestCrawler()
	c.settings.MaxDepth = 1
	if _, ok := crawlAll(t, c, srv)["/deep"]; ok {
		t.Error("/deep is two hops away and must not be crawled with max_depth=1")
	}
}

func TestFetchSkipsNonTextContent(t *testing.T) {
	srv, _ := testSite(t)
	c := newTestCrawler()
	u, _ := url.Parse(srv.URL)
	f := newFetcher(c.settings, c.opts, NewScope(u.Hostname()), c.normalizer, unlimited())
	res := f.FetchPage(context.Background(), srv.URL+"/image")
	if res.SkipReason != model.SkipContentType || res.Body != nil {
		t.Fatalf("image should be skipped by content type: %+v", res)
	}
	res = f.FetchPage(context.Background(), srv.URL+"/missing")
	if res.Err == nil || res.StatusCode != http.StatusNotFound {
		t.Fatalf("404 should be an error with status: %+v", res)
	}
}

func TestFetchRetriesTransientErrors(t *testing.T) {
	var calls atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if calls.Add(1) < 2 {
			w.WriteHeader(http.StatusServiceUnavailable)
			return
		}
		w.Header().Set("Content-Type", "text/plain")
		io.WriteString(w, "ok")
	}))
	defer srv.Close()
	c := newTestCrawler()
	c.settings.MaxRetries = 2
	u, _ := url.Parse(srv.URL)
	f := newFetcher(c.settings, c.opts, NewScope(u.Hostname()), c.normalizer, unlimited())
	f.retryDelay = time.Millisecond
	res := f.FetchPage(context.Background(), srv.URL+"/")
	if res.Err != nil || string(res.Body) != "ok" || calls.Load() != 2 {
		t.Fatalf("expected success after one retry: calls=%d res=%+v", calls.Load(), res)
	}
}

func TestFetchDecodesCharset(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/html; charset=shift_jis")
		w.Write([]byte{0x93, 0xfa, 0x96, 0x7b}) // 日本 in Shift_JIS
	}))
	defer srv.Close()
	c := newTestCrawler()
	u, _ := url.Parse(srv.URL)
	f := newFetcher(c.settings, c.opts, NewScope(u.Hostname()), c.normalizer, unlimited())
	if res := f.FetchPage(context.Background(), srv.URL+"/"); string(res.Body) != "日本" {
		t.Fatalf("body = %q, want 日本", res.Body)
	}
}

func TestCrawlStopsWhenRobotsUnavailable(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
	}))
	defer srv.Close()
	u, _ := url.Parse(srv.URL)
	_, err := newTestCrawler().Crawl(context.Background(), model.Target{Domain: u.Hostname(), BaseURL: srv.URL + "/"},
		func(context.Context, *model.FetchResult) []string { t.Error("no page may be fetched"); return nil })
	if err == nil {
		t.Fatal("a 5xx robots.txt must stop the crawl")
	}
}

func unlimited() *rate.Limiter { return rate.NewLimiter(rate.Inf, 1) }
