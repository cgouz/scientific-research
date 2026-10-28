package service

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"io"
	"mime"
	"net"
	"net/http"
	"net/url"
	"strings"
	"time"

	"golang.org/x/net/html/charset"
	"golang.org/x/time/rate"

	"ggkip/src/main/modules/crawler/model"
)

var (
	errOutOfScope      = errors.New("redirect leaves the site")
	errTooManyRedirect = errors.New("too many redirects")
)

// Fetcher does rate-limited HTTP GETs for one crawl.
type Fetcher struct {
	client     *http.Client
	limiter    *rate.Limiter
	userAgent  string
	maxBody    int64
	maxRetries int
	allowed    map[string]bool
	normalizer *URLNormalizer
	retryDelay time.Duration // multiplied by the attempt number
}

func newFetcher(settings model.Settings, opts model.Options, scope Scope, normalizer *URLNormalizer, limiter *rate.Limiter) *Fetcher {
	allowed := map[string]bool{}
	for _, t := range settings.AllowedContentTypes {
		allowed[strings.ToLower(t)] = true
	}
	maxRedirects := settings.MaxRedirects
	transport := &http.Transport{
		Proxy:                 http.ProxyFromEnvironment,
		DialContext:           (&net.Dialer{Timeout: 10 * time.Second, KeepAlive: 30 * time.Second}).DialContext,
		MaxIdleConnsPerHost:   opts.Workers,
		IdleConnTimeout:       90 * time.Second,
		TLSHandshakeTimeout:   10 * time.Second,
		ResponseHeaderTimeout: opts.Timeout,
	}
	return &Fetcher{
		client: &http.Client{
			Timeout:   opts.Timeout,
			Transport: transport,
			CheckRedirect: func(req *http.Request, via []*http.Request) error {
				if len(via) > maxRedirects {
					return errTooManyRedirect
				}
				if !scope.Allows(req.URL) {
					return errOutOfScope
				}
				return nil
			},
		},
		limiter:    limiter,
		userAgent:  opts.UserAgent,
		maxBody:    opts.MaxBodyBytes,
		maxRetries: settings.MaxRetries,
		allowed:    allowed,
		normalizer: normalizer,
		retryDelay: 2 * time.Second,
	}
}

func (f *Fetcher) contentTypeAllowed(mediaType string) bool {
	return f.allowed[mediaType] || (f.allowed["application/json"] && strings.HasSuffix(mediaType, "+json"))
}

// FetchPage downloads a page if its content type is textual. The body is
// converted to UTF-8 using the declared or detected charset.
func (f *Fetcher) FetchPage(ctx context.Context, pageURL string) *model.FetchResult {
	res := &model.FetchResult{URL: pageURL, FinalURL: pageURL}
	start := time.Now()
	defer func() { res.Duration = time.Since(start) }()

	resp, err := f.get(ctx, pageURL, "text/html,application/xhtml+xml,text/plain;q=0.9,application/json;q=0.8")
	if err != nil {
		if errors.Is(err, errOutOfScope) {
			res.SkipReason = model.SkipOutOfScope
		} else {
			res.Err = err
		}
		return res
	}
	defer resp.Body.Close()

	res.StatusCode = resp.StatusCode
	if final, err := f.normalizer.Normalize(resp.Request.URL.String(), nil); err == nil {
		res.FinalURL = final
	}
	if resp.StatusCode != http.StatusOK {
		res.Err = fmt.Errorf("http status %d", resp.StatusCode)
		return res
	}

	header := resp.Header.Get("Content-Type")
	mediaType, _, _ := mime.ParseMediaType(header)
	res.ContentType = strings.ToLower(mediaType)
	if !f.contentTypeAllowed(res.ContentType) {
		res.SkipReason = model.SkipContentType
		return res
	}

	raw, err := io.ReadAll(io.LimitReader(resp.Body, f.maxBody+1))
	if err != nil {
		res.Err = fmt.Errorf("read body: %w", err)
		return res
	}
	if int64(len(raw)) > f.maxBody {
		res.SkipReason = model.SkipTooLarge
		return res
	}
	if strings.HasPrefix(res.ContentType, "text/") || res.ContentType == "application/xhtml+xml" {
		if r, err := charset.NewReader(bytes.NewReader(raw), header); err == nil {
			if decoded, err := io.ReadAll(r); err == nil {
				raw = decoded
			}
		}
	}
	res.Body = raw
	return res
}

// FetchRaw downloads robots.txt or a sitemap without content type checks.
// It returns the status code and the body (empty unless status is 200).
func (f *Fetcher) FetchRaw(ctx context.Context, rawURL string) (int, []byte, error) {
	resp, err := f.get(ctx, rawURL, "*/*")
	if err != nil {
		return 0, nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		io.Copy(io.Discard, io.LimitReader(resp.Body, 64<<10))
		return resp.StatusCode, nil, nil
	}
	body, err := io.ReadAll(io.LimitReader(resp.Body, f.maxBody+1))
	if err != nil {
		return resp.StatusCode, nil, err
	}
	if int64(len(body)) > f.maxBody {
		return resp.StatusCode, nil, fmt.Errorf("%s: body larger than %d bytes", rawURL, f.maxBody)
	}
	return resp.StatusCode, body, nil
}

// get performs the request with rate limiting and retries for transient
// failures (network errors, 429 and 5xx). The caller closes the body.
func (f *Fetcher) get(ctx context.Context, rawURL, accept string) (*http.Response, error) {
	var lastErr error
	for attempt := 0; attempt <= f.maxRetries; attempt++ {
		if attempt > 0 {
			backoff := time.Duration(attempt) * f.retryDelay
			select {
			case <-ctx.Done():
				return nil, ctx.Err()
			case <-time.After(backoff):
			}
		}
		if err := f.limiter.Wait(ctx); err != nil {
			return nil, err
		}
		req, err := http.NewRequestWithContext(ctx, http.MethodGet, rawURL, nil)
		if err != nil {
			return nil, err
		}
		req.Header.Set("User-Agent", f.userAgent)
		req.Header.Set("Accept", accept)

		resp, err := f.client.Do(req)
		if err != nil {
			if ctx.Err() != nil {
				return nil, ctx.Err()
			}
			var urlErr *url.Error
			if errors.As(err, &urlErr) && (errors.Is(urlErr.Err, errOutOfScope) || errors.Is(urlErr.Err, errTooManyRedirect)) {
				return nil, urlErr.Err
			}
			lastErr = err
			continue
		}
		if resp.StatusCode == http.StatusTooManyRequests || resp.StatusCode >= 500 {
			io.Copy(io.Discard, io.LimitReader(resp.Body, 64<<10))
			resp.Body.Close()
			lastErr = fmt.Errorf("http status %d", resp.StatusCode)
			continue
		}
		return resp, nil
	}
	return nil, lastErr
}
